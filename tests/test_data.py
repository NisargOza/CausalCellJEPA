# Tests pin the transformations and leakage boundaries before real data is admitted.
import numpy as np
from scipy import sparse

from causalcelljepa.data import (
    assign_roles,
    eligible_targets,
    fit_hvgs,
    fit_hvgs_stream,
    normalize_log1p,
    representation_fit_mask,
    required_embedding_mask,
    target_manifest,
)
from causalcelljepa.external import tokenize_normalized_cell


def test_normalization_preserves_sparse_input_and_library_size():
    counts = sparse.csr_matrix([[1, 1, 2], [0, 3, 1]])
    normalized = normalize_log1p(counts)
    assert sparse.isspmatrix_csr(normalized)
    np.testing.assert_allclose(np.expm1(normalized.toarray()).sum(axis=1), 10_000, rtol=1e-6)


def test_external_normalized_tokenization_preserves_frozen_vocabulary_positions():
    values = np.array([0.5, 3.0, 3.0, 0.0], dtype=np.float32)
    positions = np.array([9, 7, 2, 4])
    genes, expression, padding = tokenize_normalized_cell(
        values, positions, vocab_size=10, max_tokens=3
    )
    assert genes.tolist() == [2, 7, 9]
    assert expression.tolist() == [3.0, 3.0, 0.5]
    assert not padding.any()


def test_hvg_fit_does_not_see_heldout_outcome():
    counts = np.array([[1, 5, 0], [9, 5, 0], [1, 5, 0], [1, 5, 10_000]])
    assert fit_hvgs(counts, [True, True, False, False], n_genes=1).tolist() == [0]
    assert fit_hvgs(counts, [True, True, True, True], n_genes=1).tolist() == [2]
    streamed, diagnostics, fit_cells = fit_hvgs_stream(
        [(counts, np.ones(4, bool), np.arange(3), counts.sum(axis=1) + np.arange(4))],
        1,
        block_size=2,
    )
    assert streamed.tolist() == [2]
    assert diagnostics == [
        {
            "min_count": 0.0,
            "max_count": 10_000.0,
            "zero_library_cells": 0,
            "max_umi_excluded_by_gene_filter": 3.0,
        }
    ]
    assert fit_cells == 4


def test_target_manifest_is_deterministic_disjoint_and_hashed():
    targets = [f"G{i}" for i in range(10)]
    first = target_manifest(targets, seed=17)
    second = target_manifest(reversed(targets), seed=17)
    assert first == second
    assert tuple(map(len, first["targets"].values())) == (7, 1, 2)
    groups = [set(group) for group in first["targets"].values()]
    assert set.union(*groups) == set(targets)
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    assert len(first["sha256"]) == 64


def test_cell_roles_encode_all_four_regimes_without_leakage():
    target_names = [f"G{i}" for i in range(10)]
    cell_ids, targets, contexts, controls = [], [], [], []
    for context in ("K562", "RPE1"):
        for target in target_names:
            for cell in range(63 if (context, target) == ("RPE1", "G9") else 64):
                cell_ids.append(f"{context}-{target}-{cell}")
                targets.append(target)
                contexts.append(context)
                controls.append(False)
        for cell in range(64):
            cell_ids.append(f"{context}-control-{cell}")
            targets.append("NTC")
            contexts.append(context)
            controls.append(True)
    arrays = tuple(map(np.asarray, (cell_ids, targets, contexts, controls)))
    eligible = eligible_targets(arrays[1], arrays[2], arrays[3])
    assert eligible == tuple(target_names[:-1])
    manifest = target_manifest(eligible, seed=17)
    roles = assign_roles(*arrays, manifest)
    assert set(roles) == {
        "control_train",
        "control_inference",
        "dynamics_train",
        "iid_test",
        "perturbation_ood_validation",
        "perturbation_ood_test",
        "context_ood_test",
        "double_ood_validation_locked",
        "double_ood_test",
        "excluded",
    }
    dynamics = roles == "dynamics_train"
    assert set(arrays[1][dynamics]) == set(manifest["targets"]["train"])
    assert set(arrays[2][dynamics]) == {"K562"}
    fit = representation_fit_mask(roles)
    assert not np.any(fit & np.char.endswith(roles.astype(str), "test"))
    assert not np.any(fit & ((arrays[2] == "RPE1") & ~arrays[3]))
    required = required_embedding_mask(roles)
    assert required.sum() == np.count_nonzero(roles != "excluded")
    assert not required[roles == "excluded"].any()
