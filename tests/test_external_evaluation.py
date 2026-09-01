import gzip
import json
from hashlib import sha256

import h5py
import numpy as np
import torch
import yaml
from scipy import sparse

from causalcelljepa.external import prepare_adamson_external
from causalcelljepa.external_evaluation import (
    NadigLatentPopulationDataset,
    _centroid_accuracy,
    grouped_expression_moments,
    load_adamson_expression,
    run_adamson_external_evaluation,
)


def test_adamson_preparation_freezes_metadata_without_opening_matrix(tmp_path):
    root = tmp_path / "adamson"
    root.mkdir()
    files = {
        "matrix": ("matrix.mtx.gz", b"sealed-expression-outcomes"),
        "barcodes": ("barcodes.tsv.gz", b"c1-1\nc2-1\nc3-1\nc4-1\n"),
        "genes": ("genes.tsv.gz", b"ENSG1\tG1\nENSG2\tG2\n"),
        "identities": (
            "identities.csv.gz",
            (
                b"cell BC,guide identity,good coverage,number of cells\n"
                b"c1-1,ctrl1,TRUE,1\nc2-1,ctrl2,TRUE,1\n"
                b"c3-1,A_g1,TRUE,1\nc4-1,B_g1,TRUE,1\n"
            ),
        ),
    }
    source = {}
    for name, (filename, content) in files.items():
        path = root / filename
        if name == "matrix":
            path.write_bytes(content)
        else:
            with gzip.open(path, "wb") as handle:
                handle.write(content)
        source[name] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    action_path = tmp_path / "action.pt"
    torch.save({"targets": ["A"], "known": torch.tensor([True])}, action_path)
    replogle = {"genes": {"hvg_gene_ids": ["ENSG1", "ENSG2"]}}
    replogle["manifest_sha256"] = sha256(
        json.dumps(replogle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    replogle_path = tmp_path / "replogle.json"
    replogle_path.write_text(json.dumps(replogle))
    output = tmp_path / "manifest.json"
    config = {
        "source": {"raw_directory": str(root), "files": source},
        "metadata_audit": {
            "barcodes": 4,
            "genes": 2,
            "unique_gene_symbols": 2,
            "identity_rows": 4,
            "valid_cells": 4,
            "control_cells": 2,
            "scored_target_cells": 1,
            "reference_only_target_cells": 1,
            "unknown_identity_cells_excluded": 0,
            "minimum_batch_matched_controls": 2,
        },
        "filtering": {
            "good_coverage": True,
            "number_of_cells": 1,
            "unknown_identity": "*",
            "controls": ["ctrl1", "ctrl2"],
            "minimum_cells_per_target": 1,
        },
        "targets": {"scored": ["A"], "systematic_reference_only": ["B"]},
        "frozen_candidate": {
            "action_path": str(action_path),
            "action_bytes": action_path.stat().st_size,
            "action_sha256": sha256(action_path.read_bytes()).hexdigest(),
            "replogle_manifest": str(replogle_path),
            "replogle_manifest_sha256": replogle["manifest_sha256"],
        },
        "outputs": {"preparation_manifest": str(output)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    report = prepare_adamson_external(config_path)
    assert report["cohort"]["scored_targets"] == ["A"]
    assert report["cohort"]["reference_only_targets"] == ["B"]
    assert report["cohort"]["frozen_hvg_overlap"] == 2
    assert report["leakage"]["expression_matrix_decompressed_during_preparation"] is False


def test_adamson_expression_uses_full_library_and_exact_ensembl_order(tmp_path):
    root = tmp_path / "adamson"
    root.mkdir()
    contents = {
        "barcodes": ("barcodes.tsv.gz", "c1-1\nc2-1\nc3-2\nc4-2\n"),
        "genes": ("genes.tsv.gz", "ENSG1\tDUP\nENSGX\tDUP\nENSG2\tG2\n"),
        "identities": (
            "identities.csv.gz",
            (
                "cell BC,guide identity,good coverage,number of cells\n"
                "c1-1,ctrl,TRUE,1\nc2-1,ctrl,TRUE,1\n"
                "c3-2,A_g1,TRUE,1\nc4-2,B_g1,TRUE,1\n"
            ),
        ),
        "matrix": (
            "matrix.mtx.gz",
            (
                "%%MatrixMarket matrix coordinate integer general\n% test\n3 4 12\n"
                "1 1 1\n2 1 9\n3 1 2\n"
                "1 2 3\n2 2 9\n3 2 4\n"
                "1 3 5\n2 3 9\n3 3 6\n"
                "1 4 7\n2 4 9\n3 4 8\n"
            ),
        ),
    }
    source = {}
    for name, (filename, content) in contents.items():
        path = root / filename
        with gzip.open(path, "wt") as handle:
            handle.write(content)
        source[name] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    preregistered = {
        "source": {"raw_directory": str(root), "files": source},
        "filtering": {
            "good_coverage": True,
            "number_of_cells": 1,
            "controls": ["ctrl"],
            "unknown_identity": "*",
        },
        "targets": {"scored": ["A"], "systematic_reference_only": ["B"]},
    }
    expression, metadata, observed = load_adamson_expression(
        preregistered, ["ENSG2", "ENSG1"], {"control", "scored", "reference"}
    )
    expected = np.log1p(
        10_000
        * np.asarray([[2, 1], [4, 3], [6, 5], [8, 7]], dtype=np.float32)
        / np.asarray([12, 16, 20, 24], dtype=np.float32)[:, None]
    )
    assert np.allclose(expression.toarray(), expected)
    assert metadata["target"].tolist() == ["control", "control", "A", "B"]
    assert metadata["batch"].tolist() == ["1", "1", "2", "2"]
    assert np.array_equal(observed, [0, 1])


def test_external_population_sampling_is_deterministic_and_unpaired(tmp_path):
    cache_path = tmp_path / "external.h5"
    string = h5py.string_dtype("utf-8")
    controls = 40
    outcomes = 64
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset(
            "latent",
            data=np.arange((controls + outcomes) * 4, dtype=np.float32).reshape(-1, 4),
        )
        cache.create_dataset(
            "role",
            data=np.asarray(
                ["external_control"] * controls + ["external_test"] * outcomes,
                dtype=object,
            ),
            dtype=string,
        )
        cache.create_dataset(
            "context",
            data=np.asarray(["HepG2"] * (controls + outcomes), dtype=object),
            dtype=string,
        )
        cache.create_dataset(
            "target",
            data=np.asarray(["control"] * controls + ["A"] * 32 + ["B"] * 32, dtype=object),
            dtype=string,
        )
        cache.create_dataset(
            "source_batch",
            data=np.asarray(["unavailable"] * (controls + outcomes), dtype=object),
            dtype=string,
        )
        cache.create_dataset(
            "cell_id",
            data=np.asarray(
                [f"cell-{index}" for index in range(controls + outcomes)], dtype=object
            ),
            dtype=string,
        )
    action_path = tmp_path / "actions.pt"
    torch.save(
        {
            "targets": ["A", "B"],
            "embedding": torch.arange(12, dtype=torch.float32).reshape(2, 6),
            "known": torch.ones(2, dtype=torch.bool),
        },
        action_path,
    )
    manifest_path = tmp_path / "dynamics.json"
    manifest_path.write_text(
        json.dumps(
            {
                "normalization": {
                    "latent_mean": [0.0] * 4,
                    "latent_std": [1.0] * 4,
                    "dimension_scale": 1.0,
                }
            }
        )
    )
    dataset = NadigLatentPopulationDataset(
        cache_path,
        action_path=action_path,
        dynamics_manifest_path=manifest_path,
        context="HepG2",
        population_size=16,
        seed=7,
        expected_targets=["A", "B"],
    )
    first = dataset.sample_indices(0)
    assert np.array_equal(first[0], dataset.sample_indices(0)[0])
    assert np.array_equal(first[1], dataset.sample_indices(0)[1])
    assert set(first[0]).issubset(set(range(controls)))
    assert set(first[1]).issubset(set(range(controls, controls + 32)))
    assert set(first[0]).isdisjoint(first[1])
    dataset.set_epoch(1)
    second = dataset.sample_indices(0)
    assert not np.array_equal(first[0], second[0])
    assert not np.array_equal(first[1], second[1])
    item = dataset[0]
    assert item["control"].shape == item["perturbed"].shape == (16, 4)
    assert item["target"] == "A" and bool(item["action_known"])


def test_external_expression_moments_stream_sparse_unpaired_groups():
    matrix = sparse.csr_matrix(
        np.asarray([[1, 2, 8], [3, 4, 9], [5, 6, 7], [7, 8, 6]], dtype=np.float32)
    )
    means, variances, counts = grouped_expression_moments(
        matrix, np.asarray(["control", "A", "ignored", "A"]), ["control", "A"], [0, 1], 2
    )
    assert np.array_equal(counts, [1, 2])
    assert np.array_equal(means, [[1, 2], [5, 6]])
    assert np.array_equal(variances[1], [8, 8])


def test_systema_centroid_accuracy_uses_other_truth_centroids():
    truth = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    assert _centroid_accuracy(np.asarray([0.1, 0.1]), 0, truth) == 1.0
    assert _centroid_accuracy(np.asarray([1.1, 0.0]), 0, truth) == 0.5


def test_adamson_confirmation_runs_once_end_to_end_on_synthetic_outcomes(tmp_path):
    root = tmp_path / "adamson"
    root.mkdir()
    barcodes = [
        "ctrl1-1",
        "ctrl2-2",
        "a1-1",
        "a2-2",
        "c1-1",
        "c2-2",
        "b1-1",
        "b2-2",
        "b3-1",
        "d1-1",
        "d2-2",
    ]
    guides = [
        "ctrl",
        "ctrl",
        "A_g1",
        "A_g1",
        "C_g1",
        "C_g1",
        "B_g1",
        "B_g1",
        "B_g1",
        "D_g1",
        "D_g1",
    ]
    counts = np.asarray(
        [
            [5, 3, 2],
            [4, 4, 2],
            [10, 2, 1],
            [9, 3, 1],
            [2, 10, 1],
            [3, 9, 1],
            [7, 5, 1],
            [6, 6, 1],
            [9, 2, 2],
            [6, 4, 3],
            [5, 5, 3],
        ],
        dtype=int,
    )
    raw = {
        "barcodes": ("barcodes.tsv.gz", "\n".join(barcodes) + "\n"),
        "genes": ("genes.tsv.gz", "E1\tG1\nE2\tG2\nE3\tG3\n"),
        "identities": (
            "identities.csv.gz",
            "cell BC,guide identity,good coverage,number of cells\n"
            + "".join(
                f"{cell},{guide},TRUE,1\n" for cell, guide in zip(barcodes, guides, strict=True)
            ),
        ),
        "matrix": (
            "matrix.mtx.gz",
            "%%MatrixMarket matrix coordinate integer general\n3 11 33\n"
            + "".join(
                f"{gene + 1} {cell + 1} {counts[cell, gene]}\n"
                for cell in range(11)
                for gene in range(3)
            ),
        ),
    }
    source = {}
    for name, (filename, content) in raw.items():
        path = root / filename
        with gzip.open(path, "wt") as handle:
            handle.write(content)
        source[name] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    preregistered = {
        "seed": 11,
        "source": {"raw_directory": str(root), "files": source, "cell_context": "K562"},
        "filtering": {
            "good_coverage": True,
            "number_of_cells": 1,
            "controls": ["ctrl"],
            "unknown_identity": "*",
        },
        "targets": {
            "scored": ["A", "C"],
            "outcome_fit_seen": ["A"],
            "outcome_fit_unseen": ["C"],
            "systematic_reference_only": ["B", "D"],
        },
        "frozen_candidate": {},
    }
    preregistered_path = tmp_path / "preregistered.yaml"
    replogle = {"genes": {"hvg_gene_ids": ["E1", "E2", "E3"], "hvg_gene_names": ["G1", "G2", "G3"]}}
    replogle["manifest_sha256"] = sha256(
        json.dumps(replogle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    preregistered["frozen_candidate"]["replogle_manifest_sha256"] = replogle["manifest_sha256"]
    preregistered_path.write_text(yaml.safe_dump(preregistered))
    replogle_path = tmp_path / "replogle.json"
    replogle_path.write_text(json.dumps(replogle))
    go_path = tmp_path / "go.gmt"
    go_path.write_text(
        "".join(f"P{index}\tdescription\tG{index % 3 + 1}\n" for index in range(4328))
    )
    transcriptomics_path = tmp_path / "transcriptomics.yaml"
    transcriptomics_path.write_text(
        yaml.safe_dump(
            {"inputs": {"replogle_manifest_path": str(replogle_path), "go_gmt_path": str(go_path)}}
        )
    )
    preparation = {"source": {"config_sha256": sha256(preregistered_path.read_bytes()).hexdigest()}}
    preparation["manifest_sha256"] = sha256(
        json.dumps(preparation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    prediction_path, prediction_manifest_path = (
        tmp_path / "prediction.pt",
        tmp_path / "prediction.json",
    )
    effects = {
        "control_gated_external_response": [[0.3, -0.2, 0.0], [-0.2, 0.3, 0.0]],
        "external_response_multiview_rbf": [[0.2, -0.1, 0.0], [-0.1, 0.2, 0.0]],
        "string_kernel_gene_go_rbf": [[0.4, -0.3, 0.0], [-0.3, 0.4, 0.0]],
        "direct_gene_esm": [[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]],
        "mean_effect": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "no_change": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
    torch.save(
        {
            "targets": ["A", "C"],
            "hvg_gene_ids": ["E1", "E2", "E3"],
            "hvg_gene_names": ["G1", "G2", "G3"],
            "observed_hvg_positions": torch.arange(3),
            "effects": {name: torch.tensor(value) for name, value in effects.items()},
            "leakage": {"roles_read": ["control"], "perturbed_outcomes_used": False},
        },
        prediction_path,
    )
    output, evaluation_manifest_path = tmp_path / "evaluation", tmp_path / "evaluation.json"
    config = {
        "inputs": {
            "preregistered_config_path": str(preregistered_path),
            "preregistered_config_sha256": sha256(preregistered_path.read_bytes()).hexdigest(),
            "preparation_manifest_path": str(preparation_path),
            "preparation_manifest_sha256": preparation["manifest_sha256"],
            "transcriptomics_config_path": str(transcriptomics_path),
            "transcriptomics_config_sha256": sha256(transcriptomics_path.read_bytes()).hexdigest(),
        },
        "metrics": {
            "bootstrap_resamples": 100,
            "deg_batch_fdr": 0.05,
            "deg_min_abs_effect": 0.1,
            "retrospective_top_genes": [2],
            "pathway_top_k": [2],
        },
        "comparisons": [
            {
                "candidate": "control_gated_external_response",
                "reference": reference,
                "hypothesis": "synthetic integration",
            }
            for reference in [
                "perturbed_mean",
                "external_response_multiview_rbf",
                "string_kernel_gene_go_rbf",
            ]
        ],
        "outputs": {
            "prediction_manifest_path": str(prediction_manifest_path),
            "evaluation_directory": str(output),
            "evaluation_manifest_path": str(evaluation_manifest_path),
        },
    }
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prediction_manifest = {
        "artifact": {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path.read_bytes()).hexdigest(),
        },
        "source": {"config_sha256": sha256(config_path.read_bytes()).hexdigest()},
    }
    prediction_manifest["manifest_sha256"] = sha256(
        json.dumps(prediction_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    prediction_manifest_path.write_text(json.dumps(prediction_manifest))
    _, _, truth, decision, provenance = run_adamson_external_evaluation(config_path)
    assert truth["scored_targets"] == 2 and provenance["full_adamson_evaluation_index"] == 1
    assert decision["terminal_next_step"] in {
        "run_one_locked_systema_frontier_comparison_then_finalize",
        "stop_architecture_search_and_finalize_mixed_or_negative_result",
    }
    assert evaluation_manifest_path.exists()
