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
    grouped_expression_moments,
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
    replogle = {"genes": {"hvg_gene_names": ["G1", "G2"]}}
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
            data=np.asarray(
                ["control"] * controls + ["A"] * 32 + ["B"] * 32, dtype=object
            ),
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
