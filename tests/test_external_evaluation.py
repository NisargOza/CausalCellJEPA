import json

import h5py
import numpy as np
import torch

from causalcelljepa.external_evaluation import NadigLatentPopulationDataset


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
