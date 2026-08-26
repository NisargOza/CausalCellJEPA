import json
from collections import Counter

import h5py
import numpy as np
import torch

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    PopulationDynamics,
    dynamics_loss,
    dynamics_objective,
)
from causalcelljepa.evaluation import (
    paired_condition_comparisons,
    paired_model_comparisons,
    population_metrics,
)


def test_population_sampling_is_independent_deterministic_and_batch_matched(tmp_path):
    cache_path = tmp_path / "latents.h5"
    controls = 8
    roles = ["control_train"] * controls + ["dynamics_train"] * 4 + [
        "perturbation_ood_validation"
    ] * 4
    targets = ["non-targeting"] * controls + ["T1"] * 4 + ["T2"] * 4
    batches = ["A"] * 4 + ["B"] * 4 + ["A", "A", "B", "B"] * 2
    with h5py.File(cache_path, "w") as cache:
        string = h5py.string_dtype("utf-8")
        cache.create_dataset("latent", data=np.arange(64, dtype=np.float32).reshape(16, 4))
        for name, values in {
            "role": roles,
            "target": targets,
            "source_batch": batches,
            "context": ["K562"] * 16,
        }.items():
            cache.create_dataset(name, data=values, dtype=string)
    action_path = tmp_path / "actions.pt"
    torch.save(
        {
            "targets": ["T1", "T2"],
            "embedding": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "known": torch.ones(2, dtype=torch.bool),
        },
        action_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "normalization": {
                    "latent_mean": [0.0] * 4,
                    "latent_std": [1.0] * 4,
                    "dimension_scale": 2.0,
                },
                "conditions": {
                    "train": {"role": "dynamics_train", "targets": 1},
                    "validation": {"role": "perturbation_ood_validation", "targets": 1},
                    "controls": {"role": "control_train"},
                },
            }
        )
    )
    dataset = LatentPopulationDataset(cache_path, action_path, manifest_path, "train", 2, 7)
    control_indices, outcome_indices, target = dataset.sample_indices(0)
    repeated = dataset.sample_indices(0)
    assert target == "T1" and all(np.array_equal(a, b) for a, b in zip(repeated[:2], (control_indices, outcome_indices)))
    assert not set(control_indices) & set(outcome_indices)
    assert Counter(dataset.batches[control_indices]) == Counter(dataset.batches[outcome_indices])
    sample = dataset[0]
    assert sample["control"].shape == sample["perturbed"].shape == (2, 4)
    assert sample["action"].shape == (3,) and bool(sample["action_known"])
    explicit = LatentPopulationDataset(
        cache_path,
        action_path,
        manifest_path,
        "custom_evaluation",
        2,
        7,
        "perturbation_ood_validation",
        "control_train",
        "K562",
    )
    assert explicit.condition_targets == ["T2"]


def test_dynamics_is_set_equivariant_and_distribution_loss_has_finite_gradients():
    torch.manual_seed(3)
    model = PopulationDynamics(
        cell_dim=8,
        action_input_dim=5,
        action_dim=8,
        context_blocks=2,
        transition_blocks=3,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    ).eval()
    control = torch.randn(2, 4, 8)
    action = torch.randn(2, 5)
    known = torch.tensor([True, False])
    permutation = torch.tensor([2, 0, 3, 1])
    predicted = model(control, action, known)
    assert torch.equal(predicted, control)
    permuted = model(control[:, permutation], action, known)
    assert torch.allclose(permuted, predicted[:, permutation], atol=1e-6, rtol=1e-6)

    loss_config = {
        "sinkhorn_blur_ratio": 0.1,
        "sinkhorn_p": 2,
        "sinkhorn_debias": True,
        "sinkhorn_backend": "tensorized",
        "mmd_bandwidth_ratio": 1.0,
        "weights": {"sinkhorn": 1.0, "mmd": 0.25, "direction": 0.5, "magnitude": 0.1},
    }
    identical = dynamics_loss(predicted, predicted[:, permutation], control, loss_config, 1.0, 0.01)
    assert float(identical["loss"].detach()) < 1e-5
    train_model = PopulationDynamics(
        cell_dim=8,
        action_input_dim=5,
        action_dim=8,
        context_blocks=1,
        transition_blocks=1,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    )
    predicted = train_model(control, action, known)
    losses = dynamics_loss(predicted, torch.randn_like(predicted), control, loss_config, 1.0, 0.1)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in train_model.parameters())


def test_context_ablation_modes_remove_or_simplify_the_global_summary():
    torch.manual_seed(12)
    control, action, known = torch.randn(2, 4, 8), torch.randn(2, 5), torch.ones(2, dtype=torch.bool)
    interactions = {}
    for mode in ("set_transformer", "mean", "none"):
        model = PopulationDynamics(
            cell_dim=8,
            action_input_dim=5,
            action_dim=8,
            context_blocks=1,
            transition_blocks=1,
            heads=2,
            ffn_dim=16,
            dropout=0.0,
            context_mode=mode,
        ).eval()
        model.interaction[0].register_forward_pre_hook(
            lambda _module, inputs, mode=mode: (
                interactions.__setitem__(mode, inputs[0].detach()),
                None,
            )[1]
        )
        model(control, action, known)
    assert torch.count_nonzero(interactions["none"][:, :8]) == 0
    assert torch.count_nonzero(interactions["none"][:, 16:24]) == 0
    assert torch.count_nonzero(interactions["mean"][:, :8]) > 0
    assert torch.count_nonzero(interactions["set_transformer"][:, :8]) > 0


def test_population_evaluation_metrics_are_exact_for_identical_populations():
    torch.manual_seed(9)
    control = torch.randn(2, 4, 8)
    observed = control + torch.randn(2, 1, 8)
    metrics = population_metrics(
        observed,
        observed,
        control,
        1.0,
        {"reference_sinkhorn_blur_ratio": 0.1, "mmd_bandwidth_ratio": 1.0},
    )
    for name in ("sinkhorn", "mmd", "energy_distance", "centroid_shift_error", "covariance_shift_error"):
        assert torch.allclose(metrics[name], torch.zeros(2), atol=1e-6)
    assert torch.allclose(metrics["effect_pearson"], torch.ones(2), atol=1e-6)
    assert torch.allclose(metrics["effect_spearman"], torch.ones(2), atol=1e-6)
    assert torch.allclose(metrics["magnitude_ratio"], torch.ones(2), atol=1e-6)


def test_pseudo_paired_objective_is_pointwise_and_pairing_dependent():
    predicted = torch.tensor([[[0.0], [1.0], [3.0]]], requires_grad=True)
    observed = torch.tensor([[[0.0], [2.0], [4.0]]])
    config = {"objective": "pseudo_paired_mse"}
    direct = dynamics_objective(predicted, observed, predicted.detach(), config, {})
    permuted = dynamics_objective(
        predicted, observed[:, torch.tensor([2, 0, 1])], predicted.detach(), config, {}
    )
    assert torch.equal(direct["loss"], direct["pointwise_mse"])
    assert not torch.equal(direct["loss"], permuted["loss"])
    direct["loss"].backward()
    assert torch.isfinite(predicted.grad).all()


def test_paired_comparisons_orient_improvement_and_adjust_multiplicity():
    records = []
    for target in range(12):
        for model, pearson, sinkhorn, ratio in (
            ("causalcelljepa", 0.8, 0.2, 0.9),
            ("linear_esm", 0.4, 0.5, 0.5),
        ):
            records.append(
                {
                    "regime": "double_ood",
                    "context": "RPE1",
                    "outcome_role": "double_ood_test",
                    "target": str(target),
                    "repeat": 0,
                    "model": model,
                    "effect_pearson": pearson,
                    "sinkhorn": sinkhorn,
                    "magnitude_ratio": ratio,
                }
            )
    comparisons = paired_condition_comparisons(records, 100, 3)
    assert len(comparisons) == 3
    assert all(comparison["targets"] == 12 for comparison in comparisons)
    assert all(comparison["mean_improvement"] > 0 for comparison in comparisons)
    assert all(
        comparison["wilcoxon_two_sided_p"] <= comparison["benjamini_hochberg_q"] <= 1
        for comparison in comparisons
    )


def test_arbitrary_model_comparisons_use_matched_targets_and_metric_orientation():
    records = []
    for target in range(12):
        for model, pearson, sinkhorn, ratio in (
            ("ablation", 0.8, 0.2, 0.9),
            ("reference", 0.4, 0.5, 0.5),
        ):
            records.append(
                {
                    "regime": "double_ood",
                    "context": "RPE1",
                    "outcome_role": "double_ood_test",
                    "target": str(target),
                    "repeat": 0,
                    "model": model,
                    "effect_pearson": pearson,
                    "sinkhorn": sinkhorn,
                    "magnitude_ratio": ratio,
                }
            )
    comparisons = paired_model_comparisons(
        records,
        [
            {
                "candidate": "ablation",
                "reference": "reference",
                "hypothesis": "synthetic orientation check",
            }
        ],
        100,
        3,
    )
    assert len(comparisons) == 3
    assert all(comparison["targets"] == 12 for comparison in comparisons)
    assert all(comparison["mean_improvement"] > 0 for comparison in comparisons)
    assert {comparison["candidate"] for comparison in comparisons} == {"ablation"}
    assert {comparison["reference"] for comparison in comparisons} == {"reference"}
