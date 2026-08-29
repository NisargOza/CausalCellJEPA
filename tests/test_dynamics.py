import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from causalcelljepa.dynamics import (
    AnchoredPopulationDynamics,
    LatentPopulationDataset,
    ModalityAttentiveActionProjection,
    PopulationDynamics,
    anchored_selected_entry,
    dynamics_loss,
    dynamics_objective,
    dynamics_replication_configs,
    select_multiteacher_candidate,
)
from causalcelljepa.evaluation import (
    paired_condition_comparisons,
    paired_model_comparisons,
    population_metrics,
)


def test_modality_attention_fuses_all_teachers_and_dropout_keeps_one_visible():
    torch.manual_seed(5)
    projection = ModalityAttentiveActionProjection([3, 2], 4, modality_dropout=0.9)
    action = torch.randn(16, 5)
    train = projection.train()(action)
    evaluation = projection.eval()(action)
    assert train.shape == evaluation.shape == (16, 4)
    assert torch.isfinite(train).all() and torch.isfinite(evaluation).all()


def test_anchored_selection_entry_locks_candidate_without_test_leakage():
    artifact = {"bytes": 7, "path": "best.pt", "sha256": "abc"}
    run = {"best_validation_epoch": 4, "best_validation_loss": 0.25}
    training = {
        "protocol": {
            "sealed_test_outcomes_used_for_fit_or_selection": False,
            "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        },
        "artifacts": {"candidates": {"anchor_only": {"best_checkpoint": artifact, "full_run": run}}},
    }
    selection = {
        "leakage": {
            "context": "K562",
            "outcome_role": "perturbation_ood_validation",
            "rpe1_outcomes_used": False,
            "sealed_test_outcomes_used": False,
        },
        "selected": {
            "candidate": "anchor_only",
            "best_validation_epoch": 4,
            "best_validation_loss": 0.25,
            **artifact,
        },
    }
    assert anchored_selected_entry(training, selection) == (
        "anchor_only",
        training["artifacts"]["candidates"]["anchor_only"],
    )
    selection["leakage"]["sealed_test_outcomes_used"] = True
    with pytest.raises(AssertionError):
        anchored_selected_entry(training, selection)


def test_multiteacher_selection_requires_frozen_dropout_margin():
    training = {
        "protocol": {
            "sealed_test_outcomes_used_for_fit_or_selection": False,
            "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        },
        "artifacts": {
            "candidates": {
                "attention_full": {"full_run": {"best_validation_loss": 0.690}},
                "attention_dropout_025": {"full_run": {"best_validation_loss": 0.6849}},
            }
        },
    }
    specification = {
        "experiments": {
            "attention_full": {"action_modality_dropout": 0.0},
            "attention_dropout_025": {"action_modality_dropout": 0.25},
        },
        "selection": {
            "checkpoint_rule": "minimum_original_latent_validation_loss",
            "dropout_minimum_loss_improvement": 0.005,
            "fallback_candidate": "attention_full",
            "viewed_test_outcomes_used": False,
        },
    }
    selected, improvement = select_multiteacher_candidate(training, specification)
    assert selected == "attention_dropout_025"
    assert improvement == pytest.approx(0.0051)
    training["artifacts"]["candidates"]["attention_dropout_025"]["full_run"][
        "best_validation_loss"
    ] = 0.6851
    selected, improvement = select_multiteacher_candidate(training, specification)
    assert selected == "attention_full"
    assert improvement == pytest.approx(0.0049)


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


def test_anchored_dynamics_preserves_mean_prior_and_bounds_action_correction():
    torch.manual_seed(21)
    anchor = {
        "format_version": 1,
        "architecture": "esm2_low_rank_latent_effect_ridge",
        "x_mean": torch.zeros(3),
        "x_std": torch.ones(3),
        "y_mean": torch.tensor([0.1, -0.2, 0.3, -0.1]),
        "components": torch.eye(4),
        "weights": torch.tensor(
            [[0.2, 0.1, 0.0, -0.1], [0.0, 0.2, -0.1, 0.1], [0.1, 0.0, 0.2, 0.0]]
        ),
        "report": {"training_null_effect_threshold": 0.05},
    }
    model = AnchoredPopulationDynamics(
        cell_dim=4,
        action_input_dim=3,
        action_dim=4,
        context_blocks=1,
        transition_blocks=1,
        heads=2,
        ffn_dim=8,
        dropout=0.0,
        effect_anchor=anchor,
        mean_residual_max_ratio=0.25,
    ).eval()
    with torch.no_grad():
        model.delta[-1].weight.normal_()
        model.mean_residual[-1].bias.fill_(10)
    control = torch.randn(2, 5, 4)
    action = torch.tensor([[1.0, 0.5, -0.5], [0.2, -0.3, 0.7]])
    known = torch.tensor([True, False])
    predicted = model(control, action, known)
    frozen = model.effect_anchor(action, known)
    correction = (predicted - control).mean(1) - frozen
    maximum = 0.25 * torch.linalg.vector_norm(frozen, dim=1).clamp_min(0.05)
    assert torch.all(torch.linalg.vector_norm(correction, dim=1) <= maximum + 1e-6)
    heterogeneity = predicted - control - frozen[:, None] - correction[:, None]
    assert torch.allclose(heterogeneity.mean(1), torch.zeros(2, 4), atol=1e-6)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(control[:, permutation], action, known)
    assert torch.allclose(permuted, predicted[:, permutation], atol=1e-5, rtol=1e-5)
    assert torch.allclose(frozen[1], anchor["y_mean"])

    gain_model = AnchoredPopulationDynamics(
        cell_dim=4,
        action_input_dim=3,
        action_dim=4,
        context_blocks=1,
        transition_blocks=1,
        heads=2,
        ffn_dim=8,
        dropout=0.0,
        effect_anchor=anchor,
        anchor_gain_max=4.0,
        mean_residual_max_ratio=0.0,
    ).eval()
    with torch.no_grad():
        gain_model.anchor_gain[-1].bias.fill_(100)
    gained = (gain_model(control, action, known) - control).mean(1)
    ratio = torch.linalg.vector_norm(gained, dim=1) / torch.linalg.vector_norm(frozen, dim=1)
    assert torch.all(ratio <= 4.0 + 1e-6)
    assert torch.allclose(
        torch.nn.functional.cosine_similarity(gained, frozen), torch.ones(2), atol=1e-6
    )


def test_control_ood_gate_preserves_means_and_suppresses_only_ood_heterogeneity():
    torch.manual_seed(13)
    anchor = {
        "format_version": 1,
        "architecture": "esm2_low_rank_latent_effect_ridge",
        "x_mean": torch.zeros(3),
        "x_std": torch.ones(3),
        "y_mean": torch.ones(4),
        "components": torch.eye(4),
        "weights": torch.zeros(3, 4),
        "report": {"training_null_effect_threshold": 0.05},
    }
    model = AnchoredPopulationDynamics(
        cell_dim=4, action_input_dim=3, action_dim=4, context_blocks=1,
        transition_blocks=1, heads=2, ffn_dim=8, dropout=0.0, effect_anchor=anchor,
    ).eval()
    with torch.no_grad():
        model.delta[-1].weight.normal_()
    population = torch.tensor([
        [-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0],
        [-0.5, 0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
    ])
    control = torch.stack((population, population + 3))
    action, known = torch.zeros(2, 3), torch.ones(2, dtype=torch.bool)
    ungated = model(control, action, known)
    model.configure_residual_gate({
        "format_version": 1, "architecture": "control_population_residual_gate",
        "center": torch.zeros(4), "scale": torch.ones(4), "threshold": 1.0,
        "temperature": 0.5,
    })
    gated = model(control, action, known)
    confidence = model.residual_gate_confidence(control)
    assert torch.equal(confidence[0], torch.tensor(1.0)) and confidence[1] < 1e-6
    assert torch.allclose(gated[0], ungated[0])
    assert torch.allclose((gated - control).mean(1), (ungated - control).mean(1), atol=1e-6)
    assert torch.linalg.vector_norm(gated[1] - control[1] - 1) < 1e-5


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


def test_stage2_replications_change_only_model_sampling_seed_and_output():
    configs, specification = dynamics_replication_configs()
    base = yaml.safe_load(Path(specification["base_config_path"]).read_text())
    assert set(configs) == {20260824, 20260825}
    for seed, config in configs.items():
        assert config["seed"] == config["replication"]["model_and_sampling_seed"] == seed
        assert config["replication"]["target_split_seed"] == base["seed"] == 20260823
        for section in ("inputs", "data", "normalization", "model", "loss"):
            assert config[section] == base[section]
        assert config["training"] == {
            **base["training"],
            "output_directory": f"artifacts/replications/dynamics_seed_{seed}",
            "resume_from": None,
        }


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
