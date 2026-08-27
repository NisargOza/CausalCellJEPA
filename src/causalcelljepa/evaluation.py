# Condition-level latent evaluation and mandatory simple baselines for the four OOD regimes.
# Only K562 training/validation outcomes are read while fitting or selecting baselines.
import json
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import torch
from geomloss import SamplesLoss
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    anchored_selected_entry,
    build_dynamics_model,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


def _role_effects(config, roles):
    inputs = config["inputs"]
    manifest = json.loads(Path(inputs["dynamics_manifest_path"]).read_text())
    normalization = manifest["normalization"]
    mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
    scale = np.asarray(normalization["latent_std"], dtype=np.float32) * normalization[
        "dimension_scale"
    ]
    with h5py.File(inputs["latent_cache_path"], "r") as cache:
        cache_roles = cache["role"].asstr()[:]
        targets = cache["target"].asstr()[:]
        batches = cache["source_batch"].asstr()[:]
        controls = np.flatnonzero(cache_roles == "control_train")
        control_means = {
            batch: ((cache["latent"][controls[batches[controls] == batch]] - mean) / scale).mean(0)
            for batch in sorted(set(batches[controls]))
        }
        effects = {}
        for role in roles:
            role_indices = np.flatnonzero(cache_roles == role)
            effects[role] = {}
            for target in sorted(set(targets[role_indices])):
                indices = role_indices[targets[role_indices] == target]
                outcome = ((cache["latent"][indices] - mean) / scale).mean(0)
                counts = {batch: np.count_nonzero(batches[indices] == batch) for batch in set(batches[indices])}
                control = sum(count * control_means[batch] for batch, count in counts.items()) / len(indices)
                effects[role][target] = outcome - control
    return effects, normalization


def fit_linear_baseline(config):
    """Fit a low-rank ESM-to-effect ridge baseline and tune alpha on K562 validation."""
    effects, _ = _role_effects(config, ("dynamics_train", "perturbation_ood_validation"))
    action = torch.load(config["inputs"]["action_cache_path"], map_location="cpu", weights_only=True)
    action_map = {
        target: (action["embedding"][index].numpy(), bool(action["known"][index]))
        for index, target in enumerate(action["targets"])
    }
    train = effects["dynamics_train"]
    known_targets = [target for target in sorted(train) if action_map[target][1]]
    x = np.stack([action_map[target][0] for target in known_targets]).astype(np.float64)
    y = np.stack([train[target] for target in known_targets]).astype(np.float64)
    x_mean, x_std = x.mean(0), x.std(0).clip(1e-8)
    y_mean = y.mean(0)
    _, _, components = np.linalg.svd(y - y_mean, full_matrices=False)
    components = components[: config["baselines"]["linear_rank"]]
    scores = (y - y_mean) @ components.T
    standardized = (x - x_mean) / x_std
    gram, cross = standardized.T @ standardized, standardized.T @ scores
    validation = effects["perturbation_ood_validation"]
    validation_targets = sorted(validation)
    validation_x = np.stack([action_map[target][0] for target in validation_targets])
    validation_y = np.stack([validation[target] for target in validation_targets])
    candidates = []
    for alpha in config["baselines"]["ridge_candidates"]:
        weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        prediction = ((validation_x - x_mean) / x_std) @ weights @ components + y_mean
        candidates.append((float(np.mean((prediction - validation_y) ** 2)), alpha, weights))
    validation_mse, alpha, weights = min(candidates, key=lambda item: (item[0], item[1]))
    predictions = {
        target: (
            ((embedding - x_mean) / x_std) @ weights @ components + y_mean
            if known
            else y_mean
        ).astype(np.float32)
        for target, (embedding, known) in action_map.items()
    }
    return predictions, np.mean(np.stack(list(train.values())), axis=0).astype(np.float32), {
        "fit_outcome_role": "dynamics_train",
        "selection_outcome_role": "perturbation_ood_validation",
        "fit_targets": len(train),
        "fit_targets_with_known_action": len(known_targets),
        "selection_targets": len(validation_targets),
        "rank": len(components),
        "selected_ridge": alpha,
        "selection_mse": validation_mse,
        "ridge_candidates": [float(item[1]) for item in candidates],
        "ridge_validation_mse": [float(item[0]) for item in candidates],
    }


def population_metrics(predicted, observed, control, median_distance, config):
    """Return one value per perturbation-condition for all locked latent metrics."""
    sinkhorn = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=config["reference_sinkhorn_blur_ratio"] * median_distance,
        debias=True,
        backend="tensorized",
    )(predicted, observed).reshape(-1)
    bandwidth = config["mmd_bandwidth_ratio"] * median_distance
    xx = torch.exp(-torch.cdist(predicted, predicted).square() / (2 * bandwidth**2)).mean((1, 2))
    yy = torch.exp(-torch.cdist(observed, observed).square() / (2 * bandwidth**2)).mean((1, 2))
    xy = torch.exp(-torch.cdist(predicted, observed).square() / (2 * bandwidth**2)).mean((1, 2))
    true_effect = observed.mean(1) - control.mean(1)
    predicted_effect = predicted.mean(1) - control.mean(1)
    true_magnitude = torch.linalg.vector_norm(true_effect, dim=1)
    predicted_magnitude = torch.linalg.vector_norm(predicted_effect, dim=1)
    centered_true = true_effect - true_effect.mean(1, keepdim=True)
    centered_predicted = predicted_effect - predicted_effect.mean(1, keepdim=True)
    denominator = torch.linalg.vector_norm(centered_true, dim=1) * torch.linalg.vector_norm(
        centered_predicted, dim=1
    )
    pearson = (centered_true * centered_predicted).sum(1) / denominator
    true_rank = true_effect.argsort(1).argsort(1).float()
    predicted_rank = predicted_effect.argsort(1).argsort(1).float()
    true_rank -= true_rank.mean(1, keepdim=True)
    predicted_rank -= predicted_rank.mean(1, keepdim=True)
    rank_denominator = torch.linalg.vector_norm(true_rank, dim=1) * torch.linalg.vector_norm(
        predicted_rank, dim=1
    )
    spearman = (true_rank * predicted_rank).sum(1) / rank_denominator
    invalid = (denominator <= 1e-12) | (predicted_effect.std(1) <= 1e-12)
    pearson[invalid], spearman[invalid] = torch.nan, torch.nan
    predicted_covariance = torch.bmm(
        (predicted - predicted.mean(1, keepdim=True)).transpose(1, 2),
        predicted - predicted.mean(1, keepdim=True),
    ) / (predicted.shape[1] - 1)
    observed_covariance = torch.bmm(
        (observed - observed.mean(1, keepdim=True)).transpose(1, 2),
        observed - observed.mean(1, keepdim=True),
    ) / (observed.shape[1] - 1)
    return {
        "effect_pearson": pearson,
        "effect_spearman": spearman,
        "direction_cosine": torch.nn.functional.cosine_similarity(predicted_effect, true_effect),
        "magnitude_ratio": predicted_magnitude / true_magnitude.clamp_min(1e-12),
        "magnitude_absolute_error": torch.abs(predicted_magnitude - true_magnitude),
        "sinkhorn": sinkhorn,
        "mmd": (xx + yy - 2 * xy).clamp_min(0),
        "energy_distance": (
            2 * torch.cdist(predicted, observed).mean((1, 2))
            - torch.cdist(predicted, predicted).mean((1, 2))
            - torch.cdist(observed, observed).mean((1, 2))
        ).clamp_min(0),
        "centroid_shift_error": torch.linalg.vector_norm(predicted_effect - true_effect, dim=1),
        "covariance_shift_error": torch.linalg.matrix_norm(
            predicted_covariance - observed_covariance, ord="fro"
        ),
    }


def paired_condition_comparisons(records, bootstrap_resamples, seed):
    """Compare models on matched targets with paired bootstrap CIs and Wilcoxon tests."""
    excluded = {"regime", "context", "outcome_role", "target", "repeat", "model"}
    metric_names = sorted(set(records[0]) - excluded)
    higher = {"effect_pearson", "effect_spearman", "direction_cosine"}
    closer = {"magnitude_ratio"}
    target_values = defaultdict(list)
    for record in records:
        for metric in metric_names:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    averaged = {key: float(np.mean(values)) for key, values in target_values.items()}
    comparisons = []
    regimes = sorted({record["regime"] for record in records})
    for regime in regimes:
        for baseline in ("no_change", "mean_effect", "linear_esm", "pseudo_paired"):
            for metric in metric_names:
                targets = sorted(
                    {
                        key[2]
                        for key in averaged
                        if key[0] == regime
                        and key[1] == "causalcelljepa"
                        and key[3] == metric
                        and (regime, baseline, key[2], metric) in averaged
                    }
                )
                if not targets:
                    continue
                model = np.asarray(
                    [averaged[regime, "causalcelljepa", target, metric] for target in targets]
                )
                reference = np.asarray(
                    [averaged[regime, baseline, target, metric] for target in targets]
                )
                if metric in higher:
                    improvement, direction = model - reference, "higher_is_better"
                elif metric in closer:
                    improvement, direction = (
                        np.abs(reference - 1) - np.abs(model - 1),
                        "closer_to_one_is_better",
                    )
                else:
                    improvement, direction = reference - model, "lower_is_better"
                generator = np.random.default_rng(
                    (int.from_bytes(sha256(f"{regime}\0{baseline}\0{metric}".encode()).digest()[:8], "little") + seed)
                    % (1 << 64)
                )
                bootstrap = improvement[
                    generator.integers(0, len(improvement), (bootstrap_resamples, len(improvement)))
                ].mean(1)
                p_value = (
                    1.0
                    if np.all(improvement == 0)
                    else float(wilcoxon(improvement, alternative="two-sided", method="approx").pvalue)
                )
                comparisons.append(
                    {
                        "regime": regime,
                        "baseline": baseline,
                        "metric": metric,
                        "direction": direction,
                        "targets": len(targets),
                        "causalcelljepa_mean": float(model.mean()),
                        "baseline_mean": float(reference.mean()),
                        "mean_improvement": float(improvement.mean()),
                        "median_improvement": float(np.median(improvement)),
                        "mean_improvement_bootstrap_95ci": [
                            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                        ],
                        "wilcoxon_two_sided_p": p_value,
                    }
                )
    p_values = np.asarray([comparison["wilcoxon_two_sided_p"] for comparison in comparisons])
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    for comparison, q_value in zip(comparisons, q_values):
        comparison["benjamini_hochberg_q"] = float(q_value)
    return comparisons


def paired_model_comparisons(records, model_pairs, bootstrap_resamples, seed):
    """Compare arbitrary learned models on matched perturbation-condition units."""
    excluded = {"regime", "context", "outcome_role", "target", "repeat", "model"}
    metric_names = sorted(set(records[0]) - excluded)
    higher = {"effect_pearson", "effect_spearman", "direction_cosine"}
    closer = {"magnitude_ratio"}
    target_values = defaultdict(list)
    for record in records:
        for metric in metric_names:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    averaged = {key: float(np.mean(values)) for key, values in target_values.items()}
    comparisons = []
    regimes = sorted({record["regime"] for record in records})
    for regime in regimes:
        for pair in model_pairs:
            candidate, reference = pair["candidate"], pair["reference"]
            for metric in metric_names:
                targets = sorted(
                    {
                        key[2]
                        for key in averaged
                        if key[0] == regime
                        and key[1] == candidate
                        and key[3] == metric
                        and (regime, reference, key[2], metric) in averaged
                    }
                )
                if not targets:
                    continue
                candidate_values = np.asarray(
                    [averaged[regime, candidate, target, metric] for target in targets]
                )
                reference_values = np.asarray(
                    [averaged[regime, reference, target, metric] for target in targets]
                )
                if metric in higher:
                    improvement = candidate_values - reference_values
                    direction = "higher_is_better"
                elif metric in closer:
                    improvement = np.abs(reference_values - 1) - np.abs(candidate_values - 1)
                    direction = "closer_to_one_is_better"
                else:
                    improvement = reference_values - candidate_values
                    direction = "lower_is_better"
                key = f"{regime}\0{candidate}\0{reference}\0{metric}"
                generator = np.random.default_rng(
                    (int.from_bytes(sha256(key.encode()).digest()[:8], "little") + seed)
                    % (1 << 64)
                )
                bootstrap = improvement[
                    generator.integers(
                        0, len(improvement), (bootstrap_resamples, len(improvement))
                    )
                ].mean(1)
                p_value = (
                    1.0
                    if np.all(improvement == 0)
                    else float(
                        wilcoxon(improvement, alternative="two-sided", method="approx").pvalue
                    )
                )
                comparisons.append(
                    {
                        "regime": regime,
                        "candidate": candidate,
                        "reference": reference,
                        "hypothesis": pair["hypothesis"],
                        "metric": metric,
                        "direction": direction,
                        "targets": len(targets),
                        "candidate_mean": float(candidate_values.mean()),
                        "reference_mean": float(reference_values.mean()),
                        "mean_improvement": float(improvement.mean()),
                        "median_improvement": float(np.median(improvement)),
                        "mean_improvement_bootstrap_95ci": [
                            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                        ],
                        "wilcoxon_two_sided_p": p_value,
                    }
                )
    if not comparisons:
        return comparisons
    p_values = np.asarray([comparison["wilcoxon_two_sided_p"] for comparison in comparisons])
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    for comparison, q_value in zip(comparisons, q_values):
        comparison["benjamini_hochberg_q"] = float(q_value)
    return comparisons


def _condition_metric_summaries(records, bootstrap_resamples, seed):
    excluded = {"regime", "context", "outcome_role", "target", "repeat", "model"}
    metric_names = sorted(set(records[0]) - excluded)
    target_values = defaultdict(list)
    for record in records:
        for metric in metric_names:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    grouped = defaultdict(list)
    for (regime, model, _target, metric), values in target_values.items():
        grouped[regime, model, metric].append(float(np.mean(values)))
    summaries = []
    for key, values in sorted(grouped.items()):
        regime, model, metric = key
        values = np.asarray(values)
        generator = np.random.default_rng(
            int.from_bytes(sha256("\0".join(key).encode()).digest()[:8], "little") + seed
        )
        bootstrap = values[
            generator.integers(0, len(values), (bootstrap_resamples, len(values)))
        ].mean(1)
        summaries.append(
            {
                "regime": regime,
                "model": model,
                "metric": metric,
                "targets": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "mean_bootstrap_95ci": [
                    float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                ],
            }
        )
    return summaries


def _retrieval_summaries(signatures, regimes, models):
    summaries = []
    for regime in regimes:
        targets = sorted(key[2] for key in signatures if key[:2] == (regime, "true"))
        truth = np.stack(
            [signatures[regime, "true", target][1] / signatures[regime, "true", target][0]
             for target in targets]
        )
        truth /= np.linalg.norm(truth, axis=1, keepdims=True).clip(1e-12)
        for model in models:
            predicted = np.stack(
                [signatures[regime, model, target][1] / signatures[regime, model, target][0]
                 for target in targets]
            )
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True).clip(1e-12)
            similarity = predicted @ truth.T
            ranks = np.asarray(
                [
                    1
                    + np.count_nonzero(row > row[index])
                    + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                    for index, row in enumerate(similarity)
                ]
            )
            summaries.append(
                {
                    "regime": regime,
                    "model": model,
                    "targets": len(targets),
                    "top_1": float(np.mean(ranks <= 1)),
                    "top_5": float(np.mean(ranks <= 5)),
                    "mean_reciprocal_rank": float(np.mean(1 / ranks)),
                    "median_rank": float(np.median(ranks)),
                }
            )
    return summaries


def _self_hashed_manifest(path, expected_sha256):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    computed = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert declared == expected_sha256 == computed
    return payload, declared


@torch.no_grad()
def run_evaluation(config, regimes=None, repeats=None, max_conditions=None, output_directory=None):
    """Run deterministic population evaluation and summarize at the target-condition level."""
    inputs = config["inputs"]
    for kind in ("latent_cache", "action_cache", "checkpoint", "pseudo_paired_checkpoint"):
        assert file_sha256(inputs[f"{kind}_path"]) == inputs[f"{kind}_sha256"]
    selection = json.loads(Path(inputs["selection_manifest_path"]).read_text())
    declared = selection.pop("manifest_sha256")
    assert declared == inputs["selection_manifest_sha256"] == sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint = torch.load(inputs["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["configuration"]["loss"]["sinkhorn_blur_ratio"] == selection["selected"][
        "blur_ratio"
    ]
    model = build_dynamics_model(checkpoint["configuration"]).eval()
    model.load_state_dict(checkpoint["model"])
    pseudo_manifest = json.loads(Path(inputs["pseudo_paired_manifest_path"]).read_text())
    pseudo_declared = pseudo_manifest.pop("manifest_sha256")
    assert pseudo_declared == inputs["pseudo_paired_manifest_sha256"] == sha256(
        json.dumps(pseudo_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert (
        pseudo_manifest["artifacts"]["best_checkpoint"]["sha256"]
        == inputs["pseudo_paired_checkpoint_sha256"]
    )
    pseudo_checkpoint = torch.load(
        inputs["pseudo_paired_checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert pseudo_checkpoint["configuration"]["objective"] == "pseudo_paired_mse"
    pseudo_model = build_dynamics_model(pseudo_checkpoint["configuration"]).eval()
    pseudo_model.load_state_dict(pseudo_checkpoint["model"])
    linear_effect, mean_effect, baseline_report = fit_linear_baseline(config)
    dynamics_manifest = json.loads(Path(inputs["dynamics_manifest_path"]).read_text())
    median_distance = dynamics_manifest["normalization"]["median_training_latent_distance"]
    regimes = regimes or config["regimes"]
    repeats = repeats or config["sampling"]["repeats"]
    output = Path(output_directory or config["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    records, signatures = [], defaultdict(lambda: [0, None])
    for regime, specification in regimes.items():
        dataset = LatentPopulationDataset(
            inputs["latent_cache_path"],
            inputs["action_cache_path"],
            inputs["dynamics_manifest_path"],
            regime,
            config["sampling"]["population_size"],
            config["seed"],
            specification["outcome_role"],
            specification["control_role"],
            specification["context"],
        )
        indices = range(min(len(dataset), max_conditions or len(dataset)))
        for repeat in range(repeats):
            dataset.set_epoch(repeat)
            loader = DataLoader(
                dataset,
                batch_size=config["sampling"]["batch_size"],
                sampler=list(indices),
                num_workers=config["sampling"]["num_workers"],
            )
            for batch in loader:
                control, observed = batch["control"], batch["perturbed"]
                predictions = {
                    "causalcelljepa": model(control, batch["action"], batch["action_known"]),
                    "pseudo_paired": pseudo_model(
                        control, batch["action"], batch["action_known"]
                    ),
                    "no_change": control,
                    "mean_effect": control + torch.from_numpy(mean_effect)[None, None],
                    "linear_esm": control
                    + torch.from_numpy(np.stack([linear_effect[target] for target in batch["target"]]))[
                        :, None
                    ],
                }
                assert all(predicted.dtype == observed.dtype for predicted in predictions.values())
                true_effect = observed.mean(1) - control.mean(1)
                for index, target in enumerate(batch["target"]):
                    key = (regime, "true", target)
                    signatures[key][0] += 1
                    value = true_effect[index].numpy()
                    signatures[key][1] = (
                        value if signatures[key][1] is None else signatures[key][1] + value
                )
                for baseline, predicted in predictions.items():
                    predicted_effect = (predicted.mean(1) - control.mean(1)).detach()
                    metrics = population_metrics(
                        predicted, observed, control, median_distance, config["metrics"]
                    )
                    for index, target in enumerate(batch["target"]):
                        record = {
                            "regime": regime,
                            "context": specification["context"],
                            "outcome_role": specification["outcome_role"],
                            "target": target,
                            "repeat": repeat,
                            "model": baseline,
                        }
                        for name, values in metrics.items():
                            value = float(values[index].detach())
                            record[name] = value if np.isfinite(value) else None
                        records.append(record)
                        key = (regime, baseline, target)
                        signatures[key][0] += 1
                        value = predicted_effect[index].numpy()
                        signatures[key][1] = (
                            value if signatures[key][1] is None else signatures[key][1] + value
                        )
    with (output / "condition_metrics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    target_values = defaultdict(list)
    metric_names = [key for key in records[0] if key not in {"regime", "context", "outcome_role", "target", "repeat", "model"}]
    for record in records:
        for metric in metric_names:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    summary = {"condition_metrics": [], "retrieval": []}
    grouped = defaultdict(list)
    for (regime, baseline, target, metric), values in target_values.items():
        grouped[regime, baseline, metric].append(float(np.mean(values)))
    for key, values in sorted(grouped.items()):
        regime, baseline, metric = key
        values = np.asarray(values)
        generator = np.random.default_rng(
            int.from_bytes(sha256("\0".join(key).encode()).digest()[:8], "little") + config["seed"]
        )
        bootstrap = values[
            generator.integers(0, len(values), (config["metrics"]["bootstrap_resamples"], len(values)))
        ].mean(1)
        summary["condition_metrics"].append(
            {
                "regime": regime,
                "model": baseline,
                "metric": metric,
                "targets": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "mean_bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
            }
        )
    for regime in regimes:
        targets = sorted({key[2] for key in signatures if key[0] == regime})
        truth = np.stack([signatures[regime, "true", target][1] / signatures[regime, "true", target][0] for target in targets])
        truth /= np.linalg.norm(truth, axis=1, keepdims=True).clip(1e-12)
        for baseline in sorted({record["model"] for record in records}):
            predicted = np.stack([signatures[regime, baseline, target][1] / signatures[regime, baseline, target][0] for target in targets])
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True).clip(1e-12)
            similarity = predicted @ truth.T
            ranks = np.asarray(
                [
                    1 + np.count_nonzero(row > row[index]) + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                    for index, row in enumerate(similarity)
                ]
            )
            summary["retrieval"].append(
                {
                    "regime": regime,
                    "model": baseline,
                    "targets": len(targets),
                    "top_1": float(np.mean(ranks <= 1)),
                    "top_5": float(np.mean(ranks <= 5)),
                    "mean_reciprocal_rank": float(np.mean(1 / ranks)),
                    "median_rank": float(np.median(ranks)),
                }
            )
    provenance = {
        "config": deepcopy(config),
        "executed_regimes": regimes,
        "executed_repeats": repeats,
        "maximum_conditions_per_regime": max_conditions,
        "checkpoint_provenance": checkpoint["provenance"],
        "pseudo_paired_checkpoint_provenance": pseudo_checkpoint["provenance"],
        "file_sha256": {
            kind: inputs[f"{kind}_sha256"]
            for kind in ("latent_cache", "action_cache", "checkpoint", "pseudo_paired_checkpoint")
        },
        "selection_manifest_sha256": declared,
        "pseudo_paired_manifest_sha256": pseudo_declared,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "baseline_report.json").write_text(json.dumps(baseline_report, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return summary, baseline_report, provenance


@torch.no_grad()
def run_ablation_evaluation(
    config,
    base_config,
    regimes=None,
    repeats=None,
    max_conditions=None,
    output_directory=None,
):
    """Evaluate frozen mechanism ablations under the already locked four-regime protocol."""
    inputs = config["inputs"]
    assert file_sha256(config["base_evaluation_config_path"]) == config[
        "base_evaluation_config_sha256"
    ]
    base_manifest, base_manifest_sha256 = _self_hashed_manifest(
        inputs["base_evaluation_manifest_path"],
        inputs["base_evaluation_manifest_sha256"],
    )
    model_source = config.get("model_source", "ablation")
    manifest_kinds = {
        "ablation": "ablation",
        "stage2_replication": "replication_training",
        "anchored": "anchored_training",
    }
    assert model_source in manifest_kinds
    manifest_kind = manifest_kinds[model_source]
    model_manifest, model_manifest_sha256 = _self_hashed_manifest(
        inputs[f"{manifest_kind}_manifest_path"],
        inputs[f"{manifest_kind}_manifest_sha256"],
    )
    selected_candidate = selected_entry = selection_manifest_sha256 = None
    if model_source == "anchored":
        selection_manifest, selection_manifest_sha256 = _self_hashed_manifest(
            inputs["anchored_selection_manifest_path"],
            inputs["anchored_selection_manifest_sha256"],
        )
        selected_candidate, selected_entry = anchored_selected_entry(
            model_manifest, selection_manifest
        )
        expected = "anchored_control_ood_gated" if "residual_gate" in config else "anchored_selected"
        assert config["models"] == [expected]
    for kind in ("base_condition_metrics", "base_summary", "base_provenance"):
        path = Path(inputs[f"{kind}_path"])
        assert path.stat().st_size == inputs[f"{kind}_bytes"]
        assert file_sha256(path) == inputs[f"{kind}_sha256"]
    assert (
        base_manifest["artifacts"]["condition_metrics"]["sha256"]
        == inputs["base_condition_metrics_sha256"]
    )
    assert (
        base_manifest["artifacts"]["condition_metrics"]["records"]
        == inputs["base_condition_metrics_records"]
    )
    assert base_manifest["artifacts"]["summary"]["sha256"] == inputs["base_summary_sha256"]
    assert (
        base_manifest["artifacts"]["provenance"]["sha256"]
        == inputs["base_provenance_sha256"]
    )
    base_provenance = json.loads(Path(inputs["base_provenance_path"]).read_text())
    assert base_provenance["config"] == base_config
    assert base_provenance["git"]["dirty"] is False

    models, checkpoint_provenance = {}, {}
    for name in config["models"]:
        if model_source == "ablation":
            entry = model_manifest["experiments"][name]
        elif model_source == "stage2_replication":
            seed = str(config["model_seeds"][name])
            entry = model_manifest["artifacts"]["seeds"][seed]
        else:
            entry = selected_entry
        artifact = entry["best_checkpoint"]
        checkpoint_path = Path(artifact["path"])
        assert checkpoint_path.stat().st_size == artifact["bytes"]
        assert file_sha256(checkpoint_path) == artifact["sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if model_source == "ablation":
            assert checkpoint["configuration"]["ablation"]["name"] == name
            assert checkpoint["configuration"]["model"]["context_mode"] == entry[
                "mechanism"
            ]["context_mode"]
            assert (
                float(checkpoint["configuration"]["loss"]["weights"]["direction"])
                == entry["mechanism"]["direction_weight"]
            )
        elif model_source == "stage2_replication":
            assert checkpoint["configuration"]["seed"] == int(seed)
            assert checkpoint["configuration"]["replication"] == {
                "model_and_sampling_seed": int(seed),
                "target_split_seed": base_config["seed"],
                "base_config_path": "configs/dynamics.yaml",
                "base_config_sha256": model_manifest["source"]["base_config_sha256"],
            }
            assert checkpoint["state"]["best_validation_epoch"] == entry["full_run"][
                "best_validation_epoch"
            ]
            assert checkpoint["state"]["best_validation_loss"] == entry["full_run"][
                "best_validation_loss"
            ]
        else:
            assert checkpoint["configuration"]["revision"]["candidate"] == selected_candidate
            assert checkpoint["state"]["best_validation_epoch"] == entry["full_run"][
                "best_validation_epoch"
            ]
            assert checkpoint["state"]["best_validation_loss"] == entry["full_run"][
                "best_validation_loss"
            ]
        assert checkpoint["provenance"]["git"]["dirty"] is False
        model = build_dynamics_model(checkpoint["configuration"]).eval()
        model.load_state_dict(checkpoint["model"])
        if "residual_gate" in config:
            gate = config["residual_gate"]
            gate_path = Path(gate["path"])
            assert (gate_path.stat().st_size, file_sha256(gate_path)) == (
                gate["bytes"], gate["sha256"]
            )
            model.configure_residual_gate(torch.load(gate_path, map_location="cpu", weights_only=True))
        models[name] = model
        checkpoint_provenance[name] = checkpoint["provenance"]

    dynamics_manifest = json.loads(Path(base_config["inputs"]["dynamics_manifest_path"]).read_text())
    median_distance = dynamics_manifest["normalization"]["median_training_latent_distance"]
    regimes = regimes or base_config["regimes"]
    repeats = repeats or base_config["sampling"]["repeats"]
    output = Path(output_directory or config["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    records, signatures = [], defaultdict(lambda: [0, None])
    for regime, specification in regimes.items():
        dataset = LatentPopulationDataset(
            base_config["inputs"]["latent_cache_path"],
            base_config["inputs"]["action_cache_path"],
            base_config["inputs"]["dynamics_manifest_path"],
            regime,
            base_config["sampling"]["population_size"],
            base_config["seed"],
            specification["outcome_role"],
            specification["control_role"],
            specification["context"],
        )
        indices = range(min(len(dataset), max_conditions or len(dataset)))
        for repeat in range(repeats):
            dataset.set_epoch(repeat)
            loader = DataLoader(
                dataset,
                batch_size=base_config["sampling"]["batch_size"],
                sampler=list(indices),
                num_workers=base_config["sampling"]["num_workers"],
            )
            for batch in loader:
                control, observed = batch["control"], batch["perturbed"]
                predictions = {
                    name: model(control, batch["action"], batch["action_known"])
                    for name, model in models.items()
                }
                confidences = {
                    name: model.residual_gate_confidence(control)
                    for name, model in models.items()
                    if getattr(model, "residual_gate_threshold", None) is not None
                }
                assert all(predicted.dtype == observed.dtype for predicted in predictions.values())
                true_effect = observed.mean(1) - control.mean(1)
                for index, target in enumerate(batch["target"]):
                    key = (regime, "true", target)
                    signatures[key][0] += 1
                    value = true_effect[index].numpy()
                    signatures[key][1] = (
                        value if signatures[key][1] is None else signatures[key][1] + value
                    )
                for name, predicted in predictions.items():
                    predicted_effect = (predicted.mean(1) - control.mean(1)).detach()
                    metrics = population_metrics(
                        predicted,
                        observed,
                        control,
                        median_distance,
                        base_config["metrics"],
                    )
                    for index, target in enumerate(batch["target"]):
                        record = {
                            "regime": regime,
                            "context": specification["context"],
                            "outcome_role": specification["outcome_role"],
                            "target": target,
                            "repeat": repeat,
                            "model": name,
                        }
                        for metric, values in metrics.items():
                            value = float(values[index].detach())
                            record[metric] = value if np.isfinite(value) else None
                        if name in confidences:
                            record["residual_gate_confidence"] = float(confidences[name][index])
                        records.append(record)
                        key = (regime, name, target)
                        signatures[key][0] += 1
                        value = predicted_effect[index].numpy()
                        signatures[key][1] = (
                            value if signatures[key][1] is None else signatures[key][1] + value
                        )

    metrics_path = output / "condition_metrics.jsonl"
    with metrics_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    base_records = [
        json.loads(line)
        for line in Path(inputs["base_condition_metrics_path"]).read_text().splitlines()
    ]
    assert len(base_records) == inputs["base_condition_metrics_records"]
    base_summary = json.loads(Path(inputs["base_summary_path"]).read_text())
    summary = {
        "condition_metrics": sorted(
            base_summary["condition_metrics"]
            + _condition_metric_summaries(
                records,
                base_config["metrics"]["bootstrap_resamples"],
                base_config["seed"],
            ),
            key=lambda item: (item["regime"], item["model"], item["metric"]),
        ),
        "retrieval": sorted(
            base_summary["retrieval"] + _retrieval_summaries(signatures, regimes, models),
            key=lambda item: (item["regime"], item["model"]),
        ),
    }
    comparisons = paired_model_comparisons(
        base_records + records,
        config["comparisons"],
        base_config["metrics"]["bootstrap_resamples"],
        base_config["seed"],
    )
    provenance = {
        "config": deepcopy(config),
        "base_evaluation_config": deepcopy(base_config),
        "executed_regimes": regimes,
        "executed_repeats": repeats,
        "maximum_conditions_per_regime": max_conditions,
        "checkpoint_provenance": checkpoint_provenance,
        "file_sha256": {
            "ablation_condition_metrics": file_sha256(metrics_path),
            "base_condition_metrics": inputs["base_condition_metrics_sha256"],
            "base_summary": inputs["base_summary_sha256"],
            "base_provenance": inputs["base_provenance_sha256"],
        },
        "base_evaluation_manifest_sha256": base_manifest_sha256,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    provenance[f"{manifest_kind}_manifest_sha256"] = model_manifest_sha256
    if "residual_gate" in config:
        provenance["file_sha256"]["residual_gate"] = config["residual_gate"]["sha256"]
    if selection_manifest_sha256 is not None:
        provenance["anchored_selection_manifest_sha256"] = selection_manifest_sha256
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return summary, comparisons, provenance
