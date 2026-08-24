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
from torch.utils.data import DataLoader

from causalcelljepa.dynamics import LatentPopulationDataset, build_dynamics_model
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


@torch.no_grad()
def run_evaluation(config, regimes=None, repeats=None, max_conditions=None, output_directory=None):
    """Run deterministic population evaluation and summarize at the target-condition level."""
    inputs = config["inputs"]
    for kind in ("latent_cache", "action_cache", "checkpoint"):
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
        for baseline in ("causalcelljepa", "no_change", "mean_effect", "linear_esm"):
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
        "file_sha256": {kind: inputs[f"{kind}_sha256"] for kind in ("latent_cache", "action_cache", "checkpoint")},
        "selection_manifest_sha256": declared,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "baseline_report.json").write_text(json.dumps(baseline_report, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return summary, baseline_report, provenance
