# Leakage-safe normalized-expression caching and latent-to-transcriptome readout.
# The decoder is separate from—and never backpropagates into—the frozen world model.
import json
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
import yaml
from scipy import sparse
from scipy.stats import rankdata, t, wilcoxon
from torch.utils.data import DataLoader

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    anchored_selected_entry,
    build_dynamics_model,
    learned_target_id_config,
    state_ablation_config,
)
from causalcelljepa.evaluation import fit_linear_baseline
from causalcelljepa.representations import build_autoencoder
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


def normalized_hvg_expression(counts, hvg_columns, library_size=10_000):
    """Normalize by each full measured library before selecting the frozen HVGs."""
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(1)
    assert np.isfinite(counts).all() and (counts >= 0).all() and (totals > 0).all()
    return np.log1p(
        counts[:, hvg_columns] * (np.float32(library_size) / totals)[:, None]
    ).astype(np.float32, copy=False)


def write_expression_cache(config, output_path=None, maximum_cells=None, verify_raw=True):
    """Stream raw H5AD rows into a latent-aligned normalized 3,000-HVG cache."""
    inputs, cache_config = config["inputs"], config["expression_cache"]
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    assert replogle["manifest_sha256"] == inputs["replogle_manifest_sha256"]
    replogle_config = yaml.safe_load(Path(inputs["replogle_config_path"]).read_text())
    assert file_sha256(inputs["replogle_config_path"]) == replogle["runtime"]["config_sha256"]
    latent_path = Path(inputs["latent_cache_path"])
    assert (latent_path.stat().st_size, file_sha256(latent_path)) == (
        inputs["latent_cache_bytes"],
        inputs["latent_cache_sha256"],
    )
    output = Path(output_path or cache_config["output_path"])
    assert cache_config["dtype"] == "float32" and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    assert not temporary.exists()
    with h5py.File(latent_path, "r") as latent:
        cells = min(int(latent.attrs["cells"]), maximum_cells or int(latent.attrs["cells"]))
        assert maximum_cells is not None or cells == cache_config["expected_cells"]
        contexts = latent["context"].asstr()[:cells]
        source_rows = latent["source_row"][:cells]
        roles = latent["role"].asstr()[:cells]
        with h5py.File(temporary, "w") as destination:
            destination.attrs.update(
                {
                    "format_version": 1,
                    "cells": cells,
                    "hvg_count": cache_config["hvg_count"],
                    "dtype": cache_config["dtype"],
                    "library_size": replogle_config["data"]["library_size"],
                    "latent_cache_sha256": inputs["latent_cache_sha256"],
                    "replogle_manifest_sha256": inputs["replogle_manifest_sha256"],
                    "hvg_sha256": replogle["genes"]["hvg_sha256"],
                    "role_counts_json": json.dumps(dict(sorted(Counter(roles).items()))),
                    "provenance_json": json.dumps(
                        {
                            "config_sha256": file_sha256("configs/readout.yaml"),
                            "runtime_source_sha256": _runtime_source_hash(),
                            "runtime_environment": _runtime_environment(),
                            "git": _git_state(),
                        },
                        sort_keys=True,
                    ),
                }
            )
            expression = destination.create_dataset(
                "expression",
                (cells, cache_config["hvg_count"]),
                dtype="f4",
                chunks=(cache_config["block_size"], cache_config["hvg_count"]),
            )
            hvg_ids = replogle["genes"]["hvg_gene_ids"]
            for context in replogle_config["data"]["contexts"]:
                source = replogle_config["data"]["files"][context]
                path = Path(inputs["raw_directory"]) / source["filename"]
                assert path.stat().st_size == source["bytes"]
                if verify_raw:
                    assert file_sha256(path) == source["sha256"]
                positions = np.flatnonzero(contexts == context)
                if not len(positions):
                    continue
                assert np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
                data = ad.read_h5ad(path, backed="r")
                columns = np.asarray([data.var_names.get_loc(gene) for gene in hvg_ids])
                rows = source_rows[positions]
                assert np.all(rows[1:] > rows[:-1])
                for start in range(0, len(positions), cache_config["block_size"]):
                    selected = positions[start : start + cache_config["block_size"]]
                    counts = np.asarray(data.X[source_rows[selected]])
                    expression[selected] = normalized_hvg_expression(
                        counts, columns, replogle_config["data"]["library_size"]
                    )
                data.file.close()
            destination.flush()
    temporary.replace(output)
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "cells": cells,
        "hvg_count": cache_config["hvg_count"],
        "role_counts": dict(sorted(Counter(roles).items())),
    }


def decoder_split(cell_ids, roles, fit_roles, seed, validation_fraction, maximum_cells=None):
    """Create an exact deterministic cell split restricted to representation-visible roles."""
    eligible = np.flatnonzero(np.isin(roles, fit_roles))
    ranked = sorted(
        eligible,
        key=lambda index: sha256(f"{seed}\0readout\0{cell_ids[index]}".encode()).digest(),
    )
    if maximum_cells is not None:
        ranked = ranked[:maximum_cells]
    validation_cells = max(1, round(validation_fraction * len(ranked)))
    return np.sort(ranked[validation_cells:]), np.sort(ranked[:validation_cells])


def sufficient_statistics(latent, expression, indices, mean, scale, block_size):
    """Accumulate the exact multivariate linear-regression sufficient statistics."""
    width, genes = latent.shape[1] + 1, expression.shape[1]
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros((width, genes), dtype=np.float64)
    response_square = 0.0
    for start in range(0, len(indices), block_size):
        selected = indices[start : start + block_size]
        design = np.empty((len(selected), width), dtype=np.float32)
        design[:, :-1] = (latent[selected] - mean) / scale
        design[:, -1] = 1
        response = expression[selected].astype(np.float32)
        gram += design.T @ design
        cross += design.T @ response
        response_square += float(np.square(response.astype(np.float64)).sum())
    return {"cells": len(indices), "gram": gram, "cross": cross, "response_square": response_square}


def ridge_solution(statistics, alpha):
    """Solve the linear decoder while leaving the intercept unpenalized."""
    penalty = np.eye(statistics["gram"].shape[0], dtype=np.float64) * alpha
    penalty[-1, -1] = 0
    return np.linalg.solve(statistics["gram"] + penalty, statistics["cross"])


def regression_mse(statistics, solution):
    """Evaluate an un-clipped linear decoder exactly from sufficient statistics."""
    residual = (
        statistics["response_square"]
        - 2 * np.sum(solution * statistics["cross"])
        + np.sum(solution * (statistics["gram"] @ solution))
    )
    return max(0.0, float(residual)) / (statistics["cells"] * statistics["cross"].shape[1])


def fit_readout(config, maximum_cells=None):
    """Select and refit the decoder using only explicitly permitted cell roles."""
    inputs, decoder = config["inputs"], config["decoder"]
    assert decoder["architecture"] == "linear"
    dynamics = json.loads(Path(inputs["dynamics_manifest_path"]).read_text())
    assert dynamics["manifest_sha256"] == inputs["dynamics_manifest_sha256"]
    latent_mean = np.asarray(dynamics["normalization"]["latent_mean"], dtype=np.float32)
    latent_scale = (
        np.asarray(dynamics["normalization"]["latent_std"], dtype=np.float32)
        * dynamics["normalization"]["dimension_scale"]
    )
    with h5py.File(inputs["latent_cache_path"], "r") as latent, h5py.File(
        config["expression_cache"]["output_path"], "r"
    ) as expression_cache:
        cells = int(expression_cache.attrs["cells"])
        assert expression_cache.attrs["latent_cache_sha256"] == inputs["latent_cache_sha256"]
        cell_ids, roles = latent["cell_id"].asstr()[:cells], latent["role"].asstr()[:cells]
        train, validation = decoder_split(
            cell_ids,
            roles,
            decoder["fit_roles"],
            config["seed"],
            decoder["validation_fraction"],
            maximum_cells,
        )
        train_stats = sufficient_statistics(
            latent["latent"],
            expression_cache["expression"],
            train,
            latent_mean,
            latent_scale,
            decoder["block_size"],
        )
        validation_stats = sufficient_statistics(
            latent["latent"],
            expression_cache["expression"],
            validation,
            latent_mean,
            latent_scale,
            decoder["block_size"],
        )
        candidates = []
        for alpha in decoder["ridge_candidates"]:
            solution = ridge_solution(train_stats, alpha)
            candidates.append((regression_mse(validation_stats, solution), alpha))
        validation_mse, selected_alpha = min(candidates)
        combined = {
            key: train_stats[key] + validation_stats[key]
            for key in ("cells", "gram", "cross", "response_square")
        }
        solution = ridge_solution(combined, selected_alpha)
        baseline_mean = train_stats["cross"][-1] / train_stats["cells"]
        baseline = np.vstack(
            (np.zeros((solution.shape[0] - 1, solution.shape[1])), baseline_mean)
        )
        baseline_mse = regression_mse(validation_stats, baseline)
        used_roles = sorted(set(roles[np.concatenate((train, validation))]))
        assert used_roles == sorted(decoder["fit_roles"])
        report = {
            "architecture": decoder["architecture"],
            "fit_roles": used_roles,
            "fit_cells": combined["cells"],
            "train_cells": train_stats["cells"],
            "validation_cells": validation_stats["cells"],
            "validation_fraction": decoder["validation_fraction"],
            "ridge_candidates": decoder["ridge_candidates"],
            "ridge_validation_mse": [value for value, _ in candidates],
            "selected_ridge": selected_alpha,
            "selected_validation_mse": validation_mse,
            "gene_mean_validation_mse": baseline_mse,
            "validation_explained_fraction": 1 - validation_mse / baseline_mse,
            "split_cell_ids_sha256": {
                "train": sha256("\n".join(sorted(cell_ids[train])).encode()).hexdigest(),
                "validation": sha256("\n".join(sorted(cell_ids[validation])).encode()).hexdigest(),
            },
        }
    checkpoint = {
        "format_version": 1,
        "weights": torch.from_numpy(solution[:-1].astype(np.float32)),
        "bias": torch.from_numpy(solution[-1].astype(np.float32)),
        "latent_mean": torch.from_numpy(latent_mean),
        "latent_scale": torch.from_numpy(latent_scale),
        "output_clamp_min": decoder["output_clamp_min"],
        "report": report,
        "provenance": {
            "config_sha256": file_sha256("configs/readout.yaml"),
            "latent_cache_sha256": inputs["latent_cache_sha256"],
            "expression_cache_sha256": file_sha256(config["expression_cache"]["output_path"]),
            "replogle_manifest_sha256": inputs["replogle_manifest_sha256"],
            "dynamics_manifest_sha256": inputs["dynamics_manifest_sha256"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    assert torch.isfinite(checkpoint["weights"]).all() and torch.isfinite(checkpoint["bias"]).all()
    return checkpoint


def decode_normalized_latents(latents, checkpoint):
    """Decode dynamics-space latents to nonnegative normalized log expression."""
    return (latents @ checkpoint["weights"] + checkpoint["bias"]).clamp_min(
        checkpoint["output_clamp_min"]
    )


def decode_representation_centroids(latents, decoder):
    """Map normalized state centroids through one frozen representation-specific readout."""
    if decoder["kind"] == "jepa_linear":
        return decode_normalized_latents(latents, decoder["checkpoint"])
    raw = latents * decoder["latent_scale"] + decoder["latent_mean"]
    if decoder["kind"] == "pca_inverse":
        expression = raw @ decoder["components"] + decoder["expression_mean"]
    else:
        assert decoder["kind"] == "autoencoder"
        expression = decoder["model"].decoder(raw)
    return expression.clamp_min(0)


def _bh_adjust(p_values):
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def expression_truth(config, regimes=None, maximum_conditions=None):
    """Compute batch-matched true effects and batch-level DE calls once per condition."""
    inputs, metric_config = config["inputs"], config["metrics"]
    regimes = regimes or config["regimes"]
    truth, report = {}, {}
    with h5py.File(inputs["latent_cache_path"], "r") as latent, h5py.File(
        inputs["expression_cache_path"], "r"
    ) as expression_cache:
        roles = latent["role"].asstr()[:]
        targets = latent["target"].asstr()[:]
        batches = latent["source_batch"].asstr()[:]
        contexts = latent["context"].asstr()[:]
        expression = expression_cache["expression"]
        for regime, specification in regimes.items():
            control_indices = np.flatnonzero(roles == specification["control_role"])
            assert set(contexts[control_indices]) == {specification["context"]}
            control_means = {
                batch: expression[np.sort(control_indices[batches[control_indices] == batch])].mean(
                    0, dtype=np.float64
                )
                for batch in sorted(set(batches[control_indices]))
            }
            outcome_indices = np.flatnonzero(roles == specification["outcome_role"])
            assert set(contexts[outcome_indices]) == {specification["context"]}
            condition_targets = sorted(set(targets[outcome_indices]))
            condition_targets = condition_targets[: maximum_conditions or len(condition_targets)]
            report[regime] = {
                "context": specification["context"],
                "outcome_role": specification["outcome_role"],
                "targets": len(condition_targets),
                "conditions_with_no_degs": 0,
            }
            for target in condition_targets:
                indices = np.sort(outcome_indices[targets[outcome_indices] == target])
                observed = expression[indices].astype(np.float64)
                target_batches = batches[indices]
                batch_effects, matched_control = [], np.zeros(observed.shape[1])
                for batch in sorted(set(target_batches)):
                    selected = target_batches == batch
                    batch_effects.append(observed[selected].mean(0) - control_means[batch])
                    matched_control += selected.sum() * control_means[batch]
                batch_effects = np.stack(batch_effects)
                effect = observed.mean(0) - matched_control / len(indices)
                if len(batch_effects) > 1:
                    batch_mean = batch_effects.mean(0)
                    standard_error = batch_effects.std(0, ddof=1) / np.sqrt(len(batch_effects))
                    statistic = np.divide(
                        np.abs(batch_mean),
                        standard_error,
                        out=np.where(batch_mean == 0, 0.0, np.inf),
                        where=standard_error > 0,
                    )
                    q_value = _bh_adjust(2 * t.sf(statistic, len(batch_effects) - 1))
                else:
                    q_value = np.ones(observed.shape[1])
                deg = (q_value <= metric_config["deg_batch_fdr"]) & (
                    np.abs(effect) >= metric_config["deg_min_abs_effect"]
                )
                report[regime]["conditions_with_no_degs"] += int(not deg.any())
                truth[regime, target] = {
                    "effect": effect.astype(np.float32),
                    "deg": deg,
                    "q_value": q_value.astype(np.float32),
                    "batches": len(batch_effects),
                    "cells": len(indices),
                }
    return truth, report


def _correlation(predicted, observed, rank=False):
    if np.std(predicted) <= 1e-12 or np.std(observed) <= 1e-12:
        return None
    if rank:
        predicted, observed = rankdata(predicted), rankdata(observed)
    predicted, observed = predicted - predicted.mean(), observed - observed.mean()
    return float(predicted @ observed / (np.linalg.norm(predicted) * np.linalg.norm(observed)))


def _binary_ranking_metrics(scores, labels):
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None, None
    ranks = rankdata(scores)
    auroc = (ranks[labels].sum() - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    cumulative = np.cumsum(sorted_labels)
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    true_positives = cumulative[group_ends]
    precision = true_positives / (group_ends + 1)
    average_precision = np.sum(np.diff(np.r_[0, true_positives]) * precision) / positives
    return float(average_precision), float(auroc)


def gene_effect_metrics(predicted, observed, target_index, deg, top_genes=(20, 50, 100)):
    """Evaluate one perturbation-condition, explicitly labeling outcome-selected scopes."""
    result = {}
    scopes = {"all": np.arange(len(observed))}
    excluded = np.arange(len(observed)) if target_index is None else np.delete(
        np.arange(len(observed)), target_index
    )
    scopes["target_excluded"] = excluded
    for scope, indices in scopes.items():
        predicted_scope, observed_scope = predicted[indices], observed[indices]
        predicted_magnitude, observed_magnitude = (
            np.linalg.norm(predicted_scope),
            np.linalg.norm(observed_scope),
        )
        result[f"{scope}_effect_pearson"] = _correlation(predicted_scope, observed_scope)
        result[f"{scope}_effect_spearman"] = _correlation(
            predicted_scope, observed_scope, rank=True
        )
        result[f"{scope}_direction_cosine"] = (
            float(predicted_scope @ observed_scope / (predicted_magnitude * observed_magnitude))
            if predicted_magnitude > 1e-12 and observed_magnitude > 1e-12
            else None
        )
        result[f"{scope}_magnitude_ratio"] = float(
            predicted_magnitude / max(observed_magnitude, 1e-12)
        )
        result[f"{scope}_magnitude_absolute_error"] = float(
            abs(predicted_magnitude - observed_magnitude)
        )
    ranked_true = np.argsort(-np.abs(observed), kind="stable")
    ranked_predicted = np.argsort(-np.abs(predicted), kind="stable")
    for top in top_genes:
        indices = ranked_true[:top]
        result[f"retrospective_top{top}_effect_pearson"] = _correlation(
            predicted[indices], observed[indices]
        )
        result[f"retrospective_top{top}_effect_spearman"] = _correlation(
            predicted[indices], observed[indices], rank=True
        )
        result[f"retrospective_top{top}_overlap"] = float(
            len(set(indices) & set(ranked_predicted[:top])) / top
        )
    auprc, auroc = _binary_ranking_metrics(np.abs(predicted), deg)
    result["deg_auprc"], result["deg_auroc"] = auprc, auroc
    result["deg_sign_accuracy"] = (
        float(np.mean(np.sign(predicted[deg]) == np.sign(observed[deg]))) if deg.any() else None
    )
    return result


def _load_pathway_matrix(path, hvg_genes):
    gene_index = {gene: index for index, gene in enumerate(hvg_genes)}
    rows, columns, labels = [], [], []
    with Path(path).open() as handle:
        for row, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            indices = sorted({gene_index[gene] for gene in fields[2:] if gene in gene_index})
            labels.append(fields[0])
            rows.extend([row] * len(indices))
            columns.extend(indices)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(labels), len(hvg_genes)),
    )
    return matrix, labels


def pathway_scores(effect, matrix):
    """Return an analytic size-normalized signed rank-enrichment score."""
    ranks = rankdata(effect).astype(np.float32)
    if ranks.std() == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    ranks = (ranks - ranks.mean()) / ranks.std()
    sizes = np.asarray(matrix.sum(1)).ravel()
    denominator = np.sqrt(sizes * (len(effect) - sizes) / (len(effect) - 1))
    return np.asarray(matrix @ ranks).ravel() / denominator


def pathway_agreement(predicted, observed, matrix, top_k=(10, 25, 50)):
    predicted_score, observed_score = pathway_scores(predicted, matrix), pathway_scores(
        observed, matrix
    )
    result = {
        "pathway_nes_pearson": _correlation(predicted_score, observed_score),
        "pathway_rank_spearman": _correlation(predicted_score, observed_score, rank=True),
        "pathway_nes_rmse": float(np.sqrt(np.mean(np.square(predicted_score - observed_score)))),
    }
    predicted_order, observed_order = (
        np.argsort(-np.abs(predicted_score)),
        np.argsort(-np.abs(observed_score)),
    )
    for top in top_k:
        if np.std(predicted) <= 1e-12:
            result[f"pathway_top{top}_jaccard"] = None
        else:
            intersection = len(set(predicted_order[:top]) & set(observed_order[:top]))
            result[f"pathway_top{top}_jaccard"] = float(
                intersection / (2 * top - intersection)
            )
    return result


_TRANSCRIPTOMIC_METADATA = {
    "regime",
    "context",
    "outcome_role",
    "target",
    "repeat",
    "model",
    "truth_batches",
    "truth_cells",
    "true_deg_count",
    "target_in_hvg",
}


def _condition_summary(records, resamples, seed):
    target_values = defaultdict(list)
    for record in records:
        for metric in set(record) - _TRANSCRIPTOMIC_METADATA:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    grouped = defaultdict(list)
    for (regime, model, target, metric), values in target_values.items():
        grouped[regime, model, metric].append(float(np.mean(values)))
    summary = []
    for key, values in sorted(grouped.items()):
        values = np.asarray(values)
        generator = np.random.default_rng(
            int.from_bytes(sha256("\0".join(key).encode()).digest()[:8], "little") + seed
        )
        bootstrap = values[
            generator.integers(0, len(values), (resamples, len(values)))
        ].mean(1)
        summary.append(
            {
                "regime": key[0],
                "model": key[1],
                "metric": key[2],
                "targets": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "mean_bootstrap_95ci": [
                    float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                ],
            }
        )
    return summary


def _paired_comparisons(
    records,
    resamples,
    seed,
    baselines=("no_change", "mean_effect", "linear_esm", "pseudo_paired"),
):
    target_values = defaultdict(list)
    for record in records:
        for metric in set(record) - _TRANSCRIPTOMIC_METADATA:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    averaged = {key: float(np.mean(values)) for key, values in target_values.items()}
    comparisons = []
    for regime in sorted({record["regime"] for record in records}):
        metrics = sorted({key[3] for key in averaged if key[:2] == (regime, "causalcelljepa")})
        for baseline in baselines:
            for metric in metrics:
                targets = sorted(
                    key[2]
                    for key in averaged
                    if key[:2] == (regime, "causalcelljepa")
                    and key[3] == metric
                    and (regime, baseline, key[2], metric) in averaged
                )
                if not targets:
                    continue
                model = np.asarray([averaged[regime, "causalcelljepa", x, metric] for x in targets])
                reference = np.asarray([averaged[regime, baseline, x, metric] for x in targets])
                if metric.endswith("magnitude_ratio"):
                    improvement, direction = np.abs(reference - 1) - np.abs(model - 1), "closer_to_one_is_better"
                elif metric.endswith(("absolute_error", "rmse")):
                    improvement, direction = reference - model, "lower_is_better"
                else:
                    improvement, direction = model - reference, "higher_is_better"
                generator = np.random.default_rng(
                    (int.from_bytes(sha256(f"{regime}\0{baseline}\0{metric}".encode()).digest()[:8], "little") + seed)
                    % (1 << 64)
                )
                bootstrap = improvement[
                    generator.integers(0, len(improvement), (resamples, len(improvement)))
                ].mean(1)
                p_value = 1.0 if np.all(improvement == 0) else float(
                    wilcoxon(improvement, alternative="two-sided", method="approx").pvalue
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
                        "mean_improvement_bootstrap_95ci": [
                            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
                        ],
                        "wilcoxon_two_sided_p": p_value,
                    }
                )
    q_values = _bh_adjust(np.asarray([item["wilcoxon_two_sided_p"] for item in comparisons]))
    for comparison, q_value in zip(comparisons, q_values):
        comparison["benjamini_hochberg_q"] = float(q_value)
    return comparisons


def _paired_transcriptomic_models(records, pairs, resamples, seed):
    """Compare arbitrary models on matched perturbations using transcriptomic metric directions."""
    target_values = defaultdict(list)
    for record in records:
        for metric in set(record) - _TRANSCRIPTOMIC_METADATA:
            if record[metric] is not None:
                target_values[record["regime"], record["model"], record["target"], metric].append(
                    record[metric]
                )
    averaged = {key: float(np.mean(values)) for key, values in target_values.items()}
    comparisons = []
    for regime in sorted({record["regime"] for record in records}):
        for pair in pairs:
            candidate, reference = pair["candidate"], pair["reference"]
            metrics = sorted({
                key[3]
                for key in averaged
                if key[:2] == (regime, candidate)
                and (regime, reference, key[2], key[3]) in averaged
            })
            for metric in metrics:
                targets = sorted(
                    key[2]
                    for key in averaged
                    if key[:2] == (regime, candidate)
                    and key[3] == metric
                    and (regime, reference, key[2], metric) in averaged
                )
                candidate_values = np.asarray(
                    [averaged[regime, candidate, target, metric] for target in targets]
                )
                reference_values = np.asarray(
                    [averaged[regime, reference, target, metric] for target in targets]
                )
                if metric.endswith("magnitude_ratio"):
                    improvement = np.abs(reference_values - 1) - np.abs(candidate_values - 1)
                    direction = "closer_to_one_is_better"
                elif metric.endswith(("absolute_error", "rmse")):
                    improvement, direction = reference_values - candidate_values, "lower_is_better"
                else:
                    improvement, direction = candidate_values - reference_values, "higher_is_better"
                key = f"{regime}\0{candidate}\0{reference}\0{metric}"
                generator = np.random.default_rng(
                    (int.from_bytes(sha256(key.encode()).digest()[:8], "little") + seed)
                    % (1 << 64)
                )
                bootstrap = improvement[
                    generator.integers(0, len(improvement), (resamples, len(improvement)))
                ].mean(1)
                p_value = 1.0 if np.all(improvement == 0) else float(
                    wilcoxon(improvement, alternative="two-sided", method="approx").pvalue
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
    q_values = _bh_adjust(np.asarray([item["wilcoxon_two_sided_p"] for item in comparisons]))
    for comparison, q_value in zip(comparisons, q_values):
        comparison["benjamini_hochberg_q"] = float(q_value)
    return comparisons


@torch.no_grad()
def run_transcriptomic_evaluation(
    config, regimes=None, repeats=None, maximum_conditions=None, output_directory=None
):
    """Decode all frozen models and evaluate gene effects without fitting on outcomes."""
    inputs = config["inputs"]
    for kind in (
        "latent_cache",
        "expression_cache",
        "action_cache",
        "checkpoint",
        "pseudo_paired_checkpoint",
        "readout_checkpoint",
        "go_gmt",
    ):
        assert file_sha256(inputs[f"{kind}_path"]) == inputs[f"{kind}_sha256"]
    for kind in ("readout", "go"):
        manifest = json.loads(Path(inputs[f"{kind}_manifest_path"]).read_text())
        declared = manifest.pop("manifest_sha256")
        assert declared == inputs[f"{kind}_manifest_sha256"] == sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    primary = torch.load(inputs["checkpoint_path"], map_location="cpu", weights_only=False)
    pseudo = torch.load(
        inputs["pseudo_paired_checkpoint_path"], map_location="cpu", weights_only=False
    )
    readout = torch.load(inputs["readout_checkpoint_path"], map_location="cpu", weights_only=False)
    model = build_dynamics_model(primary["configuration"]).eval()
    model.load_state_dict(primary["model"])
    pseudo_model = build_dynamics_model(pseudo["configuration"]).eval()
    pseudo_model.load_state_dict(pseudo["model"])
    assert pseudo["configuration"]["objective"] == "pseudo_paired_mse"
    linear_effect, mean_effect, baseline_report = fit_linear_baseline(config)
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    hvg_genes = replogle["genes"]["hvg_gene_names"]
    hvg_index = {gene: index for index, gene in enumerate(hvg_genes)}
    pathway_matrix, pathway_labels = _load_pathway_matrix(inputs["go_gmt_path"], hvg_genes)
    assert len(pathway_labels) == 4328
    regimes = regimes or config["regimes"]
    repeats = repeats or config["sampling"]["repeats"]
    truth, truth_report = expression_truth(config, regimes, maximum_conditions)
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
        indices = range(min(len(dataset), maximum_conditions or len(dataset)))
        for repeat in range(repeats):
            dataset.set_epoch(repeat)
            loader = DataLoader(
                dataset,
                batch_size=config["sampling"]["batch_size"],
                sampler=list(indices),
                num_workers=config["sampling"]["num_workers"],
            )
            for batch in loader:
                control = batch["control"]
                predictions = {
                    "causalcelljepa": model(control, batch["action"], batch["action_known"]),
                    "pseudo_paired": pseudo_model(
                        control, batch["action"], batch["action_known"]
                    ),
                    "no_change": control,
                    "mean_effect": control + torch.from_numpy(mean_effect)[None, None],
                    "linear_esm": control
                    + torch.from_numpy(np.stack([linear_effect[x] for x in batch["target"]]))[
                        :, None
                    ],
                }
                control_expression = decode_normalized_latents(control.mean(1), readout)
                for baseline, predicted in predictions.items():
                    predicted_effect = (
                        decode_normalized_latents(predicted.mean(1), readout) - control_expression
                    ).numpy()
                    for index, target in enumerate(batch["target"]):
                        observed = truth[regime, target]
                        metrics = gene_effect_metrics(
                            predicted_effect[index],
                            observed["effect"],
                            hvg_index.get(target),
                            observed["deg"],
                            config["metrics"]["retrospective_top_genes"],
                        )
                        record = {
                            "regime": regime,
                            "context": specification["context"],
                            "outcome_role": specification["outcome_role"],
                            "target": target,
                            "repeat": repeat,
                            "model": baseline,
                            "truth_batches": observed["batches"],
                            "truth_cells": observed["cells"],
                            "true_deg_count": int(observed["deg"].sum()),
                            "target_in_hvg": target in hvg_index,
                            **metrics,
                        }
                        records.append(record)
                        key = (regime, baseline, target)
                        signatures[key][0] += 1
                        signatures[key][1] = (
                            predicted_effect[index]
                            if signatures[key][1] is None
                            else signatures[key][1] + predicted_effect[index]
                        )
    pathway_records, retrieval = [], []
    for regime in regimes:
        targets = sorted(target for key, target in truth if key == regime)
        true_signatures = np.stack([truth[regime, target]["effect"] for target in targets])
        normalized_truth = true_signatures / np.linalg.norm(
            true_signatures, axis=1, keepdims=True
        ).clip(1e-12)
        for baseline in sorted({record["model"] for record in records}):
            predicted_signatures = np.stack(
                [
                    signatures[regime, baseline, target][1]
                    / signatures[regime, baseline, target][0]
                    for target in targets
                ]
            )
            normalized_prediction = predicted_signatures / np.linalg.norm(
                predicted_signatures, axis=1, keepdims=True
            ).clip(1e-12)
            similarity = normalized_prediction @ normalized_truth.T
            ranks = np.asarray(
                [
                    1
                    + np.count_nonzero(row > row[index])
                    + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                    for index, row in enumerate(similarity)
                ]
            )
            retrieval.append(
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
            for index, target in enumerate(targets):
                pathway_records.append(
                    {
                        "regime": regime,
                        "context": regimes[regime]["context"],
                        "outcome_role": regimes[regime]["outcome_role"],
                        "target": target,
                        "repeat": 0,
                        "model": baseline,
                        "truth_batches": truth[regime, target]["batches"],
                        "truth_cells": truth[regime, target]["cells"],
                        "true_deg_count": int(truth[regime, target]["deg"].sum()),
                        "target_in_hvg": target in hvg_index,
                        **pathway_agreement(
                            predicted_signatures[index],
                            true_signatures[index],
                            pathway_matrix,
                            config["metrics"]["pathway_top_k"],
                        ),
                    }
                )
    resamples = config["metrics"]["bootstrap_resamples"]
    summary = {
        "condition_metrics": _condition_summary(records, resamples, config["seed"]),
        "pathway_metrics": _condition_summary(pathway_records, resamples, config["seed"]),
        "retrieval": retrieval,
    }
    paired = {
        "condition_comparisons": _paired_comparisons(records, resamples, config["seed"]),
        "pathway_comparisons": _paired_comparisons(
            pathway_records, resamples, config["seed"]
        ),
    }
    provenance = {
        "config": deepcopy(config),
        "executed_regimes": regimes,
        "executed_repeats": repeats,
        "maximum_conditions_per_regime": maximum_conditions,
        "checkpoint_provenance": primary["provenance"],
        "pseudo_paired_checkpoint_provenance": pseudo["provenance"],
        "readout_provenance": readout["provenance"],
        "file_sha256": {
            kind: inputs[f"{kind}_sha256"]
            for kind in (
                "latent_cache",
                "expression_cache",
                "action_cache",
                "checkpoint",
                "pseudo_paired_checkpoint",
                "readout_checkpoint",
                "go_gmt",
            )
        },
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    for filename, payload in (
        ("summary.json", summary),
        ("paired_comparisons.json", paired),
        ("truth_report.json", truth_report),
        ("baseline_report.json", baseline_report),
        ("provenance.json", provenance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    return summary, paired, truth_report, provenance


@torch.no_grad()
def run_remaining_comparator_evaluation(
    config, base_config, regimes=None, repeats=None, maximum_conditions=None, output_directory=None
):
    """Evaluate frozen action/state comparators through leakage-safe representation readouts."""
    inputs = config["inputs"]
    assert file_sha256(config["base_transcriptomics_config_path"]) == config[
        "base_transcriptomics_config_sha256"
    ]
    model_source = config.get("model_source", "comparator")
    manifest_kinds = {
        "comparator": "comparator",
        "stage2_replication": "replication_training",
        "anchored": "anchored_training",
        "multiteacher": "multiteacher_training",
    }
    assert model_source in manifest_kinds
    model_manifest_kind = manifest_kinds[model_source]
    manifests = {}
    for name in ("base", model_manifest_kind):
        payload = json.loads(Path(inputs[f"{name}_manifest_path"]).read_text())
        declared = payload.pop("manifest_sha256")
        assert declared == inputs[f"{name}_manifest_sha256"] == sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifests[name] = payload
    selected_candidate = selected_entry = selection_manifest_sha256 = None
    frozen_selection_sources = {"anchored", "multiteacher"}
    if model_source in frozen_selection_sources:
        selection = json.loads(
            Path(inputs[f"{model_source}_selection_manifest_path"]).read_text()
        )
        selection_manifest_sha256 = selection.pop("manifest_sha256")
        assert (
            selection_manifest_sha256
            == inputs[f"{model_source}_selection_manifest_sha256"]
            == sha256(
                json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        selected_candidate, selected_entry = anchored_selected_entry(
            manifests[model_manifest_kind], selection
        )
        assert config["models"] == {f"{model_source}_selected": selected_candidate}
    base_artifacts = manifests["base"]["artifacts"]["predictive_evaluation"]
    for name in ("condition_metrics", "pathway_metrics", "summary", "provenance"):
        path = Path(inputs[f"base_{name}_path"])
        assert (path.stat().st_size, file_sha256(path)) == (
            inputs[f"base_{name}_bytes"],
            inputs[f"base_{name}_sha256"],
        )
        assert base_artifacts[name]["sha256"] == inputs[f"base_{name}_sha256"]
    assert base_artifacts["condition_metrics"]["records"] == inputs[
        "base_condition_metrics_records"
    ]
    assert base_artifacts["pathway_metrics"]["records"] == inputs[
        "base_pathway_metrics_records"
    ]
    base_provenance = json.loads(Path(inputs["base_provenance_path"]).read_text())
    assert base_provenance["config"] == base_config and base_provenance["git"]["dirty"] is False
    for kind in ("latent_cache", "expression_cache", "readout_checkpoint", "go_gmt"):
        assert file_sha256(base_config["inputs"][f"{kind}_path"]) == base_config["inputs"][
            f"{kind}_sha256"
        ]

    readout = torch.load(
        base_config["inputs"]["readout_checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert readout["report"]["fit_roles"] == [
        "control_inference",
        "control_train",
        "dynamics_train",
    ]
    models, decoders, model_configs, checkpoint_provenance = {}, {}, {}, {}
    representation_provenance = {"jepa_linear": readout["provenance"]}
    model_manifest = manifests[model_manifest_kind]
    assert model_manifest["protocol"]["rpe1_perturbed_outcomes_used_for_fit_or_selection"] is False
    assert model_manifest["protocol"]["sealed_test_outcomes_used_for_fit_or_selection"] is False
    for name, path in config["models"].items():
        if model_source == "stage2_replication":
            entry = model_manifest["artifacts"]["seeds"][str(path)]["best_checkpoint"]
            decoder = {"kind": "jepa_linear", "checkpoint": readout}
        elif model_source in frozen_selection_sources:
            entry = selected_entry["best_checkpoint"]
            decoder = {"kind": "jepa_linear", "checkpoint": readout}
        elif name == "learned_target_id":
            model_config, _, representation_manifest = learned_target_id_config(path)
            policy = representation_manifest["policy"]
            assert policy["heldout_target_identity_used_for_fit"] is False
            assert policy["validation_target_identity_used_for_fit"] is False
            decoder = {"kind": "jepa_linear", "checkpoint": readout}
        else:
            model_config, specification, representation_manifest = state_ablation_config(path)
            policy = representation_manifest["policy"]
            assert policy["rpe1_perturbed_outcomes_used_for_fit"] is False
            assert policy["sealed_test_outcomes_used_for_fit"] is False
            dynamics_manifest = json.loads(
                Path(model_config["inputs"]["dynamics_manifest_path"]).read_text()
            )
            normalization = dynamics_manifest["normalization"]
            mean = torch.tensor(normalization["latent_mean"], dtype=torch.float32)
            scale = torch.tensor(normalization["latent_std"], dtype=torch.float32) * normalization[
                "dimension_scale"
            ]
            if name == "pca_state":
                with h5py.File(model_config["inputs"]["latent_cache_path"], "r") as cache:
                    components = np.ascontiguousarray(cache["component"][:])
                    expression_mean = np.ascontiguousarray(cache["expression_mean"][:])
                artifact = representation_manifest["artifact"]
                assert sha256(components.tobytes()).hexdigest() == artifact["component_sha256"]
                assert sha256(expression_mean.tobytes()).hexdigest() == artifact[
                    "expression_mean_sha256"
                ]
                decoder = {
                    "kind": "pca_inverse",
                    "latent_mean": mean,
                    "latent_scale": scale,
                    "components": torch.from_numpy(components),
                    "expression_mean": torch.from_numpy(expression_mean),
                }
            else:
                assert name == "autoencoder_state"
                artifact = representation_manifest["training"]["best_checkpoint"]
                assert (Path(artifact["path"]).stat().st_size, file_sha256(artifact["path"])) == (
                    artifact["bytes"],
                    artifact["sha256"],
                )
                representation = torch.load(
                    artifact["path"], map_location="cpu", weights_only=False
                )
                assert representation["configuration"] == specification
                autoencoder = build_autoencoder(specification).eval()
                autoencoder.load_state_dict(representation["model"])
                decoder = {
                    "kind": "autoencoder",
                    "latent_mean": mean,
                    "latent_scale": scale,
                    "model": autoencoder,
                }
                representation_provenance[name] = representation["provenance"]
        if model_source == "comparator":
            entry = model_manifest["experiments"][name]["best_checkpoint"]
        assert (Path(entry["path"]).stat().st_size, file_sha256(entry["path"])) == (
            entry["bytes"],
            entry["sha256"],
        )
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        if model_source == "stage2_replication":
            model_config = checkpoint["configuration"]
            assert model_config["seed"] == path
            assert model_config["replication"]["model_and_sampling_seed"] == path
            assert model_config["replication"]["target_split_seed"] == base_config["seed"]
            assert checkpoint["state"]["best_validation_epoch"] == model_manifest["artifacts"][
                "seeds"
            ][str(path)]["full_run"]["best_validation_epoch"]
        elif model_source in frozen_selection_sources:
            model_config = checkpoint["configuration"]
            assert model_config["revision"]["candidate"] == path == selected_candidate
            assert checkpoint["state"]["best_validation_epoch"] == selected_entry["full_run"][
                "best_validation_epoch"
            ]
        else:
            assert checkpoint["configuration"] == model_config
        assert checkpoint["provenance"]["git"]["dirty"] is False
        model = build_dynamics_model(model_config).eval()
        model.load_state_dict(checkpoint["model"])
        models[name], decoders[name], model_configs[name] = model, decoder, model_config
        checkpoint_provenance[name] = checkpoint["provenance"]
        if model_source == "comparator":
            representation_provenance[f"{name}_manifest_sha256"] = representation_manifest[
                "manifest_sha256"
            ]

    replogle = json.loads(Path(base_config["inputs"]["replogle_manifest_path"]).read_text())
    hvg_genes = replogle["genes"]["hvg_gene_names"]
    hvg_index = {gene: index for index, gene in enumerate(hvg_genes)}
    pathway_matrix, pathway_labels = _load_pathway_matrix(
        base_config["inputs"]["go_gmt_path"], hvg_genes
    )
    assert len(pathway_labels) == 4328
    regimes = regimes or base_config["regimes"]
    repeats = repeats or base_config["sampling"]["repeats"]
    truth, truth_report = expression_truth(base_config, regimes, maximum_conditions)
    output = Path(output_directory or config["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    records, signatures = [], defaultdict(lambda: [0, None])
    for regime, regime_config in regimes.items():
        for name, model in models.items():
            model_config = model_configs[name]
            dataset = LatentPopulationDataset(
                model_config["inputs"]["latent_cache_path"],
                model_config["inputs"]["action_cache_path"],
                model_config["inputs"]["dynamics_manifest_path"],
                regime,
                base_config["sampling"]["population_size"],
                base_config["seed"],
                regime_config["outcome_role"],
                regime_config["control_role"],
                regime_config["context"],
            )
            indices = range(min(len(dataset), maximum_conditions or len(dataset)))
            for repeat in range(repeats):
                dataset.set_epoch(repeat)
                loader = DataLoader(
                    dataset,
                    batch_size=base_config["sampling"]["batch_size"],
                    sampler=list(indices),
                    num_workers=base_config["sampling"]["num_workers"],
                )
                for batch in loader:
                    control = batch["control"]
                    predicted = model(control, batch["action"], batch["action_known"])
                    predicted_effect = (
                        decode_representation_centroids(predicted.mean(1), decoders[name])
                        - decode_representation_centroids(control.mean(1), decoders[name])
                    ).numpy()
                    for index, target in enumerate(batch["target"]):
                        observed = truth[regime, target]
                        records.append(
                            {
                                "regime": regime,
                                "context": regime_config["context"],
                                "outcome_role": regime_config["outcome_role"],
                                "target": target,
                                "repeat": repeat,
                                "model": name,
                                "truth_batches": observed["batches"],
                                "truth_cells": observed["cells"],
                                "true_deg_count": int(observed["deg"].sum()),
                                "target_in_hvg": target in hvg_index,
                                **gene_effect_metrics(
                                    predicted_effect[index],
                                    observed["effect"],
                                    hvg_index.get(target),
                                    observed["deg"],
                                    base_config["metrics"]["retrospective_top_genes"],
                                ),
                            }
                        )
                        key = (regime, name, target)
                        signatures[key][0] += 1
                        signatures[key][1] = (
                            predicted_effect[index]
                            if signatures[key][1] is None
                            else signatures[key][1] + predicted_effect[index]
                        )

    pathway_records, retrieval = [], []
    for regime, regime_config in regimes.items():
        targets = sorted(target for key, target in truth if key == regime)
        true_signatures = np.stack([truth[regime, target]["effect"] for target in targets])
        normalized_truth = true_signatures / np.linalg.norm(
            true_signatures, axis=1, keepdims=True
        ).clip(1e-12)
        for name in models:
            predicted_signatures = np.stack(
                [signatures[regime, name, target][1] / signatures[regime, name, target][0] for target in targets]
            )
            normalized_prediction = predicted_signatures / np.linalg.norm(
                predicted_signatures, axis=1, keepdims=True
            ).clip(1e-12)
            similarity = normalized_prediction @ normalized_truth.T
            ranks = np.asarray(
                [
                    1
                    + np.count_nonzero(row > row[index])
                    + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                    for index, row in enumerate(similarity)
                ]
            )
            retrieval.append(
                {
                    "regime": regime,
                    "model": name,
                    "targets": len(targets),
                    "top_1": float(np.mean(ranks <= 1)),
                    "top_5": float(np.mean(ranks <= 5)),
                    "mean_reciprocal_rank": float(np.mean(1 / ranks)),
                    "median_rank": float(np.median(ranks)),
                }
            )
            for index, target in enumerate(targets):
                observed = truth[regime, target]
                pathway_records.append(
                    {
                        "regime": regime,
                        "context": regime_config["context"],
                        "outcome_role": regime_config["outcome_role"],
                        "target": target,
                        "repeat": 0,
                        "model": name,
                        "truth_batches": observed["batches"],
                        "truth_cells": observed["cells"],
                        "true_deg_count": int(observed["deg"].sum()),
                        "target_in_hvg": target in hvg_index,
                        **pathway_agreement(
                            predicted_signatures[index],
                            true_signatures[index],
                            pathway_matrix,
                            base_config["metrics"]["pathway_top_k"],
                        ),
                    }
                )

    base_summary = json.loads(Path(inputs["base_summary_path"]).read_text())
    resamples = base_config["metrics"]["bootstrap_resamples"]
    summary = {
        "condition_metrics": sorted(
            base_summary["condition_metrics"]
            + _condition_summary(records, resamples, base_config["seed"]),
            key=lambda item: (item["regime"], item["model"], item["metric"]),
        ),
        "pathway_metrics": sorted(
            base_summary["pathway_metrics"]
            + _condition_summary(pathway_records, resamples, base_config["seed"]),
            key=lambda item: (item["regime"], item["model"], item["metric"]),
        ),
        "retrieval": sorted(
            base_summary["retrieval"] + retrieval,
            key=lambda item: (item["regime"], item["model"]),
        ),
    }
    base_models = {
        model
        for pair in config["comparisons"]
        for model in (pair["candidate"], pair["reference"])
        if model not in models
    }
    base_condition_lines = Path(inputs["base_condition_metrics_path"]).read_text().splitlines()
    assert len(base_condition_lines) == inputs["base_condition_metrics_records"]
    base_records = []
    for line in base_condition_lines:
        record = json.loads(line)
        if record["model"] in base_models:
            base_records.append(record)
    base_pathway_lines = Path(inputs["base_pathway_metrics_path"]).read_text().splitlines()
    assert len(base_pathway_lines) == inputs["base_pathway_metrics_records"]
    base_pathway_records = []
    for line in base_pathway_lines:
        record = json.loads(line)
        if record["model"] in base_models:
            base_pathway_records.append(record)
    paired = {
        "condition_comparisons": _paired_transcriptomic_models(
            base_records + records, config["comparisons"], resamples, base_config["seed"]
        ),
        "pathway_comparisons": _paired_transcriptomic_models(
            base_pathway_records + pathway_records,
            config["comparisons"],
            resamples,
            base_config["seed"],
        ),
    }
    provenance = {
        "config": deepcopy(config),
        "base_transcriptomics_config": deepcopy(base_config),
        "executed_regimes": regimes,
        "executed_repeats": repeats,
        "maximum_conditions_per_regime": maximum_conditions,
        "checkpoint_provenance": checkpoint_provenance,
        "representation_provenance": representation_provenance,
        "base_artifact_sha256": {
            name: inputs[f"base_{name}_sha256"]
            for name in ("condition_metrics", "pathway_metrics", "summary", "provenance")
        },
        "base_manifest_sha256": inputs["base_manifest_sha256"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    provenance[f"{model_manifest_kind}_manifest_sha256"] = inputs[
        f"{model_manifest_kind}_manifest_sha256"
    ]
    if selection_manifest_sha256 is not None:
        provenance[f"{model_source}_selection_manifest_sha256"] = selection_manifest_sha256
    expected_records = repeats * len(models) * sum(item["targets"] for item in truth_report.values())
    assert len(records) == expected_records
    assert all(
        value is None or not isinstance(value, float) or np.isfinite(value)
        for record in records + pathway_records
        for value in record.values()
    )
    for filename, payload in (
        ("summary.json", summary),
        ("paired_comparisons.json", paired),
        ("truth_report.json", truth_report),
        ("provenance.json", provenance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    return summary, paired, truth_report, provenance


@torch.no_grad()
def run_readout_oracle(config, regimes=None, maximum_conditions=None, output_directory=None):
    """Audit the decoder ceiling using observed outcome latents, never as a predictor."""
    inputs = config["inputs"]
    for kind in ("latent_cache", "expression_cache", "readout_checkpoint", "go_gmt"):
        assert file_sha256(inputs[f"{kind}_path"]) == inputs[f"{kind}_sha256"]
    readout = torch.load(inputs["readout_checkpoint_path"], map_location="cpu", weights_only=False)
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    hvg_genes = replogle["genes"]["hvg_gene_names"]
    hvg_index = {gene: index for index, gene in enumerate(hvg_genes)}
    pathway_matrix, pathway_labels = _load_pathway_matrix(inputs["go_gmt_path"], hvg_genes)
    assert len(pathway_labels) == 4328
    regimes = regimes or config["regimes"]
    truth, truth_report = expression_truth(config, regimes, maximum_conditions)
    records, pathway_records = [], []
    with h5py.File(inputs["latent_cache_path"], "r") as latent:
        roles = latent["role"].asstr()[:]
        targets = latent["target"].asstr()[:]
        batches = latent["source_batch"].asstr()[:]
        contexts = latent["context"].asstr()[:]
        mean, scale = readout["latent_mean"].numpy(), readout["latent_scale"].numpy()
        for regime, specification in regimes.items():
            control_indices = np.flatnonzero(roles == specification["control_role"])
            assert set(contexts[control_indices]) == {specification["context"]}
            control_means = {
                batch: (latent["latent"][np.sort(control_indices[batches[control_indices] == batch])].mean(0) - mean)
                / scale
                for batch in sorted(set(batches[control_indices]))
            }
            outcome_indices = np.flatnonzero(roles == specification["outcome_role"])
            condition_targets = sorted(set(targets[outcome_indices]))
            condition_targets = condition_targets[: maximum_conditions or len(condition_targets)]
            for target in condition_targets:
                indices = np.sort(outcome_indices[targets[outcome_indices] == target])
                target_batches = batches[indices]
                observed_latent = (latent["latent"][indices].mean(0) - mean) / scale
                matched_control = sum(
                    np.count_nonzero(target_batches == batch) * control_means[batch]
                    for batch in set(target_batches)
                ).astype(np.float32) / len(indices)
                decoded_effect = (
                    decode_normalized_latents(torch.from_numpy(observed_latent[None]), readout)
                    - decode_normalized_latents(torch.from_numpy(matched_control[None]), readout)
                )[0].numpy()
                observed = truth[regime, target]
                metadata = {
                    "regime": regime,
                    "context": specification["context"],
                    "outcome_role": specification["outcome_role"],
                    "target": target,
                    "repeat": 0,
                    "model": "observed_latent_readout",
                    "truth_batches": observed["batches"],
                    "truth_cells": observed["cells"],
                    "true_deg_count": int(observed["deg"].sum()),
                    "target_in_hvg": target in hvg_index,
                }
                records.append(
                    {
                        **metadata,
                        **gene_effect_metrics(
                            decoded_effect,
                            observed["effect"],
                            hvg_index.get(target),
                            observed["deg"],
                            config["metrics"]["retrospective_top_genes"],
                        ),
                    }
                )
                pathway_records.append(
                    {
                        **metadata,
                        **pathway_agreement(
                            decoded_effect,
                            observed["effect"],
                            pathway_matrix,
                            config["metrics"]["pathway_top_k"],
                        ),
                    }
                )
    summary = {
        "condition_metrics": _condition_summary(
            records, config["metrics"]["bootstrap_resamples"], config["seed"]
        ),
        "pathway_metrics": _condition_summary(
            pathway_records, config["metrics"]["bootstrap_resamples"], config["seed"]
        ),
    }
    provenance = {
        "diagnostic_only": True,
        "observed_outcomes_used_as_model_inputs": True,
        "valid_predictive_baseline": False,
        "executed_regimes": regimes,
        "maximum_conditions_per_regime": maximum_conditions,
        "readout_provenance": readout["provenance"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    output = Path(output_directory or "artifacts/readout_oracle")
    assert not output.exists()
    output.mkdir(parents=True)
    for filename, payload in (
        ("summary.json", summary),
        ("truth_report.json", truth_report),
        ("provenance.json", provenance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    return summary, truth_report, provenance


def _expression_role_effects(config, roles, maximum_targets=None):
    inputs = config["inputs"]
    with h5py.File(inputs["latent_cache_path"], "r") as latent, h5py.File(
        inputs["expression_cache_path"], "r"
    ) as expression_cache:
        cache_roles = latent["role"].asstr()[:]
        targets = latent["target"].asstr()[:]
        batches = latent["source_batch"].asstr()[:]
        contexts = latent["context"].asstr()[:]
        control_indices = np.flatnonzero(cache_roles == "control_train")
        assert set(contexts[control_indices]) == {"K562"}
        expression = expression_cache["expression"]
        control_means = {
            batch: expression[np.sort(control_indices[batches[control_indices] == batch])].mean(
                0, dtype=np.float64
            )
            for batch in sorted(set(batches[control_indices]))
        }
        effects = {}
        for role in roles:
            outcome_indices = np.flatnonzero(cache_roles == role)
            assert set(contexts[outcome_indices]) == {"K562"}
            role_targets = sorted(set(targets[outcome_indices]))
            role_targets = role_targets[: maximum_targets or len(role_targets)]
            effects[role] = {}
            for target in role_targets:
                indices = np.sort(outcome_indices[targets[outcome_indices] == target])
                target_batches = batches[indices]
                matched_control = sum(
                    np.count_nonzero(target_batches == batch) * control_means[batch]
                    for batch in set(target_batches)
                ) / len(indices)
                effects[role][target] = (
                    expression[indices].mean(0, dtype=np.float64) - matched_control
                ).astype(np.float32)
    return effects


def fit_direct_gene_baseline(config, maximum_targets=None):
    """Fit ESM-to-expression effects with K562 train/validation outcomes only."""
    direct, base = config["direct_gene"], config["transcriptomics"]
    effects = _expression_role_effects(
        base,
        (direct["fit_outcome_role"], direct["selection_outcome_role"]),
        maximum_targets,
    )
    action = torch.load(base["inputs"]["action_cache_path"], map_location="cpu", weights_only=True)
    action_map = {
        target: (action["embedding"][index].numpy(), bool(action["known"][index]))
        for index, target in enumerate(action["targets"])
    }
    train = effects[direct["fit_outcome_role"]]
    known_targets = [target for target in sorted(train) if action_map[target][1]]
    x = np.stack([action_map[target][0] for target in known_targets]).astype(np.float64)
    y = np.stack([train[target] for target in known_targets]).astype(np.float64)
    x_mean, x_std = x.mean(0), x.std(0).clip(1e-8)
    y_mean = y.mean(0)
    _, _, components = np.linalg.svd(y - y_mean, full_matrices=False)
    components = components[: min(direct["rank"], len(known_targets))]
    scores = (y - y_mean) @ components.T
    standardized = (x - x_mean) / x_std
    gram, cross = standardized.T @ standardized, standardized.T @ scores
    validation = effects[direct["selection_outcome_role"]]
    validation_targets = sorted(validation)
    validation_x = np.stack([action_map[target][0] for target in validation_targets])
    validation_y = np.stack([validation[target] for target in validation_targets])
    candidates = []
    for alpha in direct["ridge_candidates"]:
        weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        prediction = ((validation_x - x_mean) / x_std) @ weights @ components + y_mean
        candidates.append((float(np.mean((prediction - validation_y) ** 2)), alpha, weights))
    validation_mse, alpha, weights = min(candidates, key=lambda item: (item[0], item[1]))
    checkpoint = {
        "format_version": 1,
        "architecture": "direct_gene_esm_low_rank_ridge",
        "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "x_std": torch.from_numpy(x_std.astype(np.float32)),
        "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "components": torch.from_numpy(components.astype(np.float32)),
        "weights": torch.from_numpy(weights.astype(np.float32)),
        "report": {
            "fit_outcome_role": direct["fit_outcome_role"],
            "selection_outcome_role": direct["selection_outcome_role"],
            "fit_targets": len(train),
            "fit_targets_with_known_action": len(known_targets),
            "selection_targets": len(validation_targets),
            "rank": len(components),
            "selected_ridge": alpha,
            "selection_mse": validation_mse,
            "ridge_validation_mse": [value for value, _, _ in candidates],
        },
        "provenance": {
            "config_sha256": file_sha256("configs/direct_gene.yaml"),
            "latent_cache_sha256": base["inputs"]["latent_cache_sha256"],
            "expression_cache_sha256": base["inputs"]["expression_cache_sha256"],
            "action_cache_sha256": base["inputs"]["action_cache_sha256"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    return checkpoint


def direct_gene_predictions(checkpoint, action_path):
    action = torch.load(action_path, map_location="cpu", weights_only=True)
    standardized = (action["embedding"] - checkpoint["x_mean"]) / checkpoint["x_std"]
    predicted = (
        standardized @ checkpoint["weights"] @ checkpoint["components"]
        + checkpoint["y_mean"]
    )
    predicted[~action["known"]] = checkpoint["y_mean"]
    return {target: predicted[index].numpy() for index, target in enumerate(action["targets"])}


def run_direct_gene_evaluation(config, checkpoint):
    """Evaluate the frozen direct gene-space baseline against the frozen five-model result."""
    base, direct = config["transcriptomics"], config["direct_gene"]
    manifest = json.loads(Path(config["transcriptomics_manifest_path"]).read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == config["transcriptomics_manifest_sha256"] == sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    prior_directory = Path(base["output_directory"])
    prior_hash = manifest["artifacts"]["predictive_evaluation"]["condition_metrics"][
        "sha256"
    ]
    assert file_sha256(prior_directory / "condition_metrics.jsonl") == prior_hash
    predictions = direct_gene_predictions(checkpoint, base["inputs"]["action_cache_path"])
    truth, truth_report = expression_truth(base)
    replogle = json.loads(Path(base["inputs"]["replogle_manifest_path"]).read_text())
    hvg_genes = replogle["genes"]["hvg_gene_names"]
    hvg_index = {gene: index for index, gene in enumerate(hvg_genes)}
    pathway_matrix, pathway_labels = _load_pathway_matrix(
        base["inputs"]["go_gmt_path"], hvg_genes
    )
    assert len(pathway_labels) == 4328
    records, pathway_records, retrieval = [], [], []
    for regime, specification in base["regimes"].items():
        targets = sorted(target for key, target in truth if key == regime)
        predicted_signatures = np.stack([predictions[target] for target in targets])
        true_signatures = np.stack([truth[regime, target]["effect"] for target in targets])
        normalized_prediction = predicted_signatures / np.linalg.norm(
            predicted_signatures, axis=1, keepdims=True
        ).clip(1e-12)
        normalized_truth = true_signatures / np.linalg.norm(
            true_signatures, axis=1, keepdims=True
        ).clip(1e-12)
        similarity = normalized_prediction @ normalized_truth.T
        ranks = np.asarray(
            [
                1
                + np.count_nonzero(row > row[index])
                + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                for index, row in enumerate(similarity)
            ]
        )
        retrieval.append(
            {
                "regime": regime,
                "model": "direct_gene_esm",
                "targets": len(targets),
                "top_1": float(np.mean(ranks <= 1)),
                "top_5": float(np.mean(ranks <= 5)),
                "mean_reciprocal_rank": float(np.mean(1 / ranks)),
                "median_rank": float(np.median(ranks)),
            }
        )
        for index, target in enumerate(targets):
            observed = truth[regime, target]
            metadata = {
                "regime": regime,
                "context": specification["context"],
                "outcome_role": specification["outcome_role"],
                "target": target,
                "repeat": 0,
                "model": "direct_gene_esm",
                "truth_batches": observed["batches"],
                "truth_cells": observed["cells"],
                "true_deg_count": int(observed["deg"].sum()),
                "target_in_hvg": target in hvg_index,
            }
            records.append(
                {
                    **metadata,
                    **gene_effect_metrics(
                        predicted_signatures[index],
                        observed["effect"],
                        hvg_index.get(target),
                        observed["deg"],
                        base["metrics"]["retrospective_top_genes"],
                    ),
                }
            )
            pathway_records.append(
                {
                    **metadata,
                    **pathway_agreement(
                        predicted_signatures[index],
                        observed["effect"],
                        pathway_matrix,
                        base["metrics"]["pathway_top_k"],
                    ),
                }
            )
    prior_records = [
        json.loads(line)
        for line in (prior_directory / "condition_metrics.jsonl").read_text().splitlines()
        if '"model": "causalcelljepa"' in line
    ]
    prior_pathways = [
        json.loads(line)
        for line in (prior_directory / "pathway_metrics.jsonl").read_text().splitlines()
        if '"model": "causalcelljepa"' in line
    ]
    resamples = base["metrics"]["bootstrap_resamples"]
    summary = {
        "condition_metrics": _condition_summary(records, resamples, base["seed"]),
        "pathway_metrics": _condition_summary(pathway_records, resamples, base["seed"]),
        "retrieval": retrieval,
    }
    paired = {
        "condition_comparisons": _paired_comparisons(
            prior_records + records,
            resamples,
            base["seed"],
            baselines=("direct_gene_esm",),
        ),
        "pathway_comparisons": _paired_comparisons(
            prior_pathways + pathway_records,
            resamples,
            base["seed"],
            baselines=("direct_gene_esm",),
        ),
    }
    provenance = {
        "config": deepcopy(config),
        "checkpoint_provenance": checkpoint["provenance"],
        "prior_transcriptomics_manifest_sha256": declared,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    output = Path(direct["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    for filename, payload in (
        ("summary.json", summary),
        ("paired_comparisons.json", paired),
        ("truth_report.json", truth_report),
        ("fit_report.json", checkpoint["report"]),
        ("provenance.json", provenance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    return summary, paired, truth_report, provenance
