# Generate data-backed UMAPs, paired scatterplots, and effect maps from frozen artifacts.
# This script performs no fitting, model selection, threshold selection, or outcome-driven sampling.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from causalcelljepa.dynamics import LatentPopulationDataset, build_dynamics_model
from causalcelljepa.external_evaluation import load_adamson_expression

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/empirical_figures.yaml"

INK = "#17212B"
NAVY = "#244B6B"
TEAL = "#1F9D8A"
ORANGE = "#E76F51"
GOLD = "#E9B949"
PURPLE = "#7259A5"
SKY = "#4C91C6"
GRAY = "#7A8793"
LIGHT = "#F5F7F6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def condition_values(rows: list[dict], model: str, regime: str, metric: str) -> dict[str, float]:
    """Average population repeats before treating a perturbation target as one observation."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if (
            row["model"] == model
            and row["regime"] == regime
            and value is not None
            and np.isfinite(value)
        ):
            grouped[row["target"]].append(float(value))
    return {target: float(np.mean(values)) for target, values in grouped.items()}


def blind_target_subset(targets: list[str], count: int, seed: int, regime: str) -> list[str]:
    """Select a reproducible target subset without reading outcomes or metric values."""

    def priority(target: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{regime}\0{target}".encode()).digest()

    selected = sorted(targets, key=lambda target: (priority(target), target))[:count]
    return sorted(selected)


def bootstrap_advantage(
    candidate: np.ndarray,
    reference: np.ndarray,
    higher_is_better: bool,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Bootstrap a target-level benefit, oriented so positive always favors the candidate."""
    differences = candidate - reference if higher_is_better else reference - candidate
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = generator.integers(0, len(differences), size=(stop - start, len(differences)))
        draws[start:stop] = differences[indices].mean(1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


def style() -> None:
    mpl.rcParams.update(
        {
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "causalcelljepa-empirical",
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def save_figure(fig: plt.Figure, stem: str, config: dict, hashes: dict[str, str]) -> None:
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    for extension in config["formats"]:
        path = output / f"{stem}.{extension}"
        metadata = {
            "pdf": {"Creator": "CausalCellJEPA", "CreationDate": None, "ModDate": None},
            "png": {"Software": "CausalCellJEPA"},
            "svg": {"Date": None},
        }[extension]
        fig.savefig(
            path, dpi=config["dpi"], bbox_inches="tight", pad_inches=0.08, metadata=metadata
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    plt.close(fig)


def verify_sources(config: dict) -> tuple[dict, dict, dict[str, str]]:
    """Verify every compact frozen artifact before plotting or checkpoint inference."""
    replogle = config["replogle"]
    evaluation = yaml.safe_load((ROOT / replogle["evaluation_config"]).read_text())
    evaluation_manifest = json.loads((ROOT / replogle["evaluation_manifest"]).read_text())
    adamson = config["adamson"]
    adamson_manifest = json.loads((ROOT / adamson["evaluation_manifest"]).read_text())
    prediction_manifest = json.loads((ROOT / adamson["prediction_manifest"]).read_text())
    hashes: dict[str, str] = {}

    replogle_paths = {
        "latent_cache_path": "latent_cache_sha256",
        "action_cache_path": "action_cache_sha256",
        "checkpoint_path": "checkpoint_sha256",
        "pseudo_paired_checkpoint_path": "pseudo_paired_checkpoint_sha256",
    }
    for path_key, sha_key in replogle_paths.items():
        path = ROOT / evaluation["inputs"][path_key]
        observed = file_sha256(path)
        assert observed == evaluation["inputs"][sha_key], f"hash mismatch for {path}"
        hashes[str(path.relative_to(ROOT))] = observed

    condition_path = ROOT / replogle["condition_metrics"]
    observed = file_sha256(condition_path)
    assert observed == evaluation_manifest["artifacts"]["condition_metrics"]["sha256"]
    hashes[str(condition_path.relative_to(ROOT))] = observed

    adamson_outputs = {
        "condition_metrics": "condition_metrics.jsonl",
        "paired_comparisons": "paired_comparisons.json",
    }
    for config_key, manifest_key in adamson_outputs.items():
        path = ROOT / adamson[config_key]
        observed = file_sha256(path)
        assert observed == adamson_manifest["outputs"][manifest_key]["sha256"]
        hashes[str(path.relative_to(ROOT))] = observed

    prediction_path = ROOT / adamson["predictions"]
    observed = file_sha256(prediction_path)
    assert observed == prediction_manifest["artifact"]["sha256"]
    hashes[str(prediction_path.relative_to(ROOT))] = observed
    return evaluation, prediction_manifest, hashes


def load_frozen_dynamics(evaluation: dict) -> tuple[torch.nn.Module, torch.nn.Module]:
    models = []
    for key in ("checkpoint_path", "pseudo_paired_checkpoint_path"):
        checkpoint = torch.load(
            ROOT / evaluation["inputs"][key], map_location="cpu", weights_only=False
        )
        model = build_dynamics_model(checkpoint["configuration"]).eval()
        model.load_state_dict(checkpoint["model"])
        models.append(model)
    return models[0], models[1]


def sampled_replogle_populations(
    evaluation: dict, plot_config: dict, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]]]:
    """Run frozen models on a blinded target subset for joint observed/predicted UMAPs."""
    candidate, pseudo = load_frozen_dynamics(evaluation)
    umap_config = plot_config["umap"]
    populations: dict[str, dict[str, np.ndarray]] = {}
    selected_targets: dict[str, list[str]] = {}
    for regime, specification in evaluation["regimes"].items():
        dataset = LatentPopulationDataset(
            ROOT / evaluation["inputs"]["latent_cache_path"],
            ROOT / evaluation["inputs"]["action_cache_path"],
            ROOT / evaluation["inputs"]["dynamics_manifest_path"],
            regime,
            evaluation["sampling"]["population_size"],
            evaluation["seed"],
            specification["outcome_role"],
            specification["control_role"],
            specification["context"],
        )
        dataset.set_epoch(umap_config["sampling_repeat"])
        chosen = blind_target_subset(
            dataset.condition_targets, umap_config["targets_per_regime"], seed, regime
        )
        selected_targets[regime] = chosen
        index = {target: position for position, target in enumerate(dataset.condition_targets)}
        collected: dict[str, list[np.ndarray]] = {
            "Control": [],
            "Observed": [],
            "CausalCellJEPA": [],
            "Pseudo-paired": [],
        }
        for start in range(0, len(chosen), umap_config["model_batch_size"]):
            items = [
                dataset[index[target]]
                for target in chosen[start : start + umap_config["model_batch_size"]]
            ]
            control = torch.stack([item["control"] for item in items])
            action = torch.stack([item["action"] for item in items])
            known = torch.stack([item["action_known"] for item in items])
            with torch.inference_mode():
                candidate_prediction = candidate(control, action, known)
                pseudo_prediction = pseudo(control, action, known)
            collected["Control"].append(control.numpy().reshape(-1, control.shape[-1]))
            observed = torch.stack([item["perturbed"] for item in items])
            collected["Observed"].append(observed.numpy().reshape(-1, observed.shape[-1]))
            collected["CausalCellJEPA"].append(
                candidate_prediction.numpy().reshape(-1, candidate_prediction.shape[-1])
            )
            collected["Pseudo-paired"].append(
                pseudo_prediction.numpy().reshape(-1, pseudo_prediction.shape[-1])
            )
        populations[regime] = {
            source: np.concatenate(values).astype(np.float32)
            for source, values in collected.items()
        }
    return populations, selected_targets


def figure_replogle_umap(
    populations: dict[str, dict[str, np.ndarray]],
    selected: dict[str, list[str]],
    config: dict,
    hashes: dict[str, str],
) -> None:
    source_order = ["Control", "Observed", "CausalCellJEPA", "Pseudo-paired"]
    colors = {"Control": GRAY, "Observed": PURPLE, "CausalCellJEPA": TEAL, "Pseudo-paired": ORANGE}
    labels = {
        "iid": "IID · K562",
        "perturbation_ood": "Perturbation OOD · K562",
        "context_ood": "Context OOD · RPE1",
        "double_ood": "Double OOD · RPE1",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    umap_config = config["replogle"]["umap"]
    for panel, (axis, regime) in enumerate(zip(axes.flat, labels, strict=True)):
        counts = [len(populations[regime][source]) for source in source_order]
        matrix = np.concatenate([populations[regime][source] for source in source_order])
        embedding = umap.UMAP(
            n_neighbors=umap_config["n_neighbors"],
            min_dist=umap_config["min_dist"],
            metric=umap_config["metric"],
            random_state=config["seed"] + panel,
            transform_seed=config["seed"] + panel,
            n_jobs=1,
        ).fit_transform(matrix)
        offset = 0
        for source, count in zip(source_order, counts, strict=True):
            points = embedding[offset : offset + count]
            offset += count
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=5,
                color=colors[source],
                alpha=0.42,
                edgecolors="none",
                rasterized=True,
            )
            centroid = points.mean(0)
            axis.scatter(
                centroid[0],
                centroid[1],
                marker="X",
                s=70,
                color=colors[source],
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
        axis.set_title(
            f"{labels[regime]}\n{len(selected[regime])} blinded targets · {counts[0]:,} cells/source",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        panel_label(axis, chr(97 + panel))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[source],
            markeredgecolor="none",
            markersize=7,
            label=source,
        )
        for source in source_order
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01)
    )
    figure.suptitle(
        "Observed and predicted cell populations in the frozen JEPA latent space",
        x=0.02,
        y=1.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        "Each panel fits a separate joint UMAP to control, observed, and frozen-model cells. "
        "Targets are selected by a seeded identity hash; X marks the 2-D source centroid.",
        fontsize=8.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, "empirical_1_replogle_population_umap", config, hashes)


def paired_axis(
    axis: plt.Axes,
    candidate: dict[str, float],
    reference: dict[str, float],
    title: str,
    x_label: str,
    y_label: str,
    higher_is_better: bool,
    seed: int,
) -> dict:
    targets = sorted(candidate.keys() & reference.keys())
    x = np.asarray([reference[target] for target in targets])
    y = np.asarray([candidate[target] for target in targets])
    better = y > x if higher_is_better else y < x
    axis.scatter(x[~better], y[~better], s=18, color=ORANGE, alpha=0.60, edgecolors="none")
    axis.scatter(x[better], y[better], s=18, color=TEAL, alpha=0.60, edgecolors="none")
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    padding = max((high - low) * 0.06, 1e-3)
    limits = (low - padding, high + padding)
    axis.plot(limits, limits, color=INK, lw=0.9, ls=(0, (4, 3)))
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.grid(color="#DCE2E5", lw=0.55, alpha=0.7)
    axis.set_axisbelow(True)
    benefit = bootstrap_advantage(y, x, higher_is_better, seed)
    axis.text(
        0.03,
        0.97,
        f"candidate better: {better.sum()}/{len(targets)}\n"
        f"mean benefit {benefit[0]:+.3f}\n95% CI [{benefit[1]:+.3f}, {benefit[2]:+.3f}]",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    return {
        "targets": len(targets),
        "candidate_better": int(better.sum()),
        "candidate_mean": float(y.mean()),
        "reference_mean": float(x.mean()),
        "mean_benefit_bootstrap_95ci": list(benefit),
    }


def figure_replogle_paired_scatter(
    rows: list[dict], config: dict, hashes: dict[str, str], report: dict
) -> None:
    regime = config["replogle"]["target_scatter_regime"]
    metrics = [
        ("effect_pearson", "Latent effect Pearson", True),
        ("magnitude_absolute_error", "Magnitude error", False),
        ("mmd", "Maximum mean discrepancy", False),
        ("sinkhorn", "Sinkhorn divergence", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 9.1))
    report["replogle_double_ood_paired_scatter"] = {}
    for panel, (axis, (metric, title, higher)) in enumerate(zip(axes.flat, metrics, strict=True)):
        candidate = condition_values(rows, "causalcelljepa", regime, metric)
        reference = condition_values(rows, "pseudo_paired", regime, metric)
        report["replogle_double_ood_paired_scatter"][metric] = paired_axis(
            axis,
            candidate,
            reference,
            f"{chr(97 + panel)}  {title} {'↑' if higher else '↓'}",
            "Pseudo-paired JEPA",
            "CausalCellJEPA",
            higher,
            config["seed"] + 100 + panel,
        )
    legend = [
        Patch(facecolor=TEAL, label="CausalCellJEPA better"),
        Patch(facecolor=ORANGE, label="Pseudo-paired better or tied"),
    ]
    figure.legend(
        handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01)
    )
    figure.suptitle(
        "Target-paired double-OOD comparison exposes metric-specific strengths",
        x=0.02,
        y=1.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        "Each point is one perturbation condition after averaging eight independent population resamples; "
        "confidence intervals bootstrap target identities.",
        fontsize=8.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(figure, "empirical_2_replogle_target_paired_scatter", config, hashes)


def comparison_index(payload: dict) -> dict[tuple[str, str, str, str], dict]:
    return {
        (item["regime"], item["candidate"], item["reference"], item["metric"]): item
        for item in payload["condition_comparisons"]
    }


def adamson_scatter_axis(
    axis: plt.Axes,
    rows: list[dict],
    preregistered: dict,
    comparisons: dict,
    reference: str,
    metric: str,
    title: str,
) -> dict:
    candidate_name = "control_gated_external_response"
    candidate = condition_values(rows, candidate_name, "all_scored", metric)
    baseline = condition_values(rows, reference, "all_scored", metric)
    targets = sorted(candidate.keys() & baseline.keys())
    x = np.asarray([baseline[target] for target in targets])
    y = np.asarray([candidate[target] for target in targets])
    unseen = set(preregistered["targets"]["outcome_fit_unseen"])
    unseen_mask = np.asarray([target in unseen for target in targets])
    axis.scatter(x[~unseen_mask], y[~unseen_mask], s=30, color=NAVY, alpha=0.72, edgecolors="none")
    axis.scatter(
        x[unseen_mask],
        y[unseen_mask],
        s=42,
        facecolors="white",
        edgecolors=ORANGE,
        linewidth=1.3,
    )
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    padding = max((high - low) * 0.07, 0.02)
    limits = (low - padding, high + padding)
    axis.plot(limits, limits, color=INK, lw=0.9, ls=(0, (4, 3)))
    axis.axhline(0, color=GRAY, lw=0.6, alpha=0.7)
    axis.axvline(0, color=GRAY, lw=0.6, alpha=0.7)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    reference_label = {
        "perturbed_mean": "Perturbed mean",
        "string_kernel_gene_go_rbf": "STRING + GO",
    }[reference]
    axis.set_xlabel(reference_label)
    axis.set_ylabel("CausalCellJEPA")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.grid(color="#DCE2E5", lw=0.55, alpha=0.6)
    axis.set_axisbelow(True)
    item = comparisons["all_scored", candidate_name, reference, metric]
    difference = item["mean_improvement"]
    interval = item["mean_improvement_bootstrap_95ci"]
    axis.text(
        0.03,
        0.97,
        f"paired Δ {difference:+.3f}\n95% CI [{interval[0]:+.3f}, {interval[1]:+.3f}]\n"
        f"candidate better: {int((y > x).sum())}/{len(targets)}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    return {
        "targets": len(targets),
        "candidate_better": int((y > x).sum()),
        "mean_improvement": difference,
        "mean_improvement_bootstrap_95ci": interval,
    }


def figure_adamson_systema_scatter(
    rows: list[dict],
    preregistered: dict,
    comparisons: dict,
    config: dict,
    hashes: dict[str, str],
    report: dict,
) -> None:
    references = ["perturbed_mean", "string_kernel_gene_go_rbf"]
    metrics = ["systema_all_gene_pearson_delta", "systema_target_excluded_pearson_delta"]
    titles = {
        "systema_all_gene_pearson_delta": "Systema Pearson · all genes",
        "systema_target_excluded_pearson_delta": "Systema Pearson · target excluded",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.1, 9.0))
    report["adamson_systema_paired_scatter"] = {}
    panel = 0
    for row, metric in enumerate(metrics):
        for column, reference in enumerate(references):
            title = f"{chr(97 + panel)}  {titles[metric]}\nvs. {'perturbed mean' if reference == 'perturbed_mean' else 'STRING + GO'}"
            key = f"{metric}_vs_{reference}"
            report["adamson_systema_paired_scatter"][key] = adamson_scatter_axis(
                axes[row, column], rows, preregistered, comparisons, reference, metric, title
            )
            panel += 1
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NAVY,
            markeredgecolor="none",
            markersize=7,
            label="Outcome-fit seen",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            markersize=7,
            label="Outcome-fit unseen",
        ),
    ]
    figure.legend(
        handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01)
    )
    figure.suptitle(
        "Adamson external confirmation: target-paired perturbation-specific recovery",
        x=0.02,
        y=1.04,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        "Every preregistered target is shown. The diagonal denotes equal target performance; "
        "intervals are the frozen 10,000-resample target bootstrap.",
        fontsize=8.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(figure, "empirical_3_adamson_systema_paired_scatter", config, hashes)


def adamson_effect_data(
    config: dict, source_hashes: dict[str, str]
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], np.ndarray, dict]:
    adamson = config["adamson"]
    preregistered = yaml.safe_load((ROOT / adamson["preregistered_config"]).read_text())
    replogle = json.loads((ROOT / "manifests/replogle_v1.json").read_text())
    hvg_ids = replogle["genes"]["hvg_gene_ids"]
    expression, metadata, observed = load_adamson_expression(
        preregistered, hvg_ids, {"control", "scored", "reference"}, grouped=True
    )
    for specification in preregistered["source"]["files"].values():
        path = ROOT / preregistered["source"]["raw_directory"] / specification["filename"]
        source_hashes[str(path.relative_to(ROOT))] = specification["sha256"]

    group_sums = {
        group: expression["sums"][index, observed]
        for index, group in enumerate(expression["groups"])
    }
    group_counts = {
        group: expression["counts"][index] for index, group in enumerate(expression["groups"])
    }
    roles, batches = metadata["role"], metadata["batch"]
    control_means = {
        batch: group_sums["control", "control", batch] / group_counts["control", "control", batch]
        for batch in sorted(set(batches[roles == "control"]))
    }
    targets = preregistered["targets"]["scored"]
    truth = []
    for target in targets:
        target_groups = {
            batch: group_counts["scored", target, batch]
            for role, group_target, batch in group_counts
            if role == "scored" and group_target == target
        }
        centroid = sum(
            values
            for (role, group_target, _), values in group_sums.items()
            if role == "scored" and group_target == target
        ) / sum(target_groups.values())
        matched_control = sum(
            count * control_means[batch] for batch, count in target_groups.items()
        ) / sum(target_groups.values())
        truth.append(centroid - matched_control)
    truth_array = np.stack(truth)

    prediction = torch.load(ROOT / adamson["predictions"], map_location="cpu", weights_only=True)
    assert prediction["targets"] == targets
    predicted = {
        model: values.numpy()[:, observed] for model, values in prediction["effects"].items()
    }
    gene_names = np.asarray(replogle["genes"]["hvg_gene_names"])[observed]
    return targets, truth_array, predicted, gene_names, preregistered


def figure_adamson_effect_heatmap(
    targets: list[str],
    truth: np.ndarray,
    predicted: dict[str, np.ndarray],
    genes: np.ndarray,
    preregistered: dict,
    config: dict,
    hashes: dict[str, str],
    report: dict,
) -> None:
    variances = truth.var(0)
    count = config["adamson"]["heatmap_genes"]
    gene_indices = sorted(
        range(len(genes)), key=lambda index: (-variances[index], str(genes[index]))
    )[:count]
    seen = set(preregistered["targets"]["outcome_fit_seen"])
    order = sorted(
        range(len(targets)), key=lambda index: (targets[index] not in seen, targets[index])
    )
    observed = truth[np.ix_(order, gene_indices)]
    candidate = predicted["control_gated_external_response"][np.ix_(order, gene_indices)]
    residual = candidate - observed
    effect_limit = float(
        np.quantile(np.abs(np.concatenate([observed.ravel(), candidate.ravel()])), 0.98)
    )
    residual_limit = float(np.quantile(np.abs(residual), 0.98))

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.0), gridspec_kw={"wspace": 0.08})
    images = []
    for panel, (axis, matrix, title, limit) in enumerate(
        zip(
            axes,
            (observed, candidate, residual),
            ("Observed effect", "CausalCellJEPA effect", "Prediction residual"),
            (effect_limit, effect_limit, residual_limit),
            strict=True,
        )
    ):
        image = axis.imshow(
            matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest"
        )
        images.append(image)
        axis.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontsize=11, fontweight="bold")
        axis.set_xticks(
            range(count), [str(genes[index]) for index in gene_indices], rotation=90, fontsize=6.5
        )
        if panel == 0:
            ordered_targets = [targets[index] for index in order]
            axis.set_yticks(range(len(order)), ordered_targets, fontsize=7.5)
            for tick, target in zip(axis.get_yticklabels(), ordered_targets, strict=True):
                tick.set_color(NAVY if target in seen else ORANGE)
        else:
            axis.set_yticks([])
        boundary = len(seen) - 0.5
        axis.axhline(boundary, color=INK, lw=1.1)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.colorbar(
        images[0], ax=axes[:2], fraction=0.025, pad=0.02, label="log-normalized expression shift"
    )
    figure.colorbar(images[2], ax=axes[2], fraction=0.05, pad=0.02, label="predicted − observed")
    figure.suptitle(
        "Adamson gene-effect structure across preregistered perturbations",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.015,
        "Columns are the 30 genes with highest observed between-target variance (descriptive visualization only). "
        "Navy labels: outcome-fit seen; orange labels: outcome-fit unseen.",
        fontsize=8.5,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.08, right=0.95, bottom=0.23, top=0.91)
    save_figure(figure, "empirical_4_adamson_gene_effect_heatmap", config, hashes)
    report["adamson_effect_heatmap"] = {
        "selection": "highest observed between-target variance; reporting only",
        "genes": [str(genes[index]) for index in gene_indices],
        "seen_targets": len(seen),
        "unseen_targets": len(targets) - len(seen),
    }


def row_correlations(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    centered_truth = truth - truth.mean(1, keepdims=True)
    centered_prediction = prediction - prediction.mean(1, keepdims=True)
    denominator = np.linalg.norm(centered_truth, axis=1) * np.linalg.norm(
        centered_prediction, axis=1
    )
    return np.divide(
        (centered_truth * centered_prediction).sum(1),
        denominator,
        out=np.full(len(truth), np.nan),
        where=denominator > 1e-12,
    )


def figure_adamson_effect_density(
    truth: np.ndarray,
    predicted: dict[str, np.ndarray],
    config: dict,
    hashes: dict[str, str],
    report: dict,
) -> None:
    models = [
        ("control_gated_external_response", "CausalCellJEPA"),
        ("external_response_multiview_rbf", "External-response predictor"),
        ("string_kernel_gene_go_rbf", "STRING + GO predictor"),
    ]
    maximum = max(
        float(np.abs(truth).max()),
        *(float(np.abs(predicted[model]).max()) for model, _ in models),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharex=True, sharey=True)
    report["adamson_gene_effect_density"] = {}
    for panel, (axis, (model, label)) in enumerate(zip(axes, models, strict=True)):
        values = predicted[model]
        density = axis.hexbin(
            truth.ravel(),
            values.ravel(),
            gridsize=75,
            extent=(-maximum, maximum, -maximum, maximum),
            bins="log",
            mincnt=1,
            cmap="viridis",
            rasterized=True,
        )
        axis.plot([-maximum, maximum], [-maximum, maximum], color="white", lw=1.0, ls=(0, (4, 3)))
        correlations = row_correlations(truth, values)
        axis.text(
            0.04,
            0.96,
            f"median target Pearson {np.nanmedian(correlations):+.3f}",
            transform=axis.transAxes,
            va="top",
            color="white",
            fontsize=8.5,
            bbox={"facecolor": INK, "edgecolor": "none", "alpha": 0.70, "pad": 3},
        )
        axis.set_title(f"{chr(97 + panel)}  {label}", loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel("Observed gene effect")
        if panel == 0:
            axis.set_ylabel("Predicted gene effect")
        axis.set_aspect("equal", adjustable="box")
        report["adamson_gene_effect_density"][model] = {
            "target_median_pearson": float(np.nanmedian(correlations)),
            "target_mean_pearson": float(np.nanmean(correlations)),
            "targets": len(correlations),
            "target_gene_pairs": int(truth.size),
        }
    figure.colorbar(density, ax=axes, fraction=0.025, pad=0.02, label="log₁₀ target–gene count")
    figure.suptitle(
        "Adamson observed-versus-predicted gene-effect density",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.02,
        -0.01,
        "All observed target–gene pairs are shown as hexagonal density. This panel is descriptive; "
        "reported inferential uncertainty remains perturbation-target-level.",
        fontsize=8.5,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.06, right=0.92, bottom=0.14, top=0.88, wspace=0.20)
    save_figure(figure, "empirical_5_adamson_gene_effect_density", config, hashes)


def main(config_path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(config_path.read_text())
    style()
    evaluation, prediction_manifest, source_hashes = verify_sources(config)
    output_hashes: dict[str, str] = {}
    report: dict = {
        "format_version": 1,
        "reporting_only": True,
        "fit_selection_or_tuning_performed": False,
        "statistical_unit": "perturbation condition",
        "source_hashes": source_hashes,
        "software": {
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "umap_learn": umap.__version__,
        },
    }

    replogle_rows = load_jsonl(ROOT / config["replogle"]["condition_metrics"])
    populations, selected = sampled_replogle_populations(
        evaluation, config["replogle"], config["seed"]
    )
    report["replogle_umap"] = {
        "selection": "seeded SHA-256 target-identity priority; no outcomes or metrics read",
        "population_repeat": config["replogle"]["umap"]["sampling_repeat"],
        "targets": selected,
    }
    figure_replogle_umap(populations, selected, config, output_hashes)
    figure_replogle_paired_scatter(replogle_rows, config, output_hashes, report)

    preregistered = yaml.safe_load((ROOT / config["adamson"]["preregistered_config"]).read_text())
    adamson_rows = load_jsonl(ROOT / config["adamson"]["condition_metrics"])
    comparisons_payload = json.loads((ROOT / config["adamson"]["paired_comparisons"]).read_text())
    figure_adamson_systema_scatter(
        adamson_rows,
        preregistered,
        comparison_index(comparisons_payload),
        config,
        output_hashes,
        report,
    )
    targets, truth, predicted, genes, preregistered = adamson_effect_data(config, source_hashes)
    assert prediction_manifest["prediction"]["targets"] == len(targets)
    figure_adamson_effect_heatmap(
        targets, truth, predicted, genes, preregistered, config, output_hashes, report
    )
    figure_adamson_effect_density(truth, predicted, config, output_hashes, report)

    report["source_hashes"] = dict(sorted(source_hashes.items()))
    report["output_hashes"] = dict(sorted(output_hashes.items()))
    manifest_path = ROOT / config["output_directory"] / "empirical_figure_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    arguments = parser.parse_args()
    result = main(arguments.config)
    print(
        json.dumps(
            {
                "figures": len(result["output_hashes"]),
                "manifest": str(
                    Path(arguments.config).parent.parent
                    / result.get("manifest", "figures/empirical/empirical_figure_manifest.json")
                ),
            },
            sort_keys=True,
        )
    )
