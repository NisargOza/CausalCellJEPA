# Generate manuscript-ready vector and raster figures from frozen evaluation artifacts.
# The script verifies every source hash and bootstraps perturbation identities, never cells.

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/publication_figures.yaml"
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


def condition_values(rows: list[dict], model: str, regime: str, metric: str) -> dict[str, float]:
    """Average sampling repeats so perturbation-condition remains the statistical unit."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["model"] == model and row["regime"] == regime:
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                grouped[row["target"]].append(float(value))
    return {target: float(np.mean(values)) for target, values in grouped.items()}


def bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Return condition mean and percentile interval from deterministic condition resampling."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def paired_difference(
    candidate: dict[str, float], reference: dict[str, float], resamples: int, seed: int
) -> tuple[float, float, float]:
    targets = sorted(candidate.keys() & reference.keys())
    differences = np.array([candidate[target] - reference[target] for target in targets])
    return bootstrap_mean(differences, resamples, seed)


def load_sources(config: dict) -> tuple[dict[str, object], dict[str, str]]:
    loaded: dict[str, object] = {}
    hashes: dict[str, str] = {}
    for name, source in config["sources"].items():
        path = ROOT / source["path"]
        manifest = json.loads((ROOT / source["manifest"]).read_text())
        expected: object = manifest
        for key in source["sha256_keys"]:
            expected = expected[key]
        observed = file_sha256(path)
        assert observed == expected, f"hash mismatch for {path}"
        hashes[source["path"]] = observed
        if path.suffix == ".jsonl":
            loaded[name] = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            loaded[name] = json.loads(path.read_text())
    return loaded, hashes


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
            "svg.hashsalt": "causalcelljepa",
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def diagram_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="none",
            alpha=0.96,
        )
    )
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize
    )


def arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY
) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=color)
    )


def cell_cloud(ax: plt.Axes, center: tuple[float, float], color: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for dx, dy in rng.normal(0, [0.025, 0.035], size=(16, 2)):
        ax.add_patch(
            Circle(
                (center[0] + dx, center[1] + dy), 0.008, facecolor=color, edgecolor="white", lw=0.4
            )
        )


def save_figure(fig: plt.Figure, stem: str, config: dict, output_hashes: dict[str, str]) -> None:
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
            path,
            dpi=config["dpi"],
            bbox_inches="tight",
            pad_inches=0.08,
            metadata=metadata,
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        output_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    plt.close(fig)


def figure_architecture(config: dict, output_hashes: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "CausalCellJEPA", fontsize=18, fontweight="bold")
    ax.text(
        0.02,
        0.915,
        "Multimodal joint-embedding prediction of unpaired cellular perturbation responses",
        fontsize=11,
        color=GRAY,
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.12),
            0.32,
            0.73,
            boxstyle="round,pad=0.015",
            facecolor="#EEF4F7",
            edgecolor="#C9D8E2",
        )
    )
    ax.text(
        0.04, 0.81, "Stage 1  ·  state representation", fontsize=11, fontweight="bold", color=NAVY
    )
    cell_cloud(ax, (0.085, 0.58), SKY, 1)
    ax.text(0.085, 0.48, "sparse cell", ha="center", fontsize=8)
    diagram_box(ax, (0.14, 0.53), 0.075, 0.10, "masked\nstudent", "#BBD6E8")
    diagram_box(ax, (0.25, 0.53), 0.07, 0.10, "latent\npredictor", "#A8D8D0")
    arrow(ax, (0.11, 0.58), (0.14, 0.58))
    arrow(ax, (0.215, 0.58), (0.25, 0.58))
    diagram_box(ax, (0.14, 0.28), 0.18, 0.10, "EMA teacher · full cell", "#DCCFEB")
    arrow(ax, (0.23, 0.38), (0.285, 0.53), PURPLE)
    ax.text(
        0.18,
        0.20,
        "256-D frozen cell state",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )
    arrow(ax, (0.34, 0.24), (0.40, 0.24), NAVY)

    ax.add_patch(
        FancyBboxPatch(
            (0.40, 0.12),
            0.58,
            0.73,
            boxstyle="round,pad=0.015",
            facecolor="#FAF8F1",
            edgecolor="#E8DDC5",
        )
    )
    ax.text(
        0.42,
        0.81,
        "Stage 2  ·  action-conditioned population transition",
        fontsize=11,
        fontweight="bold",
        color="#8A5A20",
    )
    cell_cloud(ax, (0.47, 0.59), TEAL, 2)
    ax.text(0.47, 0.49, "control population", ha="center", fontsize=8)
    diagram_box(ax, (0.53, 0.54), 0.10, 0.10, "Set\nTransformer", "#A8D8D0")
    arrow(ax, (0.495, 0.59), (0.53, 0.59))
    diagram_box(ax, (0.43, 0.27), 0.07, 0.09, "ESM-2", "#BBD6E8")
    diagram_box(ax, (0.515, 0.27), 0.07, 0.09, "GO", "#F4DFA8")
    diagram_box(ax, (0.60, 0.27), 0.07, 0.09, "STRING*", "#DCCFEB")
    ax.text(0.55, 0.20, "frozen biological teachers", ha="center", fontsize=8, color=GRAY)
    diagram_box(ax, (0.67, 0.48), 0.12, 0.14, "biology-aware\naction fusion", "#F2C27D")
    for x in [0.465, 0.55, 0.635]:
        arrow(ax, (x, 0.36), (0.69, 0.49), GOLD)
    arrow(ax, (0.63, 0.59), (0.67, 0.56), TEAL)
    diagram_box(ax, (0.82, 0.48), 0.12, 0.14, "residual set\ntransition", "#F1B4A2")
    arrow(ax, (0.79, 0.55), (0.82, 0.55), ORANGE)
    cell_cloud(ax, (0.89, 0.27), ORANGE, 3)
    ax.text(0.89, 0.18, "predicted population", ha="center", fontsize=8)
    arrow(ax, (0.88, 0.48), (0.89, 0.33), ORANGE)
    cell_cloud(ax, (0.76, 0.27), PURPLE, 4)
    ax.text(0.76, 0.18, "observed population", ha="center", fontsize=8)
    ax.plot([0.785, 0.86], [0.27, 0.27], color=INK, lw=1.4, ls=(0, (3, 2)))
    ax.text(0.823, 0.29, "Sinkhorn · MMD", ha="center", fontsize=7.5)
    ax.text(
        0.42,
        0.135,
        "Independent sets: no arrows pair individual control and perturbed cells.",
        fontsize=8.5,
        color=GRAY,
    )
    ax.text(0.98, 0.135, "*external response extension", ha="right", fontsize=7.5, color=GRAY)
    ax.text(0.98, 0.04, "Figure 1", ha="right", fontsize=8, color=GRAY)
    save_figure(fig, "figure_1_architecture", config, output_hashes)


def figure_evaluation_design(config: dict, output_hashes: dict[str, str]) -> None:
    fig = plt.figure(figsize=(12.2, 7.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.30)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Seen target", "Unseen target"])
    ax.set_yticks([1.5, 0.5], ["Seen response context", "Unseen response context"])
    ax.tick_params(length=0, pad=10)
    colors = [["#DDECE8", "#D9E8F2"], ["#F6EBD1", "#F3D8CF"]]
    labels = [
        ["IID\nK562 · training targets", "Perturbation OOD\nK562 · sealed targets"],
        ["Context OOD\nRPE1 · known targets", "Double OOD\nRPE1 · sealed targets"],
    ]
    for row in range(2):
        for col in range(2):
            y = 1 - row
            ax.add_patch(
                FancyBboxPatch(
                    (col + 0.04, y + 0.05),
                    0.92,
                    0.90,
                    boxstyle="round,pad=0.01",
                    facecolor=colors[row][col],
                    edgecolor="white",
                )
            )
            ax.text(
                col + 0.5,
                y + 0.57,
                labels[row][col],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold" if (row, col) == (1, 1) else "normal",
            )
            ax.text(
                col + 0.5,
                y + 0.30,
                "perturbation-condition bootstrap",
                ha="center",
                fontsize=8,
                color=GRAY,
            )
    ax.set_title(
        "Four preregistered Replogle generalization regimes",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(grid[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    panel_label(ax2, "b")
    ax2.text(0.0, 0.93, "One-shot Adamson external confirmation", fontsize=14, fontweight="bold")
    timeline = [
        (0.03, "protocol\ncommitted", NAVY),
        (0.25, "controls only\n5,241 cells", TEAL),
        (0.47, "27 target\npredictions frozen", GOLD),
        (0.69, "outcomes opened\nexactly once", ORANGE),
        (0.91, "terminal\ndecision", PURPLE),
    ]
    for index, (x, text, color) in enumerate(timeline):
        ax2.add_patch(Circle((x, 0.50), 0.045, facecolor=color, edgecolor="white", lw=1.5))
        ax2.text(x, 0.23 if index % 2 else 0.75, text, ha="center", va="center", fontsize=9)
        if index < len(timeline) - 1:
            arrow(ax2, (x + 0.05, 0.50), (timeline[index + 1][0] - 0.05, 0.50))
    ax2.text(
        0.50,
        0.04,
        "55 disjoint reference targets estimate systematic perturbed variation · no post-outcome tuning",
        ha="center",
        fontsize=8.5,
        color=GRAY,
    )
    save_figure(fig, "figure_2_evaluation_design", config, output_hashes)


def distribution_panel(
    ax: plt.Axes,
    values: dict[str, dict[str, float]],
    labels: dict[str, str],
    colors: dict[str, str],
    x_label: str,
    resamples: int,
    seed: int,
    summary_override: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    order = list(values)
    summary: dict[str, list[float]] = {}
    jitter_rng = np.random.default_rng(seed)
    for index, model in enumerate(order):
        array = np.array(list(values[model].values()))
        mean, low, high = (
            summary_override[model]
            if summary_override is not None
            else bootstrap_mean(array, resamples, seed + index + 1)
        )
        y = len(order) - 1 - index
        jitter = jitter_rng.normal(0, 0.075, len(array))
        ax.scatter(
            array,
            y + jitter,
            s=7,
            color=colors[model],
            alpha=0.13,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([low, high], [y, y], color=colors[model], lw=2.2, solid_capstyle="round")
        ax.scatter([mean], [y], s=42, color=colors[model], edgecolor="white", lw=0.7, zorder=4)
        summary[model] = [mean, low, high]
    ax.set_yticks(range(len(order) - 1, -1, -1), [labels[model] for model in order])
    ax.set_xlabel(x_label)
    ax.grid(axis="x", color="#DCE2E5", lw=0.7)
    ax.set_axisbelow(True)
    return summary


def figure_replogle_tradeoff(
    data: dict, config: dict, output_hashes: dict[str, str], report: dict
) -> None:
    primary = data["replogle_latent"]
    ablations = data["replogle_ablations"]
    models = [
        "causalcelljepa",
        "pseudo_paired",
        "mean_context",
        "no_global_context",
        "no_direction_loss",
    ]
    labels = {
        "causalcelljepa": "CausalCellJEPA",
        "pseudo_paired": "Pseudo-paired JEPA",
        "mean_context": "Mean context",
        "no_global_context": "No global context",
        "no_direction_loss": "No direction loss",
    }
    colors = {
        "causalcelljepa": NAVY,
        "pseudo_paired": GRAY,
        "mean_context": TEAL,
        "no_global_context": GOLD,
        "no_direction_loss": ORANGE,
    }
    metrics = [
        ("effect_pearson", "Latent effect Pearson ↑"),
        ("magnitude_absolute_error", "Magnitude error ↓"),
        ("mmd", "MMD ↓"),
        ("sinkhorn", "Sinkhorn divergence ↓"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.3, 4.5))
    report["replogle_double_ood"] = {}
    for panel, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        values = {}
        for model in models:
            rows = primary if model in {"causalcelljepa", "pseudo_paired"} else ablations
            values[model] = condition_values(rows, model, "double_ood", metric)
        report["replogle_double_ood"][metric] = distribution_panel(
            ax,
            values,
            labels,
            colors,
            title,
            config["bootstrap_resamples"],
            config["seed"] + panel * 100,
        )
        ax.set_title(chr(97 + panel), loc="left", fontweight="bold", fontsize=12)
        if metric == "effect_pearson":
            ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
        if panel:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Double-OOD population modeling exposes a direction–calibration trade-off",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.01,
        "Dots are perturbation conditions after averaging eight population resamples; intervals bootstrap target identities (10,000 resamples).",
        fontsize=8.5,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save_figure(fig, "figure_3_replogle_distributional_tradeoff", config, output_hashes)


def figure_seed_transfer(
    data: dict, config: dict, output_hashes: dict[str, str], report: dict
) -> None:
    latent_primary = data["replogle_latent"]
    latent_replicates = data["replogle_replication"]
    gene_primary = data["replogle_transcriptomics"]
    gene_replicates = data["replogle_replication_transcriptomics"]
    models = ["causalcelljepa", "seed_20260824", "seed_20260825", "linear_esm"]
    labels = {
        "causalcelljepa": "JEPA seed 1",
        "seed_20260824": "JEPA seed 2",
        "seed_20260825": "JEPA seed 3",
        "linear_esm": "Linear ESM",
    }
    colors = {
        "causalcelljepa": NAVY,
        "seed_20260824": TEAL,
        "seed_20260825": SKY,
        "linear_esm": GRAY,
    }
    panels = [
        ("context_ood", "effect_pearson", "Latent · context OOD"),
        ("double_ood", "effect_pearson", "Latent · double OOD"),
        ("context_ood", "all_effect_pearson", "Decoded genes · context OOD"),
        ("double_ood", "all_effect_pearson", "Decoded genes · double OOD"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.1), sharex=False)
    report["seed_transfer"] = {}
    for panel, (ax, (regime, metric, title)) in enumerate(zip(axes, panels)):
        values = {}
        for model in models:
            if metric == "effect_pearson":
                rows = latent_replicates if model.startswith("seed_") else latent_primary
            else:
                rows = gene_replicates if model.startswith("seed_") else gene_primary
            values[model] = condition_values(rows, model, regime, metric)
        report["seed_transfer"][title] = distribution_panel(
            ax,
            values,
            labels,
            colors,
            "Effect Pearson ↑",
            config["bootstrap_resamples"],
            config["seed"] + 500 + panel * 100,
        )
        ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"{chr(97 + panel)}  {title}", loc="left", fontsize=10.5, fontweight="bold")
        if panel:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Two replicate seeds recover latent transfer, but the frozen decoder does not",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.01,
        "The discrepancy localizes a major bottleneck to transcriptomic readout rather than hiding seed sensitivity.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    save_figure(fig, "figure_4_seed_and_readout_transfer", config, output_hashes)


def paired_scatter(
    ax: plt.Axes,
    x: dict[str, float],
    y: dict[str, float],
    seen: set[str],
    x_label: str,
    y_label: str,
) -> None:
    targets = sorted(x.keys() & y.keys())
    xv = np.array([x[target] for target in targets])
    yv = np.array([y[target] for target in targets])
    bounds = [min(xv.min(), yv.min()) - 0.04, max(xv.max(), yv.max()) + 0.04]
    ax.plot(bounds, bounds, color=GRAY, lw=1, ls=(0, (3, 2)))
    point_colors = [TEAL if target in seen else ORANGE for target in targets]
    ax.scatter(xv, yv, c=point_colors, s=34, edgecolor="white", lw=0.5, alpha=0.92)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color="#E1E6E8", lw=0.6)
    ax.set_axisbelow(True)
    for index in np.argsort(np.abs(yv - xv))[-2:]:
        ax.annotate(
            targets[index],
            (xv[index], yv[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )


def figure_adamson(data: dict, config: dict, output_hashes: dict[str, str], report: dict) -> None:
    rows = data["adamson_conditions"]
    models = [
        "control_gated_external_response",
        "external_response_multiview_rbf",
        "string_kernel_gene_go_rbf",
        "direct_gene_esm",
        "mean_effect",
        "no_change",
        "perturbed_mean",
    ]
    labels = {
        "control_gated_external_response": "Frozen gated candidate",
        "external_response_multiview_rbf": "External-response student",
        "string_kernel_gene_go_rbf": "STRING + GO",
        "direct_gene_esm": "Direct ESM ridge",
        "mean_effect": "Mean training effect",
        "no_change": "No change",
        "perturbed_mean": "Perturbed mean",
    }
    colors = {
        "control_gated_external_response": NAVY,
        "external_response_multiview_rbf": SKY,
        "string_kernel_gene_go_rbf": TEAL,
        "direct_gene_esm": PURPLE,
        "mean_effect": GOLD,
        "no_change": GRAY,
        "perturbed_mean": ORANGE,
    }
    fig = plt.figure(figsize=(13.4, 9.2))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)
    report["adamson"] = {}
    frozen_summary = {
        (item["model"], item["metric"]): [item["mean"], *item["mean_bootstrap_95ci"]]
        for item in data["adamson_summary"]["condition_metrics"]
        if item["regime"] == "all_scored"
    }
    for panel, metric in enumerate(
        ["systema_all_gene_pearson_delta", "systema_target_excluded_pearson_delta"]
    ):
        ax = fig.add_subplot(grid[0, panel])
        values = {model: condition_values(rows, model, "all_scored", metric) for model in models}
        summary_override = {model: frozen_summary[model, metric] for model in models}
        report["adamson"][metric] = distribution_panel(
            ax,
            values,
            labels,
            colors,
            "Systema Pearson Δ ↑",
            config["bootstrap_resamples"],
            config["seed"] + 1000 + panel * 100,
            summary_override,
        )
        ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(
            f"{chr(97 + panel)}  {'All genes' if panel == 0 else 'Target excluded'}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        if panel:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    candidate = condition_values(rows, models[0], "all_scored", "systema_all_gene_pearson_delta")
    perturbed = condition_values(
        rows, "perturbed_mean", "all_scored", "systema_all_gene_pearson_delta"
    )
    string_go = condition_values(
        rows, "string_kernel_gene_go_rbf", "all_scored", "systema_all_gene_pearson_delta"
    )
    seen = {
        row["target"]
        for row in rows
        if row["model"] == models[0] and row["regime"] == "outcome_fit_seen"
    }
    comparison_index = {
        (item["candidate"], item["reference"], item["metric"]): item
        for item in data["adamson_comparisons"]["condition_comparisons"]
        if item["regime"] == "all_scored"
    }
    ax = fig.add_subplot(grid[1, 0])
    paired_scatter(ax, perturbed, candidate, seen, "Perturbed mean", "Frozen gated candidate")
    frozen_improvement = comparison_index[
        "control_gated_external_response",
        "perturbed_mean",
        "systema_all_gene_pearson_delta",
    ]
    improvement = (
        frozen_improvement["mean_improvement"],
        *frozen_improvement["mean_improvement_bootstrap_95ci"],
    )
    report["adamson"]["candidate_minus_perturbed_mean"] = improvement
    ax.text(
        0.03,
        0.96,
        f"mean Δ = {improvement[0]:+.3f}\n95% CI [{improvement[1]:+.3f}, {improvement[2]:+.3f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_title("c  Perturbation-specific gain", loc="left", fontsize=11, fontweight="bold")
    ax = fig.add_subplot(grid[1, 1])
    paired_scatter(ax, string_go, candidate, seen, "STRING + GO", "Frozen gated candidate")
    frozen_loss = comparison_index[
        "control_gated_external_response",
        "string_kernel_gene_go_rbf",
        "systema_all_gene_pearson_delta",
    ]
    loss = (frozen_loss["mean_improvement"], *frozen_loss["mean_improvement_bootstrap_95ci"])
    report["adamson"]["candidate_minus_string_go"] = loss
    ax.text(
        0.03,
        0.96,
        f"mean Δ = {loss[0]:+.3f}\n95% CI [{loss[1]:+.3f}, {loss[2]:+.3f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_title("d  Best-component shortfall", loc="left", fontsize=11, fontweight="bold")
    fig.suptitle(
        "One-shot Adamson confirmation: real signal, but the frozen gate misses the best component",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Teal: outcome-fit seen (n=19) · orange: outcome-fit unseen (n=8). Points and intervals use perturbation conditions, not cells.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.88, hspace=0.38, wspace=0.30)
    save_figure(fig, "figure_5_adamson_external_confirmation", config, output_hashes)


def figure_generalization(
    data: dict, config: dict, output_hashes: dict[str, str], report: dict
) -> None:
    rows = data["adamson_conditions"]
    candidate_name = "control_gated_external_response"
    candidate = condition_values(
        rows, candidate_name, "all_scored", "systema_all_gene_pearson_delta"
    )
    perturbed = condition_values(
        rows, "perturbed_mean", "all_scored", "systema_all_gene_pearson_delta"
    )
    seen = {
        row["target"]
        for row in rows
        if row["model"] == candidate_name and row["regime"] == "outcome_fit_seen"
    }
    ordered = sorted(candidate, key=candidate.get)
    fig = plt.figure(figsize=(13.3, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(grid[0, :])
    values = np.array([candidate[target] for target in ordered])
    colors = [TEAL if target in seen else ORANGE for target in ordered]
    ax.vlines(np.arange(len(ordered)), 0, values, color=colors, alpha=0.55, lw=1.4)
    ax.scatter(np.arange(len(ordered)), values, c=colors, s=34, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(np.arange(len(ordered)), ordered, rotation=62, ha="right", fontsize=7.3)
    ax.set_ylabel("Candidate Systema Pearson Δ")
    ax.set_title("a  Target-level external performance", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#E1E6E8", lw=0.6)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 0])
    improvements = {target: candidate[target] - perturbed[target] for target in candidate}
    order = sorted(improvements, key=improvements.get)
    imp = np.array([improvements[target] for target in order])
    imp_colors = [TEAL if target in seen else ORANGE for target in order]
    ax.barh(np.arange(len(order)), imp, color=imp_colors, alpha=0.85)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(np.arange(len(order)), order, fontsize=6.8)
    ax.set_xlabel("Candidate − perturbed mean")
    ax.set_title(
        "b  Condition-level baseline improvement", loc="left", fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", color="#E1E6E8", lw=0.6)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    decision = data["adamson_decision"]
    criteria = [
        (
            "Systema all-gene CI > 0",
            decision["criteria"]["systema_all_gene_ci_lower_above_zero_vs_perturbed_mean"],
        ),
        (
            "Target-excluded CI > 0",
            decision["criteria"]["systema_target_excluded_ci_lower_above_zero_vs_perturbed_mean"],
        ),
        (
            "Centroid accuracy > baseline",
            decision["criteria"]["centroid_accuracy_above_perturbed_mean"],
        ),
        (
            "Systema loss ≤ 0.01",
            decision["criteria"]["systema_loss_vs_best_component_at_most_0.01"],
        ),
        (
            "Centroid loss ≤ 0.02",
            decision["criteria"]["centroid_accuracy_loss_vs_best_component_at_most_0.02"],
        ),
        (
            "Magnitude degradation ≤ 2%",
            decision["criteria"]["magnitude_error_degradation_vs_best_component_at_most_2_percent"],
        ),
    ]
    ax.text(0.0, 1.02, "c  Preregistered stopping decision", fontsize=11, fontweight="bold")
    for index, (label, passed) in enumerate(criteria):
        y = 0.86 - index * 0.13
        color = TEAL if passed else ORANGE
        ax.add_patch(Circle((0.04, y), 0.030, facecolor=color, edgecolor="none"))
        ax.text(
            0.04,
            y - 0.002,
            "✓" if passed else "×",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        ax.text(0.10, y, label, va="center", fontsize=9)
    ax.text(
        0.0,
        0.02,
        "5/6 pass  →  architecture search stops\nGlobal SOTA is not supported",
        fontsize=11,
        fontweight="bold",
        color=ORANGE,
    )
    report["adamson"]["criteria_passed"] = sum(value for _, value in criteria)
    fig.suptitle(
        "External generalization is target-dependent and sets the claim boundary",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Teal: outcomes contributed to response fitting · orange: outcome-fit unseen. Negative absolute scores remain visible.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.87, hspace=0.38, wspace=0.32)
    save_figure(fig, "figure_6_generalization_and_claim_audit", config, output_hashes)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["format_version"] == 1
    style()
    data, source_hashes = load_sources(config)
    output_hashes: dict[str, str] = {}
    report: dict[str, object] = {
        "format_version": 1,
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_sha256": source_hashes,
        "bootstrap": {
            "resamples": config["bootstrap_resamples"],
            "seed": config["seed"],
            "unit": "perturbation_condition_after_repeat_averaging",
        },
        "claim_boundary": "derived reporting only; no model selection, fitting, or post-outcome tuning",
    }
    figure_architecture(config, output_hashes)
    figure_evaluation_design(config, output_hashes)
    figure_replogle_tradeoff(data, config, output_hashes, report)
    figure_seed_transfer(data, config, output_hashes, report)
    figure_adamson(data, config, output_hashes, report)
    figure_generalization(data, config, output_hashes, report)
    report["output_sha256"] = output_hashes
    output = ROOT / config["output_directory"] / "figure_manifest.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"figures": len(output_hashes), "manifest": str(output.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
