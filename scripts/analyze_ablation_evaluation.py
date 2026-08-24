# Summarize frozen mechanism ablations without rerunning inference or selecting on test outcomes.
import json
from pathlib import Path

import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config = yaml.safe_load(Path("configs/evaluation_ablations.yaml").read_text())
root = Path(config["output_directory"])
summary_path = root / "summary.json"
comparisons_path = root / "paired_comparisons.json"
provenance_path = root / "provenance.json"
summary = json.loads(summary_path.read_text())
comparisons = json.loads(comparisons_path.read_text())
provenance = json.loads(provenance_path.read_text())
assert provenance["git"]["dirty"] is False
assert provenance["maximum_conditions_per_regime"] is None
assert provenance["executed_repeats"] == 8

models = [
    "causalcelljepa",
    "no_global_context",
    "mean_context",
    "no_direction_loss",
    "pseudo_paired",
]
metrics = ["effect_pearson", "sinkhorn", "mmd", "magnitude_absolute_error"]
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["condition_metrics"]
}
headline_means = {
    regime: {
        model: {metric: means[regime, model, metric] for metric in metrics}
        for model in models
    }
    for regime in regimes
}
retrieval = {
    regime: {
        item["model"]: {
            "mean_reciprocal_rank": item["mean_reciprocal_rank"],
            "median_rank": item["median_rank"],
            "top_1": item["top_1"],
            "top_5": item["top_5"],
        }
        for item in summary["retrieval"]
        if item["regime"] == regime and item["model"] in models
    }
    for regime in regimes
}
comparison_map = {
    (item["candidate"], item["reference"], item["regime"], item["metric"]): item
    for item in comparisons
}
primary_pairs = [
    ("no_global_context", "causalcelljepa"),
    ("mean_context", "causalcelljepa"),
    ("mean_context", "no_global_context"),
    ("no_direction_loss", "causalcelljepa"),
]
primary = {}
for candidate, reference in primary_pairs:
    pair_key = f"{candidate}_vs_{reference}"
    primary[pair_key] = {}
    for regime in regimes:
        primary[pair_key][regime] = {}
        for metric in metrics:
            item = comparison_map[candidate, reference, regime, metric]
            primary[pair_key][regime][metric] = {
                "candidate_mean": item["candidate_mean"],
                "reference_mean": item["reference_mean"],
                "mean_improvement": item["mean_improvement"],
                "mean_improvement_bootstrap_95ci": item[
                    "mean_improvement_bootstrap_95ci"
                ],
                "benjamini_hochberg_q": item["benjamini_hochberg_q"],
            }


def interval(candidate, reference, regime, metric):
    return comparison_map[candidate, reference, regime, metric][
        "mean_improvement_bootstrap_95ci"
    ]


assert all(
    interval("no_global_context", "causalcelljepa", regime, "effect_pearson")[0] > 0
    for regime in ("context_ood", "double_ood")
)
assert all(
    interval("no_global_context", "causalcelljepa", regime, "sinkhorn")[1] < 0
    for regime in ("context_ood", "double_ood")
)
assert all(
    interval("mean_context", "causalcelljepa", regime, "effect_pearson")[0] > 0
    for regime in ("iid", "perturbation_ood")
)
assert interval("mean_context", "causalcelljepa", "context_ood", "effect_pearson")[1] < 0
assert (
    interval("no_direction_loss", "causalcelljepa", "perturbation_ood", "effect_pearson")[1]
    < 0
)
assert all(
    interval("no_direction_loss", "causalcelljepa", regime, "effect_pearson")[0] > 0
    for regime in ("context_ood", "double_ood")
)

result = {
    "format_version": 1,
    "headline_means": headline_means,
    "retrieval": retrieval,
    "primary_paired_comparisons": primary,
    "mechanism_findings": {
        "global_context": {
            "finding": "Removing pooled context improves transferred effect direction but worsens transferred distribution and magnitude calibration.",
            "uniform_benefit_supported": False,
        },
        "context_structure": {
            "finding": "Mean context slightly improves K562 metrics, while set-structured context is more robust after transfer to RPE1.",
            "higher_order_population_structure_matters_for_transfer": True,
        },
        "direction_loss": {
            "finding": "Direction supervision improves perturbation-OOD direction but is associated with worse cross-context direction; it is useful but context-specialized.",
            "uniform_benefit_supported": False,
        },
        "overall": {
            "finding": "The full model is best calibrated after context transfer, whereas no-global-context has the strongest transferred effect direction; the architecture exposes a direction-calibration tradeoff rather than a dominant mechanism.",
            "single_dominant_ablation": False,
        },
    },
    "scope": {
        "latent_only": True,
        "selection_used_test_outcomes": False,
        "statistical_unit": "perturbation-condition",
        "transcriptomic_superiority_claim": False,
    },
    "provenance": {
        "summary_sha256": file_sha256(summary_path),
        "paired_comparisons_sha256": file_sha256(comparisons_path),
        "evaluation_provenance_sha256": file_sha256(provenance_path),
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert result["provenance"]["git"]["dirty"] is False
(root / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(
    {
        "primary_pairs": len(primary),
        "regimes": len(regimes),
        "metrics": len(metrics),
        "single_dominant_ablation": False,
    }
)
