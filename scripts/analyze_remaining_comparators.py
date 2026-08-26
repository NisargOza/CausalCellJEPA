# Freeze a conservative interpretation of the action/state comparator evaluation.
# Assertions encode only perturbation-level findings supported by paired bootstrap intervals.
import json
from pathlib import Path

import numpy as np
import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config = yaml.safe_load(Path("configs/evaluation_remaining_comparators.yaml").read_text())
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
root = Path(config["output_directory"])
summary_path, paired_path = root / "summary.json", root / "paired_comparisons.json"
provenance_path = root / "provenance.json"
summary = json.loads(summary_path.read_text())
paired = json.loads(paired_path.read_text())
provenance = json.loads(provenance_path.read_text())
assert provenance["git"]["dirty"] is False
assert provenance["maximum_conditions_per_regime"] is None
assert provenance["executed_repeats"] == base["sampling"]["repeats"] == 8
assert provenance["executed_regimes"] == base["regimes"]

models = ["causalcelljepa", "learned_target_id", "pca_state", "autoencoder_state"]
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
condition_metrics = [
    "all_effect_pearson",
    "target_excluded_effect_pearson",
    "all_magnitude_absolute_error",
    "deg_auprc",
]
condition_means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["condition_metrics"]
}
pathway_means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["pathway_metrics"]
}
headline = {
    regime: {
        model: {
            **{metric: condition_means[regime, model, metric] for metric in condition_metrics},
            "pathway_nes_pearson": pathway_means[regime, model, "pathway_nes_pearson"],
        }
        for model in models
    }
    for regime in regimes
}
retrieval = {
    regime: {
        item["model"]: {
            key: item[key]
            for key in ("top_1", "top_5", "mean_reciprocal_rank", "median_rank")
        }
        for item in summary["retrieval"]
        if item["regime"] == regime and item["model"] in models
    }
    for regime in regimes
}
condition_comparisons = {
    (item["reference"], item["regime"], item["metric"]): item
    for item in paired["condition_comparisons"]
}
pathway_comparisons = {
    (item["reference"], item["regime"], item["metric"]): item
    for item in paired["pathway_comparisons"]
}


def interval(reference, regime, metric, comparisons=condition_comparisons):
    return comparisons[reference, regime, metric]["mean_improvement_bootstrap_95ci"]


assert all(
    interval("learned_target_id", regime, "all_effect_pearson")[1] < 0
    for regime in ("iid", "perturbation_ood", "context_ood")
)
assert interval("learned_target_id", "double_ood", "all_effect_pearson")[0] < 0 < interval(
    "learned_target_id", "double_ood", "all_effect_pearson"
)[1]
assert all(
    interval("pca_state", regime, "all_effect_pearson")[1] < 0
    for regime in ("iid", "perturbation_ood")
)
assert all(
    interval("pca_state", regime, "all_effect_pearson")[0] > 0
    and interval("pca_state", regime, "all_magnitude_absolute_error")[0] > 0
    for regime in ("context_ood", "double_ood")
)
assert all(
    interval("autoencoder_state", regime, "all_effect_pearson")[1] < 0 for regime in regimes
)
assert all(
    interval("autoencoder_state", regime, "all_magnitude_absolute_error")[0] > 0
    for regime in ("context_ood", "double_ood")
)
assert all(
    np.sign(condition_comparisons[reference, regime, "all_effect_pearson"]["mean_improvement"])
    == np.sign(
        condition_comparisons[reference, regime, "target_excluded_effect_pearson"][
            "mean_improvement"
        ]
    )
    for reference in ("learned_target_id", "pca_state", "autoencoder_state")
    for regime in regimes
)

result = {
    "format_version": 1,
    "headline_means": headline,
    "retrieval": retrieval,
    "primary_paired_comparisons": {
        f"causalcelljepa_vs_{reference}": {
            regime: {
                metric: {
                    key: condition_comparisons[reference, regime, metric][key]
                    for key in (
                        "candidate_mean",
                        "reference_mean",
                        "mean_improvement",
                        "mean_improvement_bootstrap_95ci",
                        "benjamini_hochberg_q",
                    )
                }
                for metric in condition_metrics
            }
            for regime in regimes
        }
        for reference in ("learned_target_id", "pca_state", "autoencoder_state")
    },
    "findings": {
        "structured_action": {
            "supported": False,
            "finding": "ESM-2 actions do not improve decoded effect direction over learned target IDs in IID, perturbation-OOD, or context-OOD; the small double-OOD direction advantage is inconclusive and magnitude calibration is worse.",
        },
        "jepa_vs_pca_state": {
            "uniform_jepa_benefit": False,
            "finding": "PCA is stronger on K562 effect direction and magnitude, while JEPA is better calibrated and less anticorrelated after transfer to RPE1; neither state dominates across regimes and metrics.",
        },
        "jepa_vs_autoencoder_state": {
            "uniform_jepa_benefit": False,
            "finding": "The autoencoder has stronger decoded effect direction in every regime, while JEPA avoids the autoencoder's severe cross-context magnitude inflation; predictive pretraining is not superior overall.",
        },
        "target_gene_exclusion": {
            "changes_conclusion": False,
            "finding": "Removing each perturbed target gene preserves every headline paired direction.",
        },
        "overall": {
            "core_representation_hypotheses_supported": False,
            "finding": "The new comparisons expose direction-calibration tradeoffs and do not establish a uniform advantage for the JEPA state or structured action representation.",
        },
    },
    "scope": {
        "common_endpoint": "normalized log expression over 3000 frozen HVGs",
        "readouts": {
            "causalcelljepa_and_learned_target_id": "frozen leakage-safe linear JEPA readout",
            "pca_state": "frozen exact inverse PCA projection",
            "autoencoder_state": "frozen reconstruction decoder",
        },
        "statistical_unit": "perturbation-condition",
        "bootstrap_resamples": base["metrics"]["bootstrap_resamples"],
        "test_outcomes_used_for_fit_or_selection": False,
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
    },
    "pathway_paired_comparisons": {
        f"causalcelljepa_vs_{reference}": {
            regime: {
                key: pathway_comparisons[reference, regime, "pathway_nes_pearson"][key]
                for key in (
                    "candidate_mean",
                    "reference_mean",
                    "mean_improvement",
                    "mean_improvement_bootstrap_95ci",
                    "benjamini_hochberg_q",
                )
            }
            for regime in regimes
        }
        for reference in ("learned_target_id", "pca_state", "autoencoder_state")
    },
    "provenance": {
        "evaluation_artifact_sha256": {
            path.name: file_sha256(path)
            for path in (
                root / "condition_metrics.jsonl",
                root / "pathway_metrics.jsonl",
                summary_path,
                paired_path,
                root / "truth_report.json",
                provenance_path,
            )
        },
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert result["provenance"]["git"]["dirty"] is False
(root / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print({"models": len(models), "regimes": len(regimes), "hypotheses_supported": False})
