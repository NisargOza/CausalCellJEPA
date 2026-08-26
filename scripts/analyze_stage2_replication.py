# Summarize all Stage 2 seeds without rerunning inference or selecting on test outcomes.
import json
from pathlib import Path

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

root = Path("artifacts/evaluation_stage2_replication")
summary_path, comparisons_path = root / "summary.json", root / "paired_comparisons.json"
provenance_path = root / "provenance.json"
summary = json.loads(summary_path.read_text())
comparisons = json.loads(comparisons_path.read_text())
provenance = json.loads(provenance_path.read_text())
assert provenance["git"]["dirty"] is False
assert provenance["executed_repeats"] == 8
assert provenance["maximum_conditions_per_regime"] is None

models = ["causalcelljepa", "seed_20260824", "seed_20260825"]
model_seeds = dict(zip(models, (20260823, 20260824, 20260825)))
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
metrics = [
    "effect_pearson",
    "direction_cosine",
    "magnitude_absolute_error",
    "sinkhorn",
    "mmd",
    "covariance_shift_error",
]
references = ["no_change", "linear_esm", "pseudo_paired"]
means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["condition_metrics"]
}
seed_means = {}
for regime in regimes:
    seed_means[regime] = {}
    for metric in metrics:
        values = {model: means[regime, model, metric] for model in models}
        seed_means[regime][metric] = {
            "by_model": values,
            "mean_across_seeds": sum(values.values()) / len(values),
            "range_across_seeds": [min(values.values()), max(values.values())],
        }

comparison_index = {
    (item["candidate"], item["reference"], item["regime"], item["metric"]): item
    for item in comparisons
}
robustness = {}
for regime in regimes:
    robustness[regime] = {}
    for reference in references:
        robustness[regime][reference] = {}
        for metric in metrics:
            by_model = {}
            for model in models:
                item = comparison_index.get((model, reference, regime, metric))
                if item is not None:
                    by_model[model] = {
                        key: item[key]
                        for key in (
                            "mean_improvement",
                            "mean_improvement_bootstrap_95ci",
                            "benjamini_hochberg_q",
                        )
                    }
            robustness[regime][reference][metric] = {
                "by_model": by_model,
                "favorable_seed_count": sum(
                    item["mean_improvement"] > 0 for item in by_model.values()
                ),
                "bootstrap_ci_positive_seed_count": sum(
                    item["mean_improvement_bootstrap_95ci"][0] > 0 for item in by_model.values()
                ),
                "bootstrap_ci_negative_seed_count": sum(
                    item["mean_improvement_bootstrap_95ci"][1] < 0 for item in by_model.values()
                ),
                "seeds_compared": len(by_model),
            }

for regime in ("iid", "perturbation_ood"):
    for metric in ("effect_pearson", "direction_cosine", "magnitude_absolute_error", "sinkhorn"):
        assert robustness[regime]["linear_esm"][metric]["bootstrap_ci_positive_seed_count"] == 3
for regime in ("context_ood", "double_ood"):
    direction = robustness[regime]["linear_esm"]["effect_pearson"]
    assert direction["favorable_seed_count"] == direction["bootstrap_ci_positive_seed_count"] == 2
    assert direction["bootstrap_ci_negative_seed_count"] == 1
    for metric in ("magnitude_absolute_error", "covariance_shift_error"):
        assert robustness[regime]["linear_esm"][metric]["bootstrap_ci_positive_seed_count"] == 3
    for metric in ("sinkhorn", "mmd"):
        assert robustness[regime]["linear_esm"][metric]["bootstrap_ci_negative_seed_count"] == 3
    for metric in ("magnitude_absolute_error", "sinkhorn", "mmd", "covariance_shift_error"):
        assert robustness[regime]["pseudo_paired"][metric]["bootstrap_ci_positive_seed_count"] == 3

retrieval = {
    regime: {
        item["model"]: {
            key: item[key] for key in ("top_1", "top_5", "mean_reciprocal_rank", "median_rank")
        }
        for item in summary["retrieval"]
        if item["regime"] == regime and item["model"] in models
    }
    for regime in regimes
}
result = {
    "format_version": 1,
    "model_seeds": model_seeds,
    "headline_seed_means": seed_means,
    "headline_paired_robustness": robustness,
    "retrieval_by_seed": retrieval,
    "findings": {
        "training": "Both additional seeds converged cleanly to similar K562 validation losses.",
        "k562": "All three seeds beat the linear ESM baseline on effect direction, magnitude error, Sinkhorn distance, and covariance error; MMD did not replicate uniformly.",
        "rpe1_direction": "Both additional seeds beat linear ESM and pseudo-paired models on transferred effect direction, while the primary seed lost to both, exposing material seed sensitivity.",
        "rpe1_calibration": "All three seeds beat linear ESM on magnitude and covariance errors and beat pseudo-paired training on all reported distribution and calibration metrics.",
        "rpe1_distribution": "All three seeds lost to no-change and linear ESM baselines on transferred Sinkhorn and MMD distances.",
        "overall": "Replication strengthens the effect-direction result and the advantage over pseudo-pairing, but does not support universal distributional superiority after context transfer.",
    },
    "scope": {
        "latent_only": True,
        "population_repeats": 8,
        "selection_used_test_outcomes": False,
        "statistical_unit": "perturbation-condition",
        "target_gene_excluded_claim_added": False,
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
print({"seeds": len(models), "regimes": len(regimes), "headline_metrics": len(metrics)})
