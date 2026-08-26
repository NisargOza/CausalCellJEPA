# Summarize decoded Stage 2 seeds without rerunning inference or selecting test outcomes.
import json
from pathlib import Path

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

root = Path("artifacts/evaluation_stage2_replication_transcriptomics")
summary_path, comparisons_path = root / "summary.json", root / "paired_comparisons.json"
provenance_path = root / "provenance.json"
summary = json.loads(summary_path.read_text())
paired = json.loads(comparisons_path.read_text())
provenance = json.loads(provenance_path.read_text())
assert provenance["git"]["dirty"] is False
assert provenance["executed_repeats"] == 8
assert provenance["maximum_conditions_per_regime"] is None

models = ["causalcelljepa", "seed_20260824", "seed_20260825"]
model_seeds = dict(zip(models, (20260823, 20260824, 20260825)))
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
condition_metrics = [
    "all_effect_pearson",
    "target_excluded_effect_pearson",
    "all_magnitude_absolute_error",
    "deg_auprc",
    "retrospective_top50_effect_pearson",
]
pathway_metrics = ["pathway_nes_pearson", "pathway_nes_rmse"]
condition_means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["condition_metrics"]
}
pathway_means = {
    (item["regime"], item["model"], item["metric"]): item["mean"]
    for item in summary["pathway_metrics"]
}
seed_means = {
    regime: {
        **{
            metric: {model: condition_means[regime, model, metric] for model in models}
            for metric in condition_metrics
        },
        **{
            metric: {model: pathway_means[regime, model, metric] for model in models}
            for metric in pathway_metrics
        },
    }
    for regime in regimes
}

condition_index = {
    (item["candidate"], item["reference"], item["regime"], item["metric"]): item
    for item in paired["condition_comparisons"]
}
pathway_index = {
    (item["candidate"], item["reference"], item["regime"], item["metric"]): item
    for item in paired["pathway_comparisons"]
}
comparison_summary = {}
for regime in regimes:
    comparison_summary[regime] = {}
    for model in models[1:]:
        comparison_summary[regime][model] = {}
        for reference in ("causalcelljepa", "linear_esm", "pseudo_paired"):
            comparison_summary[regime][model][reference] = {}
            for metric in condition_metrics:
                item = condition_index[model, reference, regime, metric]
                comparison_summary[regime][model][reference][metric] = {
                    key: item[key]
                    for key in (
                        "mean_improvement",
                        "mean_improvement_bootstrap_95ci",
                        "benjamini_hochberg_q",
                    )
                }
            for metric in pathway_metrics:
                item = pathway_index[model, reference, regime, metric]
                comparison_summary[regime][model][reference][metric] = {
                    key: item[key]
                    for key in (
                        "mean_improvement",
                        "mean_improvement_bootstrap_95ci",
                        "benjamini_hochberg_q",
                    )
                }

for regime in regimes:
    for model in models[1:]:
        assert (
            comparison_summary[regime][model]["causalcelljepa"]["all_effect_pearson"][
                "mean_improvement_bootstrap_95ci"
            ][0]
            > 0
        )
        assert (
            comparison_summary[regime][model]["linear_esm"]["all_effect_pearson"][
                "mean_improvement_bootstrap_95ci"
            ][1]
            < 0
        )
        assert (
            comparison_summary[regime][model]["linear_esm"]["all_magnitude_absolute_error"][
                "mean_improvement_bootstrap_95ci"
            ][0]
            > 0
        )
        assert (
            abs(
                seed_means[regime]["all_effect_pearson"][model]
                - seed_means[regime]["target_excluded_effect_pearson"][model]
            )
            < 0.0025
        )
for regime in ("iid", "perturbation_ood"):
    for model in models[1:]:
        assert (
            comparison_summary[regime][model]["pseudo_paired"]["all_effect_pearson"][
                "mean_improvement_bootstrap_95ci"
            ][1]
            < 0
        )
for regime in ("context_ood", "double_ood"):
    assert all(seed_means[regime]["all_effect_pearson"][model] < 0 for model in models)
    assert condition_means[regime, "linear_esm", "all_effect_pearson"] > 0
    assert all(seed_means[regime]["pathway_nes_pearson"][model] < 0 for model in models)
    assert pathway_means[regime, "linear_esm", "pathway_nes_pearson"] > 0

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
    "headline_paired_comparisons": comparison_summary,
    "retrieval_by_seed": retrieval,
    "findings": {
        "seed_improvement": "Both new seeds improve decoded effect correlation over the primary seed in every regime.",
        "linear_comparison": "Both new seeds remain below linear ESM on decoded effect correlation in every regime, including after target-gene exclusion.",
        "context_transfer": "All three JEPA seeds have negative RPE1 gene-effect and pathway correlations, while linear ESM is positive.",
        "magnitude": "Both new seeds improve magnitude absolute error over linear ESM in every regime.",
        "pseudo_pairing": "Both new seeds lose to pseudo-pairing on K562 effect correlation; seed 20260825 improves RPE1 effect direction over pseudo-pairing but remains negative in absolute terms.",
        "overall": "The latent-space replication gain does not survive as positive transcriptomic direction, so it cannot rescue the falsified gene-level superiority hypothesis.",
    },
    "scope": {
        "decoder_frozen": True,
        "population_repeats": 8,
        "selection_used_test_outcomes": False,
        "statistical_unit": "perturbation-condition",
        "target_gene_excluded": True,
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
print({"seeds": len(models), "regimes": len(regimes), "metrics": len(condition_metrics)})
