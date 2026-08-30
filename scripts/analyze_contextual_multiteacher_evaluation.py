# Freeze the v4 report against the complete previously frozen 13-model comparison.
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

seed, resamples, numerical_tie_tolerance = 20260823, 10_000, 2e-6
candidate = "contextual_multiteacher_selected"
reference = "multiteacher_selected"
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
condition_metrics = [
    "all_effect_pearson",
    "target_excluded_effect_pearson",
    "all_magnitude_absolute_error",
    "deg_auprc",
    "retrospective_top50_effect_pearson",
]
pathway_metrics = ["pathway_nes_pearson", "pathway_nes_rmse"]
latent_metrics = [
    "effect_pearson",
    "magnitude_absolute_error",
    "sinkhorn",
    "mmd",
    "energy_distance",
]
evaluation_roots = {
    "latent_evaluation": Path("artifacts/evaluation_contextual_multiteacher"),
    "transcriptomic_evaluation": Path(
        "artifacts/evaluation_contextual_multiteacher_transcriptomics"
    ),
}


def self_hashed_manifest(path):
    payload = json.loads(path.read_text())
    declared = payload.pop("manifest_sha256")
    assert declared == sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload, declared


def artifact(path, records=None):
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if records is not None:
        result["records"] = records
    return result


def bootstrap_verdict(item):
    low, high = item["mean_improvement_bootstrap_95ci"]
    return "win" if low > 0 else "loss" if high < 0 else "inconclusive"


assert _git_state()["dirty"] is False
base_manifest_path = Path("manifests/evaluation_multiteacher_v1.json")
base_manifest, base_manifest_sha256 = self_hashed_manifest(base_manifest_path)
base_analysis_entry = base_manifest["artifacts"]["cross_baseline_analysis"]
base_analysis_path = Path(base_analysis_entry["path"])
assert (base_analysis_path.stat().st_size, file_sha256(base_analysis_path)) == (
    base_analysis_entry["bytes"],
    base_analysis_entry["sha256"],
)
base_analysis = json.loads(base_analysis_path.read_text())
assert base_analysis["findings"]["models_compared"] == 13
assert base_analysis["scope"]["bootstrap_resamples"] == resamples
assert base_analysis["scope"]["statistical_unit"] == "perturbation-condition"

latent_provenance = json.loads((evaluation_roots["latent_evaluation"] / "provenance.json").read_text())
transcriptomic_provenance = json.loads(
    (evaluation_roots["transcriptomic_evaluation"] / "provenance.json").read_text()
)
for provenance in (latent_provenance, transcriptomic_provenance):
    assert provenance["git"]["dirty"] is False
    assert provenance["executed_repeats"] == 8
    assert provenance["maximum_conditions_per_regime"] is None
    assert list(provenance["executed_regimes"]) == sorted(regimes)
    assert provenance["contextual_multiteacher_training_manifest_sha256"] == (
        "1370ca20cd7e743920c981a5f6ec70e4205eac568a22a5e84f80f7e20dd85e86"
    )
    assert provenance["contextual_multiteacher_selection_manifest_sha256"] == (
        "ee8a3e8aa1f128ae700fbb3548a5a47d3d74e17a50656638705eb5f0f86a5199"
    )
assert latent_provenance["git"] == transcriptomic_provenance["git"]

candidate_summary = json.loads(
    (evaluation_roots["transcriptomic_evaluation"] / "summary.json").read_text()
)
headline_means = deepcopy(base_analysis["headline_means"])
for section, metrics in (("condition_metrics", condition_metrics), ("pathway_metrics", pathway_metrics)):
    for item in candidate_summary[section]:
        if item["model"] == candidate and item["metric"] in metrics:
            headline_means[item["regime"]][item["metric"]][candidate] = item["mean"]
assert all(
    candidate in headline_means[regime][metric]
    for regime in regimes
    for metric in condition_metrics + pathway_metrics
)

old_models = [reference, *base_analysis["findings"]["reference_models"]]
assert len(old_models) == 13 and len(set(old_models)) == 13
models = [candidate, *old_models]
rankings = {}
for regime in regimes:
    rankings[regime] = {}
    for metric in condition_metrics + pathway_metrics:
        values = headline_means[regime][metric]
        assert set(values) == set(models)
        comparable = {model: value for model, value in values.items() if value is not None}
        lower_is_better = metric.endswith(("absolute_error", "rmse"))
        candidate_value = comparable[candidate]
        ordered = sorted(comparable, key=comparable.get, reverse=not lower_is_better)
        rankings[regime][metric] = {
            "contextual_multiteacher_rank": 1
            + sum(
                value < candidate_value - numerical_tie_tolerance
                if lower_is_better
                else value > candidate_value + numerical_tie_tolerance
                for value in comparable.values()
            ),
            "models_evaluated": len(comparable),
            "models_undefined": sorted(set(models) - set(comparable)),
            "ordered_models": ordered,
            "contextual_multiteacher_ties": [
                model
                for model, value in comparable.items()
                if model != candidate
                and np.isclose(value, candidate_value, rtol=0, atol=numerical_tie_tolerance)
            ],
        }

latent_pairs = json.loads(
    (evaluation_roots["latent_evaluation"] / "paired_comparisons.json").read_text()
)
decoded_pairs = json.loads(
    (evaluation_roots["transcriptomic_evaluation"] / "paired_comparisons.json").read_text()
)
latent_vs_v3 = [
    item for item in latent_pairs if item["reference"] == reference and item["metric"] in latent_metrics
]
decoded_vs_v3 = [
    item
    for section in ("condition_comparisons", "pathway_comparisons")
    for item in decoded_pairs[section]
    if item["reference"] == reference and item["metric"] in condition_metrics + pathway_metrics
]
assert len(latent_vs_v3) == len(regimes) * len(latent_metrics)
assert len(decoded_vs_v3) == len(regimes) * (len(condition_metrics) + len(pathway_metrics))
all_vs_v3 = [item for item in latent_pairs if item["reference"] == reference] + [
    item
    for section in ("condition_comparisons", "pathway_comparisons")
    for item in decoded_pairs[section]
    if item["reference"] == reference
]
verdict_counts = {name: 0 for name in ("win", "inconclusive", "loss")}
headline_verdict_counts = {name: 0 for name in verdict_counts}
for item in all_vs_v3:
    verdict_counts[bootstrap_verdict(item)] += 1
for item in latent_vs_v3 + decoded_vs_v3:
    headline_verdict_counts[bootstrap_verdict(item)] += 1

analysis = {
    "format_version": 1,
    "scope": {
        "reporting_only": True,
        "selection_used_sealed_outcomes": False,
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        "statistical_unit": "perturbation-condition",
        "repeat_handling": "mean within target before paired comparison",
        "bootstrap_resamples": resamples,
        "target_gene_excluded": True,
        "numerical_tie_tolerance": numerical_tie_tolerance,
    },
    "headline_means": headline_means,
    "rankings": rankings,
    "paired_comparisons": {
        "latent_vs_v3": latent_vs_v3,
        "decoded_vs_v3": decoded_vs_v3,
        "all_v3_bootstrap_95ci_verdict_counts": verdict_counts,
        "headline_v3_bootstrap_95ci_verdict_counts": headline_verdict_counts,
    },
    "findings": {
        "uniform_superiority_supported": verdict_counts["loss"] == 0,
        "models_compared": len(models),
        "reference_models": old_models,
        "selected_candidate": "availability_static",
        "context_conditioned_candidate_selected": False,
    },
    "provenance": {
        "base_analysis_manifest_path": str(base_manifest_path),
        "base_analysis_manifest_sha256": base_manifest_sha256,
        "base_analysis": artifact(base_analysis_path),
        "candidate_evaluation_git": latent_provenance["git"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert analysis["findings"]["uniform_superiority_supported"] is False
output = Path("artifacts/evaluation_contextual_multiteacher_analysis")
assert not output.exists()
output.mkdir(parents=True)
analysis_path = output / "analysis.json"
analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

evaluation_artifacts = {}
for name, root in evaluation_roots.items():
    filenames = ["condition_metrics.jsonl", "paired_comparisons.json", "provenance.json", "summary.json"]
    if name == "transcriptomic_evaluation":
        filenames.extend(["pathway_metrics.jsonl", "truth_report.json"])
    evaluation_artifacts[name] = {
        filename: artifact(
            root / filename,
            len((root / filename).read_text().splitlines()) if filename.endswith(".jsonl") else None,
        )
        for filename in filenames
    }
manifest = {
    "format_version": 1,
    "artifacts": {
        **evaluation_artifacts,
        "cross_baseline_analysis": artifact(analysis_path),
    },
    "protocol": {
        "candidate": candidate,
        "evaluation_repeats": 8,
        "reporting_only": True,
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        "sealed_test_outcomes_used_for_fit_or_selection": False,
        "statistical_unit": "perturbation-condition",
    },
    "result": {
        "models_compared": len(models),
        "uniform_superiority_supported": False,
        "all_v3_bootstrap_95ci_verdict_counts": verdict_counts,
        "headline_v3_bootstrap_95ci_verdict_counts": headline_verdict_counts,
    },
    "provenance": {
        "evaluation_git": latent_provenance["git"],
        "analysis_git": analysis["provenance"]["git"],
        "evaluation_config_sha256": file_sha256(
            "configs/evaluation_contextual_multiteacher.yaml"
        ),
        "contextual_multiteacher_training_manifest_sha256": latent_provenance[
            "contextual_multiteacher_training_manifest_sha256"
        ],
        "contextual_multiteacher_selection_manifest_sha256": latent_provenance[
            "contextual_multiteacher_selection_manifest_sha256"
        ],
    },
}
manifest["manifest_sha256"] = sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path("manifests/evaluation_contextual_multiteacher_v1.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(
    {
        "models": len(models),
        "comparisons_vs_v3": len(all_vs_v3),
        "all": verdict_counts,
        "headline": headline_verdict_counts,
    }
)
