# Compare the validation-frozen anchored model with every previously frozen gene-level baseline.
# This reporting-only analysis averages repeats within perturbations and never selects a model.
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from causalcelljepa.readout import _paired_transcriptomic_models
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

seed, resamples = 20260823, 10_000
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
condition_metrics = [
    "all_effect_pearson",
    "target_excluded_effect_pearson",
    "all_magnitude_absolute_error",
    "deg_auprc",
    "retrospective_top50_effect_pearson",
]
pathway_metrics = ["pathway_nes_pearson", "pathway_nes_rmse"]
source_specs = {
    "base": (
        Path("manifests/transcriptomics_v1.json"),
        Path("artifacts/transcriptomic_evaluation"),
        ["causalcelljepa", "no_change", "mean_effect", "linear_esm", "pseudo_paired"],
    ),
    "direct_gene": (
        Path("manifests/direct_gene_v1.json"),
        Path("artifacts/direct_gene_baseline"),
        ["direct_gene_esm"],
    ),
    "replication": (
        Path("manifests/evaluation_stage2_replication_transcriptomics_v1.json"),
        Path("artifacts/evaluation_stage2_replication_transcriptomics"),
        ["seed_20260824", "seed_20260825"],
    ),
    "comparators": (
        Path("manifests/evaluation_remaining_comparators_v1.json"),
        Path("artifacts/evaluation_remaining_comparators"),
        ["autoencoder_state", "learned_target_id", "pca_state"],
    ),
}
artifact_names = {
    "condition_metrics": "condition_metrics.jsonl",
    "pathway_metrics": "pathway_metrics.jsonl",
    "summary": "summary.json",
}
verified_sources = {}
for name, (manifest_path, root, models) in source_specs.items():
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.pop("manifest_sha256")
    assert (
        declared
        == sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    artifacts = {}
    for kind, filename in artifact_names.items():
        entry, path = manifest["artifacts"][kind], root / filename
        assert (path.stat().st_size, file_sha256(path)) == (entry["bytes"], entry["sha256"])
        artifacts[kind] = {"path": str(path), "bytes": entry["bytes"], "sha256": entry["sha256"]}
    verified_sources[name] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": declared,
        "models": models,
        "artifacts": artifacts,
    }

anchored_root = Path("artifacts/evaluation_anchored_transcriptomics")
anchored_provenance_path = anchored_root / "provenance.json"
anchored_provenance = json.loads(anchored_provenance_path.read_text())
assert anchored_provenance["git"]["dirty"] is False
assert anchored_provenance["executed_repeats"] == 8
assert anchored_provenance["maximum_conditions_per_regime"] is None
assert list(anchored_provenance["executed_regimes"]) == sorted(regimes)
for kind in ("training", "selection"):
    manifest_path = Path(f"manifests/anchored_dynamics_{kind}_v1.json")
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.pop("manifest_sha256")
    assert (
        declared
        == anchored_provenance[f"anchored_{kind}_manifest_sha256"]
        == sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
anchored_artifacts = {}
for kind, filename in artifact_names.items():
    path = anchored_root / filename
    anchored_artifacts[kind] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
anchored_artifacts["provenance"] = {
    "path": str(anchored_provenance_path),
    "bytes": anchored_provenance_path.stat().st_size,
    "sha256": file_sha256(anchored_provenance_path),
}
verified_sources["anchored"] = {
    "evaluation_git": anchored_provenance["git"],
    "models": ["anchored_selected"],
    "artifacts": anchored_artifacts,
}

record_fields = {"regime", "model", "target", "repeat"}
condition_records, pathway_records = [], []
raw_sources = [
    (anchored_root, ["anchored_selected"]),
    *[(root, models) for _, root, models in source_specs.values()],
]
for root, models in raw_sources:
    for filename, metrics, destination in (
        ("condition_metrics.jsonl", condition_metrics, condition_records),
        ("pathway_metrics.jsonl", pathway_metrics, pathway_records),
    ):
        with (root / filename).open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["model"] in models:
                    destination.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key in record_fields | set(metrics)
                        }
                    )

reference_models = [model for _, _, models in source_specs.values() for model in models]
pairs = [
    {
        "candidate": "anchored_selected",
        "reference": reference,
        "hypothesis": "validation-frozen anchored model differs from a previously frozen baseline",
    }
    for reference in reference_models
]
condition_comparisons = _paired_transcriptomic_models(condition_records, pairs, resamples, seed)
pathway_comparisons = _paired_transcriptomic_models(pathway_records, pairs, resamples, seed)
assert len(condition_comparisons) == len(regimes) * len(reference_models) * len(condition_metrics)
assert len(pathway_comparisons) == len(regimes) * len(reference_models) * len(pathway_metrics)

models = ["anchored_selected", *reference_models]
means = {}
summary_sources = [(anchored_root / "summary.json", ["anchored_selected"])] + [
    (root / "summary.json", source_models) for _, root, source_models in source_specs.values()
]
for path, source_models in summary_sources:
    summary = json.loads(path.read_text())
    for section in ("condition_metrics", "pathway_metrics"):
        for item in summary[section]:
            if (
                item["model"] in source_models
                and item["metric"] in condition_metrics + pathway_metrics
            ):
                means[item["regime"], item["model"], item["metric"]] = item["mean"]
assert len(means) == len(regimes) * len(models) * (len(condition_metrics) + len(pathway_metrics))
headline_means, rankings = {}, {}
for regime in regimes:
    headline_means[regime], rankings[regime] = {}, {}
    for metric in condition_metrics + pathway_metrics:
        values = {model: means[regime, model, metric] for model in models}
        lower_is_better = metric.endswith(("absolute_error", "rmse"))
        ordered = sorted(values, key=values.get, reverse=not lower_is_better)
        anchor = values["anchored_selected"]
        rankings[regime][metric] = {
            "anchored_rank": 1
            + sum(
                value < anchor if lower_is_better else value > anchor for value in values.values()
            ),
            "models": len(models),
            "ordered_models": ordered,
            "anchored_ties": [
                model
                for model, value in values.items()
                if model != "anchored_selected" and np.isclose(value, anchor, rtol=0, atol=1e-12)
            ],
        }
        headline_means[regime][metric] = values

all_comparisons = condition_comparisons + pathway_comparisons
verdict_counts = {"win": 0, "inconclusive": 0, "loss": 0}
for item in all_comparisons:
    low, high = item["mean_improvement_bootstrap_95ci"]
    verdict = "win" if low > 0 else "loss" if high < 0 else "inconclusive"
    item["bootstrap_95ci_verdict"] = verdict
    verdict_counts[verdict] += 1
linear_items = [item for item in all_comparisons if item["reference"] == "linear_esm"]
assert all(item["mean_improvement"] == 0 for item in linear_items)
result = {
    "format_version": 1,
    "scope": {
        "reporting_only": True,
        "selection_used_sealed_outcomes": False,
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        "statistical_unit": "perturbation-condition",
        "repeat_handling": "mean within target before paired comparison",
        "bootstrap_resamples": resamples,
        "target_gene_excluded": True,
    },
    "headline_means": headline_means,
    "rankings": rankings,
    "paired_comparisons": {
        "condition": condition_comparisons,
        "pathway": pathway_comparisons,
        "bootstrap_95ci_verdict_counts": verdict_counts,
    },
    "findings": {
        "uniform_superiority_supported": verdict_counts["loss"] == 0,
        "learned_residual_improves_gene_centroid_metrics_over_linear_anchor": False,
        "linear_anchor_exact_tied_comparisons": len(linear_items),
    },
    "provenance": {
        "sources": verified_sources,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert result["findings"]["uniform_superiority_supported"] is False
assert result["provenance"]["git"]["dirty"] is False
output = Path("artifacts/evaluation_anchored_analysis")
assert not output.exists()
output.mkdir(parents=True)
(output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print({"models": len(models), "paired_comparisons": len(all_comparisons), **verdict_counts})
