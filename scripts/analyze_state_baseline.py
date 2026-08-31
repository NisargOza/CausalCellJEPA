# Freeze the bounded State run against the complete validation-frozen comparison.
# This reporting step cannot select or modify a checkpoint.
import csv
import json
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import torch

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

seed, model = 20260823, "state_sm"
regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
condition_metrics = [
    "all_effect_pearson",
    "target_excluded_effect_pearson",
    "all_magnitude_absolute_error",
    "deg_auprc",
    "retrospective_top50_effect_pearson",
]
pathway_metrics = ["pathway_nes_pearson", "pathway_nes_rmse"]
training = Path("artifacts/state_baseline/training/state_sm_seed_20260823")
prediction = Path("artifacts/state_baseline/predictions")
evaluation = Path("artifacts/state_baseline/evaluation")


def artifact(path, records=None):
    path = Path(path)
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if records is not None:
        result["records"] = records
    return result


def self_hashed(path):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    assert declared == sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload, declared


assert _git_state()["dirty"] is False
base_manifest_path = Path("manifests/evaluation_contextual_multiteacher_v1.json")
base_manifest, base_manifest_sha256 = self_hashed(base_manifest_path)
base_analysis_entry = base_manifest["artifacts"]["cross_baseline_analysis"]
base_analysis_path = Path(base_analysis_entry["path"])
assert artifact(base_analysis_path) == base_analysis_entry
base_analysis = json.loads(base_analysis_path.read_text())
assert base_analysis["findings"]["models_compared"] == 14

report = json.loads((prediction / "state_sm_seed_20260823_report.json").read_text())
assert artifact(report["path"], report["records"]) == {
    key: report[key] for key in ("path", "bytes", "sha256", "records")
}
assert report["records"] == 14_352 and report["repeats"] == 8
assert report["test_outcome_expression_read"] is False
provenance = json.loads((evaluation / "provenance.json").read_text())
assert provenance["git"]["dirty"] is False and provenance["reporting_only"] is True
assert provenance["checkpoint_sha256"] == (
    "5dd727afd408e5039f6ef8b0a4141ebb66299409c97cf8804ecfab340682d262"
)

summary = json.loads((evaluation / "summary.json").read_text())
headline_means = deepcopy(base_analysis["headline_means"])
for section, metrics in (
    ("condition_metrics", condition_metrics),
    ("pathway_metrics", pathway_metrics),
):
    for item in summary[section]:
        if item["metric"] in metrics:
            headline_means[item["regime"]][item["metric"]][model] = item["mean"]
rankings = {}
for regime in regimes:
    rankings[regime] = {}
    for metric in condition_metrics + pathway_metrics:
        values = headline_means[regime][metric]
        comparable = {name: value for name, value in values.items() if value is not None}
        lower = metric.endswith(("absolute_error", "rmse"))
        ordered = sorted(comparable, key=comparable.get, reverse=not lower)
        rankings[regime][metric] = {
            "state_rank": ordered.index(model) + 1,
            "models_evaluated": len(ordered),
            "ordered_models": ordered,
        }

paired = json.loads((evaluation / "paired_comparisons.json").read_text())
verdict_counts = {}
for section in ("condition_comparisons", "pathway_comparisons"):
    for item in paired[section]:
        low, high = item["mean_improvement_bootstrap_95ci"]
        verdict = "win" if low > 0 else "loss" if high < 0 else "inconclusive"
        key = item["reference"]
        verdict_counts.setdefault(key, {name: 0 for name in ("win", "inconclusive", "loss")})
        verdict_counts[key][verdict] += 1

metrics_rows = list(csv.DictReader((training / "version_0/metrics.csv").open()))
validation = [
    {"logged_step": int(row["step"]), "loss": float(row["val_loss"])}
    for row in metrics_rows
    if row["val_loss"]
]
best_validation = min(validation, key=lambda item: (item["loss"], item["logged_step"]))
checkpoint = torch.load(training / "checkpoints/best.ckpt", map_location="cpu", weights_only=False)
callback = next(iter(checkpoint["callbacks"].values()))
assert float(callback["best_model_score"]) == best_validation["loss"]
start = datetime.fromisoformat(Path(f"{training}.start").read_text().strip())
end = datetime.fromisoformat(Path(f"{training}.end").read_text().strip())
analysis = {
    "format_version": 1,
    "scope": {
        "reporting_only": True,
        "statistical_unit": "perturbation-condition",
        "evaluation_repeats": 8,
        "models_compared": 15,
        "checkpoint_selection": "minimum K562 perturbation_ood_validation loss",
    },
    "training": {
        "steps": 10_000,
        "duration_seconds": (end - start).total_seconds(),
        "validation": validation,
        "best_validation": best_validation,
        "checkpoint_global_step": checkpoint["global_step"],
    },
    "headline_means": headline_means,
    "rankings": rankings,
    "retrieval": summary["retrieval"],
    "paired_bootstrap_95ci_verdict_counts": verdict_counts,
    "findings": {
        "uniform_superiority_supported": False,
        "state_is_best_on_any_headline_metric": any(
            item["state_rank"] == 1 for values in rankings.values() for item in values.values()
        ),
        "k562_effect_correlations_near_zero": True,
        "transferred_magnitude_miscalibrated": True,
    },
    "provenance": {
        "base_analysis_manifest_sha256": base_manifest_sha256,
        "prediction_report": report,
        "evaluation_git": provenance["git"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert analysis["findings"] == {
    "uniform_superiority_supported": False,
    "state_is_best_on_any_headline_metric": False,
    "k562_effect_correlations_near_zero": True,
    "transferred_magnitude_miscalibrated": True,
}
output = Path("artifacts/state_baseline/analysis")
assert not output.exists()
output.mkdir(parents=True)
analysis_path = output / "analysis.json"
analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

evaluation_files = [
    "condition_metrics.jsonl",
    "pathway_metrics.jsonl",
    "paired_comparisons.json",
    "provenance.json",
    "summary.json",
    "truth_report.json",
]
manifest = {
    "format_version": 1,
    "source": {
        "state_commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
        "state_version": "0.11.3",
        "cell_load_commit": "9ba45e59f6f8117bb7a21371ad38d67175586d53",
        "cell_load_version": "0.10.4",
    },
    "artifacts": {
        "best_checkpoint": artifact(training / "checkpoints/best.ckpt"),
        "training_config": artifact(training / "config.yaml"),
        "training_metrics": artifact(training / "version_0/metrics.csv", len(metrics_rows) + 1),
        "training_log": artifact(f"{training}.log"),
        "prediction": artifact(report["path"], report["records"]),
        "prediction_report": artifact(prediction / "state_sm_seed_20260823_report.json"),
        "evaluation": {
            name: artifact(
                evaluation / name,
                len((evaluation / name).read_text().splitlines()) if name.endswith(".jsonl") else None,
            )
            for name in evaluation_files
        },
        "analysis": artifact(analysis_path),
    },
    "protocol": {
        "checkpoint_selected_on": "K562 perturbation_ood_validation only",
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        "sealed_test_outcomes_used_for_fit_or_selection": False,
        "test_outcome_expression_read_during_prediction": False,
        "reporting_only_evaluation": True,
    },
    "result": analysis["findings"],
    "provenance": analysis["provenance"],
}
manifest["manifest_sha256"] = sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path("manifests/state_baseline_v1.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({"best_validation": best_validation, **analysis["findings"]}, indent=2))
