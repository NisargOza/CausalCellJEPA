# Freeze the paired distribution-transfer analysis for the post-hoc residual gate.
# This reporting step cannot tune the already-frozen gate or replace v1 conclusions.
import json
from hashlib import sha256
from pathlib import Path

import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

regimes = ["iid", "perturbation_ood", "context_ood", "double_ood"]
distribution_metrics = ["sinkhorn", "mmd", "energy_distance", "covariance_shift_error"]
invariant_metrics = ["effect_pearson", "centroid_shift_error", "magnitude_ratio"]
config_path = Path("configs/evaluation_control_ood_residual_gate.yaml")
config = yaml.safe_load(config_path.read_text())
root = Path(config["output_directory"])
provenance = json.loads((root / "provenance.json").read_text())
assert provenance["config"] == config and provenance["git"]["dirty"] is False
assert provenance["executed_repeats"] == 8 and provenance["maximum_conditions_per_regime"] is None

summary = json.loads((root / "summary.json").read_text())["condition_metrics"]
old_summary = json.loads(Path("artifacts/evaluation_anchored/summary.json").read_text())[
    "condition_metrics"
]
means = {
    (item["regime"], item["model"], item["metric"]): item["mean"] for item in summary + old_summary
}
comparisons = json.loads((root / "paired_comparisons.json").read_text())
paired = {(item["regime"], item["reference"], item["metric"]): item for item in comparisons}
gated = "anchored_control_ood_gated"
headline, invariance = {}, {}
for regime in regimes:
    headline[regime] = {
        "residual_gate_confidence": means[regime, gated, "residual_gate_confidence"],
        "metrics": {
            metric: {
                "gated": means[regime, gated, metric],
                "ungated": means[regime, "anchored_selected", metric],
                "primary": means[regime, "causalcelljepa", metric],
                "linear_anchor": means[regime, "linear_esm", metric],
                "paired_improvement_over_ungated": paired[regime, "anchored_selected", metric][
                    "mean_improvement"
                ],
                "paired_improvement_bootstrap_95ci": paired[regime, "anchored_selected", metric][
                    "mean_improvement_bootstrap_95ci"
                ],
                "benjamini_hochberg_q": paired[regime, "anchored_selected", metric][
                    "benjamini_hochberg_q"
                ],
            }
            for metric in distribution_metrics
        },
    }
    invariance[regime] = {
        metric: paired[regime, "anchored_selected", metric]["mean_improvement"]
        for metric in invariant_metrics
    }

for regime in ("context_ood", "double_ood"):
    for metric in distribution_metrics:
        assert headline[regime]["metrics"][metric]["paired_improvement_bootstrap_95ci"][0] > 0
        assert means[regime, gated, metric] == means[regime, "linear_esm", metric]
assert max(abs(value) for item in invariance.values() for value in item.values()) < 1e-8
assert min(headline[regime]["residual_gate_confidence"] for regime in regimes[:2]) > 0.99
assert max(headline[regime]["residual_gate_confidence"] for regime in regimes[2:]) < 1e-15

verdict_counts = {"win": 0, "inconclusive": 0, "loss": 0}
for item in comparisons:
    low, high = item["mean_improvement_bootstrap_95ci"]
    verdict_counts["win" if low > 0 else "loss" if high < 0 else "inconclusive"] += 1
analysis = {
    "format_version": 1,
    "scope": {
        "status": "post_hoc_exploratory",
        "selection_used_sealed_outcomes": False,
        "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        "statistical_unit": "perturbation-condition",
        "evaluation_repeats": 8,
        "bootstrap_resamples": 10_000,
        "decoded_mean_metrics_rerun": False,
    },
    "headline": headline,
    "mean_invariance_improvement_over_ungated": invariance,
    "paired_bootstrap_95ci_verdict_counts": verdict_counts,
    "findings": {
        "all_rpe1_distribution_metrics_improve_over_ungated": True,
        "rpe1_fallback_matches_linear_anchor": True,
        "transferred_learned_heterogeneity_supported": False,
        "uniform_superiority_supported": False,
        "new_untouched_context_required_for_confirmation": True,
    },
    "provenance": {
        "evaluation_git": provenance["git"],
        "evaluation_config_sha256": file_sha256(config_path),
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert analysis["provenance"]["git"]["dirty"] is False
output = Path("artifacts/evaluation_control_ood_residual_gate_analysis")
assert not output.exists()
output.mkdir(parents=True)
analysis_path = output / "analysis.json"
analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
artifacts = {}
for name, path in {
    "condition_metrics": root / "condition_metrics.jsonl",
    "paired_comparisons": root / "paired_comparisons.json",
    "provenance": root / "provenance.json",
    "summary": root / "summary.json",
    "analysis": analysis_path,
}.items():
    artifacts[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
manifest = {
    "format_version": 1,
    "protocol": analysis["scope"],
    "artifacts": artifacts,
    "headline": headline,
    "findings": analysis["findings"],
    "source": analysis["provenance"],
}
manifest["manifest_sha256"] = sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path("manifests/evaluation_control_ood_residual_gate_v1.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print({"comparisons": len(comparisons), **verdict_counts})
