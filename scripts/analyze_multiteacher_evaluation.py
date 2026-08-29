# Audit the frozen multimodal model against every previously frozen gene-level reference.
# This reporting-only analysis averages repeats within perturbations and never selects a model.
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from causalcelljepa.dynamics import build_dynamics_model
from causalcelljepa.readout import _paired_transcriptomic_models
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

seed, resamples, numerical_tie_tolerance = 20260823, 10_000, 2e-6
candidate = "multiteacher_selected"
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


verified_sources = {}
for name, (manifest_path, root, models) in source_specs.items():
    manifest, declared = self_hashed_manifest(manifest_path)
    artifact_entries = manifest["artifacts"].get("predictive_evaluation", manifest["artifacts"])
    artifacts = {}
    for kind, filename in artifact_names.items():
        entry, path = artifact_entries[kind], root / filename
        assert (path.stat().st_size, file_sha256(path)) == (entry["bytes"], entry["sha256"])
        artifacts[kind] = artifact(path)
    verified_sources[name] = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": declared,
        "models": models,
        "artifacts": artifacts,
    }

anchored_manifest_path = Path("manifests/evaluation_anchored_v1.json")
anchored_manifest, anchored_manifest_sha256 = self_hashed_manifest(anchored_manifest_path)
anchored_root = Path("artifacts/evaluation_anchored_transcriptomics")
anchored_entries = anchored_manifest["artifacts"]["transcriptomic_evaluation"]
anchored_artifacts = {}
for kind, filename in artifact_names.items():
    entry, path = anchored_entries[kind], anchored_root / filename
    assert (path.stat().st_size, file_sha256(path)) == (entry["bytes"], entry["sha256"])
    anchored_artifacts[kind] = artifact(path)
verified_sources["anchored"] = {
    "manifest_path": str(anchored_manifest_path),
    "manifest_sha256": anchored_manifest_sha256,
    "models": ["anchored_selected"],
    "artifacts": anchored_artifacts,
}

candidate_root = Path("artifacts/evaluation_multiteacher_transcriptomics")
candidate_provenance = json.loads((candidate_root / "provenance.json").read_text())
assert candidate_provenance["git"]["dirty"] is False
assert candidate_provenance["executed_repeats"] == 8
assert candidate_provenance["maximum_conditions_per_regime"] is None
assert list(candidate_provenance["executed_regimes"]) == sorted(regimes)
multiteacher_manifests = {}
for kind in ("training", "selection"):
    manifest_path = Path(f"manifests/multiteacher_dynamics_{kind}_v1.json")
    payload, declared = self_hashed_manifest(manifest_path)
    assert declared == candidate_provenance[f"multiteacher_{kind}_manifest_sha256"]
    multiteacher_manifests[kind] = payload

selected = multiteacher_manifests["selection"]["selected"]
checkpoint_path = Path(selected["path"])
assert (checkpoint_path.stat().st_size, file_sha256(checkpoint_path)) == (
    selected["bytes"],
    selected["sha256"],
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = build_dynamics_model(checkpoint["configuration"]).eval()
model.load_state_dict(checkpoint["model"])
action_path = Path(checkpoint["configuration"]["inputs"]["action_cache_path"])
assert file_sha256(action_path) == checkpoint["provenance"]["cache_sha256"]["action"]
action = torch.load(action_path, map_location="cpu", weights_only=True)
projection = model.action_projection
with torch.no_grad():
    blocks = action["embedding"].split(projection.modality_dims, 1)
    projected = torch.stack(
        [layer(block) for layer, block in zip(projection.projectors, blocks, strict=True)],
        1,
    )
    scores = (torch.tanh(projected) * projection.query).sum(2) / np.sqrt(projected.shape[2])
    attention = scores.softmax(1)
_, inverse, counts = torch.unique(
    action["embedding"][:, action["modality_dims"][0] :],
    dim=0,
    return_inverse=True,
    return_counts=True,
)
unannotated = inverse == counts.argmax()
action_manifest, _ = self_hashed_manifest(Path("manifests/multiteacher_action_v1.json"))
assert int(unannotated.sum()) == len(action["targets"]) - action_manifest["report"]["targets_with_go"]
attention_audit = {
    "targets": len(action["targets"]),
    "annotated_targets": int((~unannotated).sum()),
    "unannotated_targets": int(unannotated.sum()),
    "annotated_mean_weights": {
        name: float(attention[~unannotated, index].mean())
        for index, name in enumerate(action["modalities"])
    },
    "unannotated_mean_weights": {
        name: float(attention[unannotated, index].mean())
        for index, name in enumerate(action["modalities"])
    },
    "score_difference_standard_deviation": float((scores[:, 0] - scores[:, 1]).std()),
    "outcomes_read": False,
}

record_fields = {"regime", "model", "target", "repeat"}
condition_records, pathway_records = [], []
raw_sources = [
    (candidate_root, [candidate]),
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

reference_models = [
    "anchored_selected",
    *[model for _, _, models in source_specs.values() for model in models],
]
pairs = [
    {
        "candidate": candidate,
        "reference": reference,
        "hypothesis": "frozen multimodal model differs from a previously frozen reference",
    }
    for reference in reference_models
]
condition_comparisons = _paired_transcriptomic_models(condition_records, pairs, resamples, seed)
pathway_comparisons = _paired_transcriptomic_models(pathway_records, pairs, resamples, seed)
undefined_no_change = {
    "condition": [
        "all_effect_pearson",
        "target_excluded_effect_pearson",
        "retrospective_top50_effect_pearson",
    ],
    "pathway": ["pathway_nes_pearson"],
}
assert len(condition_comparisons) == len(regimes) * (
    len(reference_models) * len(condition_metrics) - len(undefined_no_change["condition"])
)
assert len(pathway_comparisons) == len(regimes) * (
    len(reference_models) * len(pathway_metrics) - len(undefined_no_change["pathway"])
)

models = [candidate, *reference_models]
means = {}
summary_sources = [
    (candidate_root / "summary.json", [candidate]),
    (anchored_root / "summary.json", ["anchored_selected"]),
    *[(root / "summary.json", source_models) for _, root, source_models in source_specs.values()],
]
for path, source_models in summary_sources:
    summary = json.loads(path.read_text())
    for section in ("condition_metrics", "pathway_metrics"):
        for item in summary[section]:
            if item["model"] in source_models and item["metric"] in condition_metrics + pathway_metrics:
                means[item["regime"], item["model"], item["metric"]] = item["mean"]

headline_means, rankings = {}, {}
for regime in regimes:
    headline_means[regime], rankings[regime] = {}, {}
    for metric in condition_metrics + pathway_metrics:
        values = {model: means.get((regime, model, metric)) for model in models}
        comparable = {model: value for model, value in values.items() if value is not None}
        lower_is_better = metric.endswith(("absolute_error", "rmse"))
        ordered = sorted(comparable, key=comparable.get, reverse=not lower_is_better)
        candidate_value = values[candidate]
        rankings[regime][metric] = {
            "multiteacher_rank": 1
            + sum(
                value < candidate_value - numerical_tie_tolerance
                if lower_is_better
                else value > candidate_value + numerical_tie_tolerance
                for value in comparable.values()
            ),
            "models_evaluated": len(comparable),
            "models_undefined": sorted(set(models) - set(comparable)),
            "ordered_models": ordered,
            "multiteacher_ties": [
                model
                for model, value in comparable.items()
                if model != candidate
                and np.isclose(value, candidate_value, rtol=0, atol=numerical_tie_tolerance)
            ],
        }
        headline_means[regime][metric] = values

all_comparisons = condition_comparisons + pathway_comparisons
verdict_counts = {"win": 0, "inconclusive": 0, "loss": 0}
significant_counts = {"win": 0, "loss": 0}
for item in all_comparisons:
    low, high = item["mean_improvement_bootstrap_95ci"]
    verdict = "win" if low > 0 else "loss" if high < 0 else "inconclusive"
    item["bootstrap_95ci_verdict"] = verdict
    verdict_counts[verdict] += 1
    if item["benjamini_hochberg_q"] < 0.05 and verdict in significant_counts:
        significant_counts[verdict] += 1

anchored_comparisons = [item for item in all_comparisons if item["reference"] == "anchored_selected"]
latent_pairs = json.loads(
    Path("artifacts/evaluation_multiteacher/paired_comparisons.json").read_text()
)
latent_vs_anchored = [
    item
    for item in latent_pairs
    if item["candidate"] == candidate and item["reference"] == "anchored_selected"
]
assert len(anchored_comparisons) == 28 and len(latent_vs_anchored) == 40

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
        "undefined_zero_effect_metrics": undefined_no_change,
    },
    "headline_means": headline_means,
    "rankings": rankings,
    "paired_comparisons": {
        "all_condition": condition_comparisons,
        "all_pathway": pathway_comparisons,
        "transcriptomic_vs_anchored": anchored_comparisons,
        "latent_vs_anchored": latent_vs_anchored,
        "bootstrap_95ci_verdict_counts": verdict_counts,
        "benjamini_hochberg_q05_counts": significant_counts,
    },
    "findings": {
        "uniform_superiority_supported": verdict_counts["loss"] == 0,
        "models_compared": len(models),
        "reference_models": reference_models,
        "attention_audit": attention_audit,
    },
    "provenance": {
        "sources": verified_sources,
        "candidate_evaluation_git": candidate_provenance["git"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert analysis["findings"]["uniform_superiority_supported"] is False
assert analysis["provenance"]["git"]["dirty"] is False
output = Path("artifacts/evaluation_multiteacher_analysis")
assert not output.exists()
output.mkdir(parents=True)
analysis_path = output / "analysis.json"
analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

evaluation_artifacts = {}
for name, root, filenames in (
    (
        "latent_evaluation",
        Path("artifacts/evaluation_multiteacher"),
        ["condition_metrics.jsonl", "paired_comparisons.json", "provenance.json", "summary.json"],
    ),
    (
        "transcriptomic_evaluation",
        candidate_root,
        [
            "condition_metrics.jsonl",
            "paired_comparisons.json",
            "pathway_metrics.jsonl",
            "provenance.json",
            "summary.json",
            "truth_report.json",
        ],
    ),
):
    evaluation_artifacts[name] = {}
    for filename in filenames:
        path = root / filename
        records = len(path.read_text().splitlines()) if filename.endswith(".jsonl") else None
        evaluation_artifacts[name][filename] = artifact(path, records)
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
        "bootstrap_95ci_verdict_counts": verdict_counts,
        "benjamini_hochberg_q05_counts": significant_counts,
        "attention_audit": attention_audit,
    },
    "provenance": {
        "evaluation_git": candidate_provenance["git"],
        "analysis_git": analysis["provenance"]["git"],
        "evaluation_config_sha256": file_sha256("configs/evaluation_multiteacher.yaml"),
        "multiteacher_training_manifest_sha256": candidate_provenance[
            "multiteacher_training_manifest_sha256"
        ],
        "multiteacher_selection_manifest_sha256": candidate_provenance[
            "multiteacher_selection_manifest_sha256"
        ],
    },
}
manifest["manifest_sha256"] = sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path("manifests/evaluation_multiteacher_v1.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(
    {
        "models": len(models),
        "paired_comparisons": len(all_comparisons),
        **verdict_counts,
        **{f"q05_{key}": value for key, value in significant_counts.items()},
    }
)
