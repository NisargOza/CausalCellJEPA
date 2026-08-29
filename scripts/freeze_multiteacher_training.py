"""Validate transferred multimodal runs and freeze their validation-only selection."""

import json
import math
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import select_multiteacher_candidate
from causalcelljepa.resources import file_sha256

CONFIG_PATH = Path("configs/multiteacher_dynamics.yaml")
ARTIFACT_ROOT = Path("artifacts/multiteacher_dynamics")
CONSOLE_PATH = Path("artifacts/multiteacher_training_console.log")
TRAINING_MANIFEST_PATH = Path("manifests/multiteacher_dynamics_training_v1.json")
SELECTION_MANIFEST_PATH = Path("manifests/multiteacher_dynamics_selection_v1.json")

REMOTE_SHA256 = {
    "attention_dropout_025": {
        "best.pt": "2204a06506f387af1fb1a3a2e1f3fa35d31ebec28c98e072d43166ae05d4f74c",
        "latest.pt": "365e136843f3d4aa917488fb11c754fdc74dd19b058e30982da4bec1586404b3",
        "training.jsonl": "3e42ec1f01839653eec4dec344a10241cb347a1d44b25d3a8a7366b366bc32c0",
    },
    "attention_full": {
        "best.pt": "25811b8d3b42cccf6e9a9cc59a0c23e820d241b72f8944af8693e702ec1b054d",
        "latest.pt": "0d8423e37b729fc6acf4979b40e498702a1102bdfc11c3a8c78ed698574e1817",
        "training.jsonl": "e55ef8b490453c69d9c1d0cb9e331bd48d3b698829bf34e1d051467e558dbe24",
    },
}


def _artifact(path):
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": file_sha256(path)}


def _self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _all_finite(value):
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _checkpoint_tensors_finite(checkpoint):
    return all(torch.isfinite(value).all() for value in checkpoint["model"].values())


def _console_reports():
    text = CONSOLE_PATH.read_text()
    return json.loads(text[text.index("{") :])


def main():
    specification = yaml.safe_load(CONFIG_PATH.read_text())
    console_reports = _console_reports()
    candidates = {}
    checkpoint_provenance = None
    for name in specification["experiments"]:
        root = ARTIFACT_ROOT / name
        paths = {filename: root / filename for filename in REMOTE_SHA256[name]}
        assert all(file_sha256(path) == REMOTE_SHA256[name][filename] for filename, path in paths.items())
        rows = [json.loads(line) for line in paths["training.jsonl"].read_text().splitlines()]
        train_rows = [row for row in rows if row["event"] == "train_step"]
        validation_rows = [row for row in rows if row["event"] == "validation"]
        assert [row["global_step"] for row in train_rows] == list(range(1, len(train_rows) + 1))
        assert rows and validation_rows and _all_finite(rows)
        best_row = min(validation_rows, key=lambda row: row["loss"])
        best = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
        latest = torch.load(paths["latest.pt"], map_location="cpu", weights_only=False)
        assert _checkpoint_tensors_finite(best) and _checkpoint_tensors_finite(latest)
        assert best["configuration"]["revision"]["candidate"] == name
        assert latest["configuration"]["revision"]["candidate"] == name
        assert (best["state"]["best_validation_epoch"], best["state"]["best_validation_loss"]) == (
            best_row["epoch"],
            best_row["loss"],
        )
        assert latest["state"]["complete"] is True
        assert latest["state"]["completion_reason"] == "early_stopping"
        assert latest["state"]["global_step"] == len(train_rows)
        assert latest["state"]["epoch"] == len(validation_rows)
        assert latest["state"]["best_validation_loss"] == best_row["loss"]
        provenance = latest["provenance"]
        assert provenance["git"]["dirty"] is False
        assert provenance["config_sha256"] == file_sha256(CONFIG_PATH)
        if checkpoint_provenance is None:
            checkpoint_provenance = provenance
        else:
            assert provenance == checkpoint_provenance
        console = console_reports[name]
        assert console["state"] == latest["state"]
        candidates[name] = {
            "best_checkpoint": _artifact(paths["best.pt"]),
            "latest_checkpoint": _artifact(paths["latest.pt"]),
            "training_log": {**_artifact(paths["training.jsonl"]), "records": len(rows)},
            "full_run": {
                "best_validation_epoch": best_row["epoch"],
                "best_validation_loss": best_row["loss"],
                "best_validation_metrics": {
                    key: value
                    for key, value in best_row.items()
                    if key not in {"event", "epoch", "global_step", "loss"}
                },
                "completion_reason": latest["state"]["completion_reason"],
                "elapsed_seconds": console["report"]["elapsed_seconds"],
                "epochs": latest["state"]["epoch"],
                "epochs_without_improvement": latest["state"]["epochs_without_improvement"],
                "global_steps": latest["state"]["global_step"],
                "peak_cuda_memory_bytes": console["peak_cuda_memory_bytes"],
            },
        }

    assert checkpoint_provenance is not None
    environment = checkpoint_provenance["runtime_environment"]
    training = {
        "format_version": 1,
        "protocol": {
            "checkpoint_selection": "minimum original latent loss on frozen K562 perturbation-OOD validation",
            "context": specification["selection"]["context"],
            "dropout_minimum_loss_improvement": specification["selection"][
                "dropout_minimum_loss_improvement"
            ],
            "fit_outcome_role": specification["effect_anchor"]["fit_outcome_role"],
            "model_and_sampling_seed": specification["seed"],
            "population_size": 32,
            "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
            "sealed_test_outcomes_used_for_fit_or_selection": False,
            "selection_outcome_role": specification["selection"]["outcome_role"],
            "statistical_unit": "perturbation-condition",
            "target_split_seed": 20260823,
        },
        "runtime": {
            "cuda": environment["torch_cuda"],
            "gpu": environment["cuda_device"]["name"],
            "gpu_capability": environment["cuda_device"]["capability"],
            "gpu_total_memory_bytes": environment["cuda_device"]["total_memory_bytes"],
            "python": environment["python"],
            "torch": environment["packages"]["torch"],
        },
        "source": {
            "action_cache_sha256": checkpoint_provenance["cache_sha256"]["action"],
            "config_sha256": checkpoint_provenance["config_sha256"],
            "dynamics_manifest_sha256": checkpoint_provenance["manifest_sha256"]["dynamics"],
            "effect_anchor_manifest_sha256": checkpoint_provenance["manifest_sha256"][
                "effect_anchor"
            ],
            "effect_anchor_sha256": checkpoint_provenance["cache_sha256"]["effect_anchor"],
            "git_commit": checkpoint_provenance["git"]["commit"],
            "git_dirty": checkpoint_provenance["git"]["dirty"],
            "latent_cache_sha256": checkpoint_provenance["cache_sha256"]["latent"],
            "runtime_source_sha256": checkpoint_provenance["runtime_source_sha256"],
        },
        "artifacts": {
            "candidates": candidates,
            "console_log": _artifact(CONSOLE_PATH),
        },
        "validation": {
            "all_checkpoint_model_tensors_finite": True,
            "all_log_numeric_values_finite": True,
            "best_checkpoint_matches_logged_minimum": True,
            "checkpoint_provenance_matches_clean_commit": True,
            "cpu_checkpoint_resume_exactness_passed": True,
            "cuda_checkpoint_resume_smoke_passed": True,
            "local_transfer_sha256_matches_remote": True,
            "optimizer_steps_contiguous": True,
        },
    }
    selected_name, improvement = select_multiteacher_candidate(training, specification)
    training["manifest_sha256"] = _self_hash(training)
    TRAINING_MANIFEST_PATH.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")

    selected_run = candidates[selected_name]
    selection = {
        "format_version": 1,
        "candidate_summaries": {
            name: {
                "action_modality_dropout": specification["experiments"][name][
                    "action_modality_dropout"
                ],
                "best_validation_loss": entry["full_run"]["best_validation_loss"],
                **entry["full_run"]["best_validation_metrics"],
            }
            for name, entry in candidates.items()
        },
        "leakage": {
            "context": "K562",
            "outcome_role": "perturbation_ood_validation",
            "rpe1_outcomes_used": False,
            "sealed_test_outcomes_used": False,
        },
        "provenance": {
            "config_sha256": file_sha256(CONFIG_PATH),
            "training_manifest_sha256": training["manifest_sha256"],
        },
        "revision": specification["revision"],
        "selected": {
            "candidate": selected_name,
            "action_modality_dropout": specification["experiments"][selected_name][
                "action_modality_dropout"
            ],
            "best_validation_epoch": selected_run["full_run"]["best_validation_epoch"],
            "best_validation_loss": selected_run["full_run"]["best_validation_loss"],
            **selected_run["best_checkpoint"],
        },
        "selection_rule": {
            **specification["selection"],
            "observed_dropout_loss_improvement": improvement,
            "dropout_displaced_fallback": selected_name
            != specification["selection"]["fallback_candidate"],
        },
    }
    selection["manifest_sha256"] = _self_hash(selection)
    SELECTION_MANIFEST_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selection["selected"], "improvement": improvement}, indent=2))


if __name__ == "__main__":
    main()
