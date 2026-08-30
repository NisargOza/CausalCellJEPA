"""Validate the transferred hybrid GPU run and freeze its v4-relative decision."""

import json
import math
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

CONFIG_PATH = Path("configs/salt_hybrid_dynamics.yaml")
ROOT = Path("artifacts/salt_hybrid_dynamics/salt_hybrid_static")
CONSOLE_PATH = Path("artifacts/salt_hybrid_training_console.log")
CUDA_SMOKE_PATH = Path("artifacts/salt_hybrid_dynamics_cuda_smoke/report.json")
CPU_SMOKE_PATH = Path("manifests/salt_hybrid_dynamics_cpu_smoke_v1.json")
TRAINING_MANIFEST_PATH = Path("manifests/salt_hybrid_dynamics_training_v1.json")
SELECTION_MANIFEST_PATH = Path("manifests/salt_hybrid_dynamics_selection_v1.json")
REFERENCE_SELECTION_PATH = Path("manifests/contextual_multiteacher_dynamics_selection_v1.json")
REMOTE_SHA256 = {
    "best.pt": "7befe840e28cb4b89827f2ad15173b410fd853e7192375a86fd20d2fe7f63770",
    "latest.pt": "f97f8f549033940621f9e812056db988cf68425cccdd3a948f218f8356c7fbaf",
    "training.jsonl": "1f941c954fa9a61b212ecfe2b4fc2bcc0c598c6ccf7ff92f3e56c1cb49d69c42",
    "console": "959e89b8683becc815d24d7871bebbda7a1912922842c75f5df216a418acc27b",
    "cuda_smoke": "19044e015f0e3ef8bdc45096854487d9cfa13cbd8dbd7ebdb1b49a183cc5a5ed",
}


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_self_hashed(path, field="manifest_sha256"):
    payload = json.loads(path.read_text())
    declared = payload.pop(field)
    assert declared == self_hash(payload)
    return payload, declared


def main():
    assert _git_state()["dirty"] is False, "result freezing requires a clean commit"
    paths = {
        "best.pt": ROOT / "best.pt",
        "latest.pt": ROOT / "latest.pt",
        "training.jsonl": ROOT / "training.jsonl",
        "console": CONSOLE_PATH,
        "cuda_smoke": CUDA_SMOKE_PATH,
    }
    assert all(file_sha256(path) == REMOTE_SHA256[name] for name, path in paths.items())
    specification = yaml.safe_load(CONFIG_PATH.read_text())
    config_sha256 = file_sha256(CONFIG_PATH)
    rows = [json.loads(line) for line in paths["training.jsonl"].read_text().splitlines()]
    train_rows = [row for row in rows if row["event"] == "train_step"]
    validation_rows = [row for row in rows if row["event"] == "validation"]
    assert [row["global_step"] for row in train_rows] == list(range(1, len(train_rows) + 1))
    assert len(train_rows) == 8580 and len(validation_rows) == 195 and all_finite(rows)
    best_row = min(validation_rows, key=lambda row: row["loss"])
    best = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
    latest = torch.load(paths["latest.pt"], map_location="cpu", weights_only=False)
    assert all(torch.isfinite(value).all() for value in best["model"].values())
    assert all(torch.isfinite(value).all() for value in latest["model"].values())
    assert best["state"]["best_validation_epoch"] == best_row["epoch"] == 179
    assert best["state"]["best_validation_loss"] == best_row["loss"]
    assert latest["state"] == {
        "epoch": 195,
        "batch_in_epoch": 0,
        "global_step": 8580,
        "best_validation_loss": best_row["loss"],
        "best_validation_epoch": best_row["epoch"],
        "epochs_without_improvement": 15,
        "complete": True,
        "completion_reason": "early_stopping",
    }
    provenance = latest["provenance"]
    assert best["provenance"] == provenance
    assert provenance["git"] == {
        "commit": "4e79a3eb67da3a2ae7099d64fbac2ef8ea7332e0",
        "dirty": False,
    }
    assert provenance["config_sha256"] == config_sha256
    assert latest["configuration"]["revision"]["candidate"] == "salt_hybrid_static"
    console = json.loads(CONSOLE_PATH.read_text())["salt_hybrid_static"]
    assert console["state"] == latest["state"]

    cuda_smoke, cuda_smoke_hash = load_self_hashed(CUDA_SMOKE_PATH, "report_sha256")
    cpu_smoke, cpu_smoke_hash = load_self_hashed(CPU_SMOKE_PATH)
    assert cuda_smoke["passed"] is True and cuda_smoke["config_sha256"] == config_sha256
    assert cpu_smoke["exact_model_resume"] and cpu_smoke["exact_training_log_resume"]
    assert cuda_smoke["leakage"] == cpu_smoke["leakage"] == {
        "context": "K562",
        "rpe1_outcomes_used": False,
        "sealed_test_outcomes_used": False,
    }

    training = {
        "format_version": 1,
        "architecture": specification["revision"]["name"],
        "protocol": {
            "fit_outcome_role": specification["effect_anchor"]["fit_outcome_role"],
            "selection_outcome_role": specification["selection"]["outcome_role"],
            "context": specification["selection"]["context"],
            "model_and_sampling_seed": specification["seed"],
            "sealed_test_outcomes_used_for_fit_or_selection": False,
            "rpe1_perturbed_outcomes_used_for_fit_or_selection": False,
        },
        "runtime": {
            "python": provenance["runtime_environment"]["python"],
            "torch": provenance["runtime_environment"]["packages"]["torch"],
            "cuda": provenance["runtime_environment"]["torch_cuda"],
            "gpu": provenance["runtime_environment"]["cuda_device"],
        },
        "source": {
            "git_commit": provenance["git"]["commit"],
            "config_sha256": config_sha256,
            "runtime_source_sha256": provenance["runtime_source_sha256"],
            "cache_sha256": provenance["cache_sha256"],
            "manifest_sha256": provenance["manifest_sha256"],
            "cpu_smoke_manifest_sha256": cpu_smoke_hash,
            "cuda_smoke_report_sha256": cuda_smoke_hash,
        },
        "artifacts": {
            "best_checkpoint": artifact(paths["best.pt"]),
            "latest_checkpoint": artifact(paths["latest.pt"]),
            "training_log": {**artifact(paths["training.jsonl"]), "records": len(rows)},
            "console_log": artifact(CONSOLE_PATH),
            "cuda_smoke_report": artifact(CUDA_SMOKE_PATH),
        },
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
            "global_steps": latest["state"]["global_step"],
            "peak_cuda_memory_bytes": console["peak_cuda_memory_bytes"],
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
    training["manifest_sha256"] = self_hash(training)
    TRAINING_MANIFEST_PATH.write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")

    reference, reference_hash = load_self_hashed(REFERENCE_SELECTION_PATH)
    reference_selected = reference["selected"]
    reference_loss = specification["selection"]["reference_validation_loss"]
    assert reference_selected["best_validation_loss"] == reference_loss
    improvement = reference_loss - best_row["loss"]
    margin = specification["selection"]["replacement_minimum_loss_improvement"]
    displaces_reference = improvement >= margin
    assert displaces_reference is False
    selection = {
        "format_version": 1,
        "revision": specification["revision"],
        "candidate_summaries": {
            "availability_static": {
                "best_validation_loss": reference_loss,
                "source_selection_manifest_sha256": reference_hash,
            },
            "salt_hybrid_static": {
                "best_validation_epoch": best_row["epoch"],
                "best_validation_loss": best_row["loss"],
                **training["artifacts"]["best_checkpoint"],
            },
        },
        "leakage": {
            "context": "K562",
            "outcome_role": "perturbation_ood_validation",
            "rpe1_outcomes_used": False,
            "sealed_test_outcomes_used": False,
        },
        "selection_rule": {
            **specification["selection"],
            "observed_hybrid_loss_improvement": improvement,
            "hybrid_displaces_reference": displaces_reference,
        },
        "selected": {
            **reference_selected,
            "retained_reference": True,
            "source_selection_manifest_sha256": reference_hash,
        },
        "provenance": {
            "config_sha256": config_sha256,
            "training_manifest_sha256": training["manifest_sha256"],
        },
    }
    selection["manifest_sha256"] = self_hash(selection)
    SELECTION_MANIFEST_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "hybrid_best_validation_loss": best_row["loss"],
                "reference_validation_loss": reference_loss,
                "observed_hybrid_loss_improvement": improvement,
                "selected": selection["selected"]["candidate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
