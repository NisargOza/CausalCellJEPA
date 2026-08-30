# Run a bounded CUDA checkpoint/resume and validation smoke for the hybrid candidate.
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    anchored_dynamics_configs,
    train_dynamics,
    validate_dynamics,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

PATH = Path("configs/salt_hybrid_dynamics.yaml")
PHASE_VARIABLE = "CAUSALCELLJEPA_SALT_HYBRID_CUDA_PHASE"


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_phase(phase):
    assert torch.cuda.is_available(), "hybrid CUDA smoke requires an available GPU"
    assert _git_state()["dirty"] is False, "hybrid CUDA smoke requires a clean commit"
    configs, specification = anchored_dynamics_configs(PATH)
    config = deepcopy(configs["salt_hybrid_static"])
    smoke = specification["cuda_smoke"]
    output = Path(smoke["output_directory"]) / "salt_hybrid_static"
    config["training"]["batch_size"] = smoke["batch_size"]
    config["training"]["checkpoint_every_steps"] = smoke["checkpoint_step"]
    config["training"]["output_directory"] = str(output)
    stop = smoke["checkpoint_step"]
    if phase == "resume":
        config["training"]["resume_from"] = str(output / "latest.pt")
        stop = smoke["steps"]
    torch.cuda.reset_peak_memory_stats()
    model, state, training_report = train_dynamics(
        config, torch.device("cuda"), max_steps=stop, config_path=PATH
    )
    result = {
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": training_report["elapsed_seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if phase == "resume":
        inputs, data = config["inputs"], config["data"]
        validation = LatentPopulationDataset(
            inputs["latent_cache_path"],
            inputs["action_cache_path"],
            inputs["dynamics_manifest_path"],
            "validation",
            data["population_size"],
            config["seed"],
        )
        result["validation"] = validate_dynamics(
            model, validation, config, torch.device("cuda"), smoke["validation_batches"]
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    assert torch.cuda.is_available(), "hybrid CUDA smoke requires an available GPU"
    assert _git_state()["dirty"] is False, "hybrid CUDA smoke requires a clean commit"
    configs, specification = anchored_dynamics_configs(PATH)
    assert list(configs) == ["salt_hybrid_static"]
    smoke = specification["cuda_smoke"]
    output = Path(smoke["output_directory"]) / "salt_hybrid_static"
    assert not (output / "latest.pt").exists()
    for phase in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment[PHASE_VARIABLE] = phase
        subprocess.run([sys.executable, __file__], check=True, env=environment)
    phase_reports = {
        phase: json.loads((output / f"{phase}_report.json").read_text())
        for phase in ("checkpoint", "resume")
    }
    report = {
        "format_version": 1,
        "architecture": specification["revision"]["name"],
        "config_sha256": file_sha256(PATH),
        "git": _git_state(),
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "phases": phase_reports,
        "leakage": {
            "context": "K562",
            "sealed_test_outcomes_used": False,
            "rpe1_outcomes_used": False,
        },
    }
    report["passed"] = bool(
        phase_reports["checkpoint"]["global_step"] == smoke["checkpoint_step"]
        and phase_reports["resume"]["global_step"] == smoke["steps"]
        and all_finite(phase_reports)
    )
    assert report["passed"]
    report["report_sha256"] = self_hash(report)
    report_path = Path(smoke["report_path"])
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


phase = os.environ.get(PHASE_VARIABLE)
if phase:
    run_phase(phase)
else:
    main()
