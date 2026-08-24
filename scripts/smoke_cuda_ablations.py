# Run bounded fresh-process CUDA checkpoint/resume gates for all mechanism ablations.
# The full paid batch is forbidden until every experiment validates on the device.
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    dynamics_ablation_configs,
    train_dynamics,
    validate_dynamics,
)


def load_smoke(name):
    configs, specification = dynamics_ablation_configs()
    smoke = deepcopy(configs[name])
    settings = specification["cuda_smoke"]
    smoke["training"]["batch_size"] = settings["batch_size"]
    smoke["training"]["output_directory"] = str(Path(settings["output_root"]) / name)
    smoke["training"]["checkpoint_every_steps"] = settings["checkpoint_step"]
    return smoke, settings


def run_phase(name, phase):
    assert torch.cuda.is_available()
    smoke, settings = load_smoke(name)
    output = Path(smoke["training"]["output_directory"])
    stop = settings["checkpoint_step"]
    if phase == "resume":
        smoke["training"]["resume_from"] = str(output / "latest.pt")
        stop = settings["steps"]
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_dynamics(
        smoke,
        torch.device("cuda"),
        max_steps=stop,
        config_path="configs/ablations.yaml",
    )
    result = {
        "experiment": name,
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if phase == "resume":
        inputs, data = smoke["inputs"], smoke["data"]
        validation = LatentPopulationDataset(
            inputs["latent_cache_path"],
            inputs["action_cache_path"],
            inputs["dynamics_manifest_path"],
            "validation",
            data["population_size"],
            smoke["seed"],
        )
        result["validation"] = validate_dynamics(
            model,
            validation,
            smoke,
            torch.device("cuda"),
            settings["validation_batches"],
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


name, phase = (
    os.environ.get("CAUSALCELLJEPA_ABLATION_SMOKE_NAME"),
    os.environ.get("CAUSALCELLJEPA_ABLATION_SMOKE_PHASE"),
)
if name and phase:
    run_phase(name, phase)
else:
    configs, specification = dynamics_ablation_configs()
    for child_name in configs:
        output = Path(specification["cuda_smoke"]["output_root"]) / child_name
        assert not (output / "latest.pt").exists()
        for child_phase in ("checkpoint", "resume"):
            environment = os.environ.copy()
            environment["CAUSALCELLJEPA_ABLATION_SMOKE_NAME"] = child_name
            environment["CAUSALCELLJEPA_ABLATION_SMOKE_PHASE"] = child_phase
            subprocess.run([sys.executable, __file__], check=True, env=environment)
