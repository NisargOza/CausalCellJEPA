# Run a bounded fresh-process CUDA checkpoint/resume gate for learned target ID.
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
    learned_target_id_config,
    train_dynamics,
    validate_dynamics,
)


def load_smoke():
    config, specification, _ = learned_target_id_config()
    smoke = deepcopy(config)
    settings = specification["cuda_smoke"]
    smoke["training"]["batch_size"] = settings["batch_size"]
    smoke["training"]["output_directory"] = settings["output_directory"]
    smoke["training"]["checkpoint_every_steps"] = settings["checkpoint_step"]
    return smoke, settings


def run_phase(phase):
    assert torch.cuda.is_available()
    smoke, settings = load_smoke()
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
        config_path="configs/learned_target_id.yaml",
    )
    result = {
        "experiment": "learned_target_id",
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


phase = os.environ.get("CAUSALCELLJEPA_LEARNED_ID_SMOKE_PHASE")
if phase:
    run_phase(phase)
else:
    _, settings = load_smoke()
    output = Path(settings["output_directory"])
    assert not (output / "latest.pt").exists()
    for child_phase in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment["CAUSALCELLJEPA_LEARNED_ID_SMOKE_PHASE"] = child_phase
        subprocess.run([sys.executable, __file__], check=True, env=environment)
