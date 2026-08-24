# Bound paid-device risk with a fresh-process checkpoint/resume pseudo-paired smoke.
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import LatentPopulationDataset, train_dynamics, validate_dynamics

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
config["objective"] = config["pseudo_paired"]["objective"]
config["training"]["batch_size"] = config["cuda_smoke"]["batch_size"]
config["training"]["output_directory"] = config["pseudo_paired"]["cuda_smoke_output_directory"]
config["training"]["checkpoint_every_steps"] = config["cuda_smoke"]["checkpoint_step"]
phase = os.environ.get("CAUSALCELLJEPA_PSEUDO_SMOKE_PHASE")
if phase:
    assert torch.cuda.is_available()
    output = Path(config["training"]["output_directory"])
    stop = config["cuda_smoke"]["checkpoint_step"]
    if phase == "resume":
        config["training"]["resume_from"] = str(output / "latest.pt")
        stop = config["cuda_smoke"]["steps"]
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_dynamics(config, torch.device("cuda"), max_steps=stop)
    result = {
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if phase == "resume":
        validation = LatentPopulationDataset(
            config["inputs"]["latent_cache_path"],
            config["inputs"]["action_cache_path"],
            config["inputs"]["dynamics_manifest_path"],
            "validation",
            config["data"]["population_size"],
            config["seed"],
        )
        result["validation"] = validate_dynamics(
            model, validation, config, torch.device("cuda"), config["cuda_smoke"]["validation_batches"]
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
else:
    output = Path(config["training"]["output_directory"])
    assert not output.exists()
    for child_phase in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment["CAUSALCELLJEPA_PSEUDO_SMOKE_PHASE"] = child_phase
        subprocess.run([sys.executable, __file__], check=True, env=environment)
