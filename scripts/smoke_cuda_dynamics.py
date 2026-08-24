# Run the paid-device gate with a fresh-process checkpoint/resume boundary.
# The full dynamics job is forbidden until this bounded smoke and validation pass.
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.dynamics import LatentPopulationDataset, train_dynamics, validate_dynamics


def load_smoke():
    config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
    smoke = deepcopy(config)
    smoke["training"]["batch_size"] = smoke["cuda_smoke"]["batch_size"]
    smoke["training"]["output_directory"] = smoke["cuda_smoke"]["output_directory"]
    smoke["training"]["checkpoint_every_steps"] = smoke["cuda_smoke"]["checkpoint_step"]
    return smoke


def run_phase(phase):
    assert torch.cuda.is_available(), "CUDA dynamics smoke requires an available CUDA device"
    smoke = load_smoke()
    output = Path(smoke["training"]["output_directory"])
    if phase == "resume":
        smoke["training"]["resume_from"] = str(output / "latest.pt")
        stop = smoke["cuda_smoke"]["steps"]
    else:
        stop = smoke["cuda_smoke"]["checkpoint_step"]
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_dynamics(smoke, torch.device("cuda"), max_steps=stop)
    result = {
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
        metrics = validate_dynamics(
            model,
            validation,
            smoke,
            torch.device("cuda"),
            smoke["cuda_smoke"]["validation_batches"],
        )
        resumed_steps = stop - smoke["cuda_smoke"]["checkpoint_step"]
        result.update(
            {
                "resumed_steps": resumed_steps,
                "conditions_per_second": resumed_steps
                * smoke["cuda_smoke"]["batch_size"]
                / report["elapsed_seconds"],
                "validation_batches": smoke["cuda_smoke"]["validation_batches"],
                "validation": metrics,
            }
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


phase = os.environ.get("CAUSALCELLJEPA_DYNAMICS_CUDA_SMOKE_PHASE")
if phase:
    run_phase(phase)
else:
    output = Path(load_smoke()["training"]["output_directory"])
    assert not (output / "latest.pt").exists()
    for child_phase in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment["CAUSALCELLJEPA_DYNAMICS_CUDA_SMOKE_PHASE"] = child_phase
        subprocess.run([sys.executable, __file__], check=True, env=environment)
    print((output / "resume_report.json").read_text())
