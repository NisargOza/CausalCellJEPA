# Run fresh-process bounded CUDA checkpoint/resume gates for both replication seeds.
# Full replication training is forbidden until every seed completes this script.
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
    dynamics_replication_configs,
    train_dynamics,
    validate_dynamics,
)
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "CUDA replication smoke requires a clean commit"


def run_phase(seed, phase):
    assert torch.cuda.is_available(), "CUDA replication smoke requires an available GPU"
    configs, specification = dynamics_replication_configs()
    config = deepcopy(configs[seed])
    smoke = specification["cuda_smoke"]
    output = Path(smoke["output_directory"]) / str(seed)
    config["training"]["batch_size"] = smoke["batch_size"]
    config["training"]["checkpoint_every_steps"] = smoke["checkpoint_step"]
    config["training"]["output_directory"] = str(output)
    stop = smoke["checkpoint_step"]
    if phase == "resume":
        config["training"]["resume_from"] = str(output / "latest.pt")
        stop = smoke["steps"]
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_dynamics(
        config,
        torch.device("cuda"),
        max_steps=stop,
        config_path="configs/stage2_replication.yaml",
    )
    result = {
        "seed": seed,
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": report["elapsed_seconds"],
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
            seed,
        )
        result["validation"] = validate_dynamics(
            model,
            validation,
            config,
            torch.device("cuda"),
            smoke["validation_batches"],
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


phase = os.environ.get("CAUSALCELLJEPA_REPLICATION_CUDA_PHASE")
seed_value = os.environ.get("CAUSALCELLJEPA_REPLICATION_CUDA_SEED")
if phase:
    run_phase(int(seed_value), phase)
else:
    configs, specification = dynamics_replication_configs()
    root = Path(specification["cuda_smoke"]["output_directory"])
    for seed in configs:
        assert not (root / str(seed) / "latest.pt").exists()
        for child_phase in ("checkpoint", "resume"):
            environment = os.environ.copy()
            environment["CAUSALCELLJEPA_REPLICATION_CUDA_PHASE"] = child_phase
            environment["CAUSALCELLJEPA_REPLICATION_CUDA_SEED"] = str(seed)
            subprocess.run([sys.executable, __file__], check=True, env=environment)
