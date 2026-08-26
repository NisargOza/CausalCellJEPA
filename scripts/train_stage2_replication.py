# Train or exactly resume both additional Stage 2 seeds on one paid GPU instance.
# The frozen target split and model protocol are identical to the selected primary run.
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import dynamics_replication_configs, train_dynamics
from causalcelljepa.training import _git_state

assert torch.cuda.is_available(), "Full Stage 2 replication requires CUDA"
assert _git_state()["dirty"] is False, "Full Stage 2 replication requires a clean commit"
configs, _ = dynamics_replication_configs()
reports = {}
for seed, config in configs.items():
    latest = Path(config["training"]["output_directory"]) / "latest.pt"
    config["training"]["resume_from"] = str(latest) if latest.exists() else None
    torch.cuda.reset_peak_memory_stats()
    _, state, report = train_dynamics(
        config, torch.device("cuda"), config_path="configs/stage2_replication.yaml"
    )
    reports[seed] = {
        "state": state,
        "report": report,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
print(json.dumps(reports, indent=2, sort_keys=True))
