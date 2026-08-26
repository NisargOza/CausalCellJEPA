# Train or exactly resume the preregistered anchored candidates on one GPU.
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import anchored_dynamics_configs, train_dynamics
from causalcelljepa.training import _git_state

assert torch.cuda.is_available(), "Full anchored dynamics training requires CUDA"
assert _git_state()["dirty"] is False, "Full anchored training requires a clean commit"
configs, _ = anchored_dynamics_configs()
reports = {}
for name, config in configs.items():
    latest = Path(config["training"]["output_directory"]) / "latest.pt"
    config["training"]["resume_from"] = str(latest) if latest.exists() else None
    torch.cuda.reset_peak_memory_stats()
    _, state, report = train_dynamics(
        config, torch.device("cuda"), config_path="configs/anchored_dynamics.yaml"
    )
    reports[name] = {
        "state": state,
        "report": report,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
print(json.dumps(reports, indent=2, sort_keys=True))
