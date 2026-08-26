# Train the primary Stage 2 model from checksum-pinned frozen state/action caches.
# Full training refuses CPU fallback; use scripts/smoke_dynamics.py for local validation.
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.dynamics import train_dynamics

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
if resume_from := os.environ.get("CAUSALCELLJEPA_DYNAMICS_RESUME_FROM"):
    config["training"]["resume_from"] = resume_from
assert torch.cuda.is_available(), "Full Stage 2 dynamics requires CUDA; refusing CPU fallback"
torch.cuda.reset_peak_memory_stats()
_, state, report = train_dynamics(config, torch.device("cuda"))
report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "state": state,
            "report": report,
        },
        indent=2,
    )
)
