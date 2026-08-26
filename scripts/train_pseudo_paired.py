# Train or resume the matched-capacity pseudo-paired comparator; full training refuses CPU.
import json
import os
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import train_dynamics

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
config["objective"] = config["pseudo_paired"]["objective"]
config["training"]["output_directory"] = config["pseudo_paired"]["output_directory"]
if resume_from := os.environ.get("CAUSALCELLJEPA_PSEUDO_RESUME_FROM"):
    config["training"]["resume_from"] = resume_from
assert torch.cuda.is_available(), "Full pseudo-paired training requires CUDA; refusing CPU fallback"
torch.cuda.reset_peak_memory_stats()
_, state, report = train_dynamics(config, torch.device("cuda"))
report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
print(json.dumps({"objective": config["objective"], "state": state, "report": report}, indent=2))
