# Train the CPU/CUDA-validated learned target-ID comparator on one GPU.
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import learned_target_id_config, train_dynamics

assert torch.cuda.is_available(), "Full learned target-ID training requires CUDA"
config, _, _ = learned_target_id_config()
torch.cuda.reset_peak_memory_stats()
_, state, report = train_dynamics(
    config, torch.device("cuda"), config_path="configs/learned_target_id.yaml"
)
report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
print(json.dumps({"learned_target_id": {"state": state, "report": report}}, indent=2))
