# Train the three CPU/CUDA-validated mechanism ablations sequentially on one GPU.
# Each run preserves independent checkpoints, logs, configuration, and provenance.
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import dynamics_ablation_configs, train_dynamics

assert torch.cuda.is_available(), "Full dynamics ablations require CUDA"
configs, _ = dynamics_ablation_configs()
reports = {}
for name, config in configs.items():
    torch.cuda.reset_peak_memory_stats()
    _, state, report = train_dynamics(
        config, torch.device("cuda"), config_path="configs/ablations.yaml"
    )
    report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    reports[name] = {"state": state, "report": report}
    print(json.dumps({name: reports[name]}, indent=2))
print(json.dumps(reports, indent=2, sort_keys=True))
