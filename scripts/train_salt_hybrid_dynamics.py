# Train or exactly resume the single frozen hybrid candidate on one GPU.
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from causalcelljepa.dynamics import anchored_dynamics_configs, train_dynamics
from causalcelljepa.training import _git_state

PATH = Path("configs/salt_hybrid_dynamics.yaml")


def main():
    assert torch.cuda.is_available(), "full hybrid dynamics training requires CUDA"
    assert _git_state()["dirty"] is False, "full hybrid training requires a clean commit"
    configs, _ = anchored_dynamics_configs(PATH)
    config = configs["salt_hybrid_static"]
    latest = Path(config["training"]["output_directory"]) / "latest.pt"
    config["training"]["resume_from"] = str(latest) if latest.exists() else None
    torch.cuda.reset_peak_memory_stats()
    _, state, report = train_dynamics(config, torch.device("cuda"), config_path=PATH)
    print(
        json.dumps(
            {
                "salt_hybrid_static": {
                    "state": state,
                    "report": report,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
