# Train the locked Stage 2 dynamics model in the frozen PCA state space.
# Full training is CUDA-only and may begin only after both smoke gates pass.
import json

import torch

from causalcelljepa.dynamics import state_ablation_config, train_dynamics

config_path = "configs/pca_state.yaml"
config, _, _ = state_ablation_config(config_path)
assert torch.cuda.is_available(), "Full PCA dynamics training requires CUDA"
_, state, report = train_dynamics(config, torch.device("cuda"), config_path=config_path)
print(json.dumps({"state": state, "report": report}, indent=2))
