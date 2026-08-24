# Train the locked Stage 2 population dynamics in the frozen autoencoder state space.
# Full training is CUDA-only and uses K562 perturbation-OOD validation for selection.
import json

import torch

from causalcelljepa.dynamics import state_ablation_config, train_dynamics

config_path = "configs/autoencoder_state.yaml"
config, _, _ = state_ablation_config(config_path)
assert torch.cuda.is_available(), "Full autoencoder dynamics training requires CUDA"
_, state, report = train_dynamics(config, torch.device("cuda"), config_path=config_path)
print(json.dumps({"state": state, "report": report}, indent=2))
