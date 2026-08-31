# Generate reporting-only State effects after validation has frozen the checkpoint.
# This script runs inside the pinned official State environment on CUDA.
import json
from pathlib import Path

import torch
import yaml
from state.tx.models.state_transition import StateTransitionPerturbationModel

from causalcelljepa.resources import file_sha256
from causalcelljepa.state_baseline import predict_state_baseline
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/state_baseline.yaml").read_text())
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
checkpoint = Path(config["inputs"]["checkpoint_path"])
assert _git_state()["dirty"] is False, "Full sealed prediction requires a clean commit"
assert torch.cuda.is_available() and file_sha256(checkpoint) == config["inputs"][
    "checkpoint_sha256"
]
model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint, map_location="cpu")
report = predict_state_baseline(config, base, model, device=torch.device("cuda"))
Path(config["prediction"]["report_path"]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(report)
