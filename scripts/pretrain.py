"""Run full Stage 1 pretraining from pinned configurations, with exact resume support."""

import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.training import train_stage1

replogle_config = yaml.safe_load(Path("configs/replogle.yaml").read_text())
stage1_config = yaml.safe_load(Path("configs/stage1.yaml").read_text())
replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
go_manifest = json.loads(Path(stage1_config["resource"]["manifest_path"]).read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, state, report = train_stage1(
    ReplogleTokenDataset(),
    replogle_config,
    stage1_config,
    replogle_manifest,
    go_manifest,
    device,
)
print(json.dumps({"device": str(device), "state": state, "report": report}, indent=2))
