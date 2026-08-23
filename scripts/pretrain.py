"""Run full Stage 1 pretraining from pinned configurations, with exact resume support."""

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.training import train_stage1

replogle_config = yaml.safe_load(Path("configs/replogle.yaml").read_text())
stage1_config = yaml.safe_load(Path("configs/stage1.yaml").read_text())
if resume_from := os.environ.get("CAUSALCELLJEPA_RESUME_FROM"):
    stage1_config["training"]["resume_from"] = resume_from
replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
go_manifest = json.loads(Path(stage1_config["resource"]["manifest_path"]).read_text())
if not torch.cuda.is_available():
    raise RuntimeError("Full Stage 1 pretraining requires CUDA; refusing silent CPU fallback")
device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
_, state, report = train_stage1(
    ReplogleTokenDataset(),
    replogle_config,
    stage1_config,
    replogle_manifest,
    go_manifest,
    device,
)
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
