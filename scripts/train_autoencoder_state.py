# Train the reconstruction representation on the frozen Stage 1 split.
# Full training is CUDA-only and supports exact checkpoint resumption.
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.representations import ExpressionFitDataset, train_autoencoder

config_path = "configs/autoencoder_state.yaml"
config = yaml.safe_load(Path(config_path).read_text())
if resume_from := os.environ.get("CAUSALCELLJEPA_RESUME_FROM"):
    config["training"]["resume_from"] = resume_from
roles = json.loads(Path(config["specification_manifest_path"]).read_text())["leakage"][
    "fit_roles"
]
assert torch.cuda.is_available(), "Full autoencoder pretraining requires CUDA"
torch.cuda.reset_peak_memory_stats()
_, state, report = train_autoencoder(
    ExpressionFitDataset(
        config["inputs"]["expression_cache_path"],
        config["inputs"]["metadata_cache_path"],
        roles,
    ),
    config,
    torch.device("cuda"),
    config_path=config_path,
)
report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
print(json.dumps({"state": state, "report": report}, indent=2))
