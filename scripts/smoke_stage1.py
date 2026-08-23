"""Run a capped real-data CPU check, including a checkpoint/resume boundary."""

import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.training import train_stage1

replogle_config = yaml.safe_load(Path("configs/replogle.yaml").read_text())
stage1_config = yaml.safe_load(Path("configs/stage1.yaml").read_text())
replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
go_manifest = json.loads(Path(stage1_config["resource"]["manifest_path"]).read_text())
smoke_config = deepcopy(stage1_config)
for key in ("batch_size", "num_workers", "output_directory"):
    smoke_config["training"][key] = smoke_config["cpu_smoke"][key]
smoke_config["training"]["checkpoint_every_steps"] = 1
dataset = ReplogleTokenDataset()
_, first_state, first_report = train_stage1(
    dataset,
    replogle_config,
    smoke_config,
    replogle_manifest,
    go_manifest,
    torch.device("cpu"),
    max_steps=1,
)
smoke_config["training"]["resume_from"] = first_report["checkpoint"]
resumed_model, resumed_state, resumed_report = train_stage1(
    dataset,
    replogle_config,
    smoke_config,
    replogle_manifest,
    go_manifest,
    torch.device("cpu"),
    max_steps=smoke_config["cpu_smoke"]["steps"],
)
uninterrupted_config = deepcopy(smoke_config)
uninterrupted_config["training"]["resume_from"] = None
uninterrupted_config["training"]["output_directory"] += "_uninterrupted"
uninterrupted_model, uninterrupted_state, uninterrupted_report = train_stage1(
    dataset,
    replogle_config,
    uninterrupted_config,
    replogle_manifest,
    go_manifest,
    torch.device("cpu"),
    max_steps=smoke_config["cpu_smoke"]["steps"],
)
exact_model_resume = all(
    torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_model.state_dict().items()
)
print(
    json.dumps(
        {
            "device": "cpu",
            "first_global_step": first_state["global_step"],
            "resumed_global_step": resumed_state["global_step"],
            "uninterrupted_global_step": uninterrupted_state["global_step"],
            "exact_model_resume": exact_model_resume,
            "train_cells": resumed_report["train_cells"],
            "validation_cells": resumed_report["validation_cells"],
            "programs": resumed_report["programs"],
            "resumed_elapsed_seconds": resumed_report["elapsed_seconds"],
            "uninterrupted_elapsed_seconds": uninterrupted_report["elapsed_seconds"],
            "checkpoint": resumed_report["checkpoint"],
        },
        indent=2,
    )
)
