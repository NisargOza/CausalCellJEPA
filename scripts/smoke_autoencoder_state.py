# Run exact CPU checkpoint/resume on the real admitted expression matrix.
# Validation is bounded because this gate tests correctness rather than model selection.
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from causalcelljepa.representations import (
    ExpressionFitDataset,
    train_autoencoder,
    validate_autoencoder,
)
from causalcelljepa.training import _data_loader, stage1_split

config_path = "configs/autoencoder_state.yaml"
config = yaml.safe_load(Path(config_path).read_text())
dataset = ExpressionFitDataset(
    config["inputs"]["expression_cache_path"],
    config["inputs"]["metadata_cache_path"],
    json.loads(Path(config["specification_manifest_path"]).read_text())["leakage"]["fit_roles"],
)
settings = config["cpu_smoke"]
smoke = deepcopy(config)
output = Path(settings["output_directory"])
uninterrupted_output = Path(settings["uninterrupted_output_directory"])
assert not output.exists() and not uninterrupted_output.exists()
smoke["training"].update(
    {
        "batch_size": settings["batch_size"],
        "checkpoint_every_steps": settings["checkpoint_step"],
        "output_directory": str(output),
    }
)
_, first_state, first_report = train_autoencoder(
    dataset,
    smoke,
    torch.device("cpu"),
    max_steps=settings["checkpoint_step"],
    config_path=config_path,
)
smoke["training"]["resume_from"] = first_report["checkpoint"]
resumed_model, resumed_state, _ = train_autoencoder(
    dataset,
    smoke,
    torch.device("cpu"),
    max_steps=settings["steps"],
    config_path=config_path,
)
uninterrupted = deepcopy(smoke)
uninterrupted["training"]["resume_from"] = None
uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
uninterrupted_model, uninterrupted_state, _ = train_autoencoder(
    dataset,
    uninterrupted,
    torch.device("cpu"),
    max_steps=settings["steps"],
    config_path=config_path,
)
exact_model = all(
    torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_model.state_dict().items()
)
assert exact_model and resumed_state == uninterrupted_state
assert (output / "training.jsonl").read_bytes() == (
    uninterrupted_output / "training.jsonl"
).read_bytes()
_, validation_indices = stage1_split(dataset.cell_ids, 0.05, config["seed"])
validation_loader = _data_loader(
    dataset,
    validation_indices[: settings["validation_batches"] * settings["batch_size"]],
    settings["batch_size"],
    0,
    config["seed"] + 1,
)
validation = validate_autoencoder(resumed_model, validation_loader, torch.device("cpu"))
print(
    json.dumps(
        {
            "admitted_cells": len(dataset),
            "first_global_step": first_state["global_step"],
            "resumed_global_step": resumed_state["global_step"],
            "uninterrupted_global_step": uninterrupted_state["global_step"],
            "exact_model_resume": exact_model,
            "exact_training_log_resume": True,
            "validation": validation,
        },
        indent=2,
        sort_keys=True,
    )
)
