# Run exact CPU checkpoint/resume gates for every fixed dynamics mechanism ablation.
# Passing all gates is required before the three experiments share one paid GPU instance.
import json
from copy import deepcopy
from pathlib import Path

import torch

from causalcelljepa.dynamics import dynamics_ablation_configs, train_dynamics

configs, specification = dynamics_ablation_configs()
reports = {}
for name, config in configs.items():
    smoke = deepcopy(config)
    settings = specification["cpu_smoke"]
    output = Path(settings["output_root"]) / name
    uninterrupted_output = Path(str(output) + "_uninterrupted")
    assert not output.exists() and not uninterrupted_output.exists()
    smoke["training"].update(
        {
            "batch_size": settings["batch_size"],
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "checkpoint_every_steps": settings["checkpoint_step"],
            "output_directory": str(output),
        }
    )
    _, first_state, first_report = train_dynamics(
        smoke,
        torch.device("cpu"),
        max_steps=settings["checkpoint_step"],
        config_path="configs/ablations.yaml",
    )
    smoke["training"]["resume_from"] = first_report["checkpoint"]
    resumed_model, resumed_state, _ = train_dynamics(
        smoke,
        torch.device("cpu"),
        max_steps=settings["steps"],
        config_path="configs/ablations.yaml",
    )
    uninterrupted = deepcopy(smoke)
    uninterrupted["training"]["resume_from"] = None
    uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
    uninterrupted_model, uninterrupted_state, _ = train_dynamics(
        uninterrupted,
        torch.device("cpu"),
        max_steps=settings["steps"],
        config_path="configs/ablations.yaml",
    )
    exact_model = all(
        torch.equal(resumed_model.state_dict()[key], value)
        for key, value in uninterrupted_model.state_dict().items()
    )
    resumed_log = (output / "training.jsonl").read_text()
    uninterrupted_log = (uninterrupted_output / "training.jsonl").read_text()
    assert exact_model and resumed_log == uninterrupted_log
    epoch_config = deepcopy(config)
    epoch_output = Path(settings["epoch_output_root"]) / name
    assert not epoch_output.exists()
    epoch_config["training"].update(
        {
            "batch_size": settings["epoch_batch_size"],
            "epochs": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "output_directory": str(epoch_output),
        }
    )
    _, epoch_state, epoch_report = train_dynamics(
        epoch_config, torch.device("cpu"), config_path="configs/ablations.yaml"
    )
    assert epoch_state["complete"] and epoch_state["completion_reason"] == "configured_epochs"
    assert Path(epoch_report["best_checkpoint"]).exists()
    reports[name] = {
        "context_mode": config["model"]["context_mode"],
        "direction_weight": config["loss"]["weights"]["direction"],
        "first_global_step": first_state["global_step"],
        "resumed_global_step": resumed_state["global_step"],
        "uninterrupted_global_step": uninterrupted_state["global_step"],
        "exact_model_resume": exact_model,
        "exact_training_log_resume": True,
        "full_epoch_global_steps": epoch_state["global_step"],
        "full_epoch_validation_records": sum(
            json.loads(line)["event"] == "validation"
            for line in (epoch_output / "training.jsonl").read_text().splitlines()
        ),
    }
print(json.dumps(reports, indent=2, sort_keys=True))
