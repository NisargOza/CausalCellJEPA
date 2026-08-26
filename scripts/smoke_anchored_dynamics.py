# Verify exact CPU checkpoint/resume behavior for every anchored candidate.
import json
from copy import deepcopy
from pathlib import Path

import torch

from causalcelljepa.dynamics import anchored_dynamics_configs, train_dynamics

configs, specification = anchored_dynamics_configs()
reports = {}
for name, config in configs.items():
    smoke = deepcopy(config)
    root = Path(specification["cpu_smoke"]["output_directory"])
    output, uninterrupted_output = root / name, root / f"{name}_uninterrupted"
    assert not output.exists() and not uninterrupted_output.exists()
    smoke["training"].update(
        {
            "batch_size": specification["cpu_smoke"]["batch_size"],
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "checkpoint_every_steps": specification["cpu_smoke"]["checkpoint_step"],
            "output_directory": str(output),
        }
    )
    _, first, first_report = train_dynamics(
        smoke,
        torch.device("cpu"),
        max_steps=specification["cpu_smoke"]["checkpoint_step"],
        config_path="configs/anchored_dynamics.yaml",
    )
    smoke["training"]["resume_from"] = first_report["checkpoint"]
    resumed_model, resumed, resumed_report = train_dynamics(
        smoke,
        torch.device("cpu"),
        max_steps=specification["cpu_smoke"]["steps"],
        config_path="configs/anchored_dynamics.yaml",
    )
    uninterrupted = deepcopy(smoke)
    uninterrupted["training"]["resume_from"] = None
    uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
    uninterrupted_model, uninterrupted_state, _ = train_dynamics(
        uninterrupted,
        torch.device("cpu"),
        max_steps=specification["cpu_smoke"]["steps"],
        config_path="configs/anchored_dynamics.yaml",
    )
    exact_model = all(
        torch.equal(resumed_model.state_dict()[key], value)
        for key, value in uninterrupted_model.state_dict().items()
    )
    resumed_events = [
        json.loads(line)
        for line in (output / "training.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "train_step"
    ]
    uninterrupted_events = [
        json.loads(line)
        for line in (uninterrupted_output / "training.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "train_step"
    ]
    assert exact_model and resumed_events == uninterrupted_events
    reports[name] = {
        "first_global_step": first["global_step"],
        "resumed_global_step": resumed["global_step"],
        "uninterrupted_global_step": uninterrupted_state["global_step"],
        "exact_model_resume": exact_model,
        "exact_training_log_resume": True,
        "train_conditions": resumed_report["train_conditions"],
        "validation_conditions": resumed_report["validation_conditions"],
    }
print(json.dumps(reports, indent=2, sort_keys=True))
