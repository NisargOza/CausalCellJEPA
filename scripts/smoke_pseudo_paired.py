# Prove exact CPU checkpoint/resume for the random pseudo-paired pointwise baseline.
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import train_dynamics

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
config["objective"] = config["pseudo_paired"]["objective"]
config["training"].update(
    {
        "batch_size": config["cpu_smoke"]["batch_size"],
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "checkpoint_every_steps": config["cpu_smoke"]["checkpoint_step"],
        "output_directory": config["pseudo_paired"]["cpu_smoke_output_directory"],
    }
)
output = Path(config["training"]["output_directory"])
uninterrupted_output = Path(str(output) + "_uninterrupted")
assert not output.exists() and not uninterrupted_output.exists()
_, first_state, first_report = train_dynamics(
    config, torch.device("cpu"), max_steps=config["cpu_smoke"]["checkpoint_step"]
)
config["training"]["resume_from"] = first_report["checkpoint"]
resumed_model, resumed_state, _ = train_dynamics(
    config, torch.device("cpu"), max_steps=config["cpu_smoke"]["steps"]
)
uninterrupted = deepcopy(config)
uninterrupted["training"]["resume_from"] = None
uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
uninterrupted_model, uninterrupted_state, _ = train_dynamics(
    uninterrupted, torch.device("cpu"), max_steps=config["cpu_smoke"]["steps"]
)
assert resumed_state == uninterrupted_state
assert all(
    torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_model.state_dict().items()
)
resumed_log = (output / "training.jsonl").read_text()
assert resumed_log == (uninterrupted_output / "training.jsonl").read_text()
print(
    json.dumps(
        {
            "objective": config["objective"],
            "first_step": first_state["global_step"],
            "resumed_step": resumed_state["global_step"],
            "exact_model_resume": True,
            "exact_log_resume": True,
        },
        indent=2,
    )
)
