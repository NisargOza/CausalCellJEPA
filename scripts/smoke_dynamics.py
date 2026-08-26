# Run an exact checkpoint/resume gate on real frozen caches using CPU only.
# This precedes both the bounded CUDA smoke and every paid full dynamics job.
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import train_dynamics

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
smoke = deepcopy(config)
smoke["training"].update(
    {
        "batch_size": smoke["cpu_smoke"]["batch_size"],
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "checkpoint_every_steps": smoke["cpu_smoke"]["checkpoint_step"],
        "output_directory": smoke["cpu_smoke"]["output_directory"],
    }
)
output = Path(smoke["training"]["output_directory"])
uninterrupted_output = Path(str(output) + "_uninterrupted")
assert not output.exists() and not uninterrupted_output.exists()
_, first_state, first_report = train_dynamics(
    smoke, torch.device("cpu"), max_steps=smoke["cpu_smoke"]["checkpoint_step"]
)
smoke["training"]["resume_from"] = first_report["checkpoint"]
resumed_model, resumed_state, resumed_report = train_dynamics(
    smoke, torch.device("cpu"), max_steps=smoke["cpu_smoke"]["steps"]
)
uninterrupted = deepcopy(smoke)
uninterrupted["training"]["resume_from"] = None
uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
uninterrupted_model, uninterrupted_state, uninterrupted_report = train_dynamics(
    uninterrupted, torch.device("cpu"), max_steps=smoke["cpu_smoke"]["steps"]
)
exact_model_resume = all(
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
assert exact_model_resume and resumed_events == uninterrupted_events
print(
    json.dumps(
        {
            "device": "cpu",
            "first_global_step": first_state["global_step"],
            "resumed_global_step": resumed_state["global_step"],
            "uninterrupted_global_step": uninterrupted_state["global_step"],
            "exact_model_resume": exact_model_resume,
            "exact_training_log_resume": True,
            "train_conditions": resumed_report["train_conditions"],
            "validation_conditions": resumed_report["validation_conditions"],
            "resumed_elapsed_seconds": resumed_report["elapsed_seconds"],
            "uninterrupted_elapsed_seconds": uninterrupted_report["elapsed_seconds"],
            "checkpoint": resumed_report["checkpoint"],
        },
        indent=2,
    )
)
