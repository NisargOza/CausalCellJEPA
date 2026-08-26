# Run exact CPU resume and full-epoch gates after the frozen PCA state cache is materialized.
# The downstream model and data roles remain identical to the canonical JEPA dynamics run.
import json
from copy import deepcopy
from pathlib import Path

import torch

from causalcelljepa.dynamics import state_ablation_config, train_dynamics

config_path = "configs/pca_state.yaml"
config, specification, _ = state_ablation_config(config_path)
settings = specification["cpu_smoke"]
smoke = deepcopy(config)
output = Path(settings["output_directory"])
uninterrupted_output = Path(settings["uninterrupted_output_directory"])
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
    config_path=config_path,
)
smoke["training"]["resume_from"] = first_report["checkpoint"]
resumed_model, resumed_state, _ = train_dynamics(
    smoke, torch.device("cpu"), max_steps=settings["steps"], config_path=config_path
)
uninterrupted = deepcopy(smoke)
uninterrupted["training"]["resume_from"] = None
uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
uninterrupted_model, uninterrupted_state, _ = train_dynamics(
    uninterrupted, torch.device("cpu"), max_steps=settings["steps"], config_path=config_path
)
exact_model = all(
    torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_model.state_dict().items()
)
assert exact_model and resumed_state == uninterrupted_state
assert (output / "training.jsonl").read_bytes() == (
    uninterrupted_output / "training.jsonl"
).read_bytes()

epoch_config = deepcopy(config)
epoch_output = Path(settings["epoch_output_directory"])
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
    epoch_config, torch.device("cpu"), config_path=config_path
)
assert epoch_state["complete"] and epoch_state["completion_reason"] == "configured_epochs"
assert Path(epoch_report["best_checkpoint"]).exists()
print(
    json.dumps(
        {
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
        },
        indent=2,
        sort_keys=True,
    )
)
