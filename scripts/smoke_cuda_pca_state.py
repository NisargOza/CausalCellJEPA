# Require a short fresh-process CUDA checkpoint/resume gate before the full PCA dynamics run.
# This script is intentionally bounded so a broken job cannot consume material GPU credit.
import json
from copy import deepcopy
from pathlib import Path

import torch

from causalcelljepa.dynamics import (
    LatentPopulationDataset,
    state_ablation_config,
    train_dynamics,
    validate_dynamics,
)

config_path = "configs/pca_state.yaml"
config, specification, _ = state_ablation_config(config_path)
settings = specification["cuda_smoke"]
assert torch.cuda.is_available()
smoke = deepcopy(config)
output = Path(settings["output_directory"])
smoke["training"].update(
    {
        "batch_size": settings["batch_size"],
        "checkpoint_every_steps": settings["checkpoint_step"],
        "output_directory": str(output),
    }
)
stop = settings["checkpoint_step"] if not (output / "latest.pt").exists() else settings["steps"]
if stop == settings["steps"]:
    smoke["training"]["resume_from"] = str(output / "latest.pt")
model, state, report = train_dynamics(
    smoke, torch.device("cuda"), max_steps=stop, config_path=config_path
)
validation = None
if stop == settings["steps"]:
    validation = validate_dynamics(
        model,
        LatentPopulationDataset(
            config["inputs"]["latent_cache_path"],
            config["inputs"]["action_cache_path"],
            config["inputs"]["dynamics_manifest_path"],
            "validation",
            config["data"]["population_size"],
            config["seed"],
        ),
        config,
        torch.device("cuda"),
        settings["validation_batches"],
    )
print(json.dumps({"global_step": state["global_step"], "report": report, "validation": validation}, indent=2))
