# Require a bounded fresh-process CUDA resume and validation gate before full training.
# Each phase exits independently so device checkpoint restoration is actually exercised.
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.representations import (
    ExpressionFitDataset,
    train_autoencoder,
    validate_autoencoder,
)
from causalcelljepa.training import _data_loader, stage1_split


def load_inputs():
    config = yaml.safe_load(Path("configs/autoencoder_state.yaml").read_text())
    roles = json.loads(Path(config["specification_manifest_path"]).read_text())["leakage"][
        "fit_roles"
    ]
    return config, ExpressionFitDataset(
        config["inputs"]["expression_cache_path"],
        config["inputs"]["metadata_cache_path"],
        roles,
    )


def run_phase(phase):
    assert torch.cuda.is_available()
    config, dataset = load_inputs()
    settings = config["cuda_smoke"]
    smoke = deepcopy(config)
    output = Path(settings["output_directory"])
    smoke["training"].update(
        {
            "batch_size": settings["batch_size"],
            "checkpoint_every_steps": settings["checkpoint_step"],
            "output_directory": str(output),
        }
    )
    stop = settings["checkpoint_step"]
    if phase == "resume":
        stop = settings["steps"]
        smoke["training"]["resume_from"] = str(output / "latest.pt")
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_autoencoder(
        dataset,
        smoke,
        torch.device("cuda"),
        max_steps=stop,
        config_path="configs/autoencoder_state.yaml",
    )
    result = {
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if phase == "resume":
        _, validation_indices = stage1_split(dataset.cell_ids, 0.05, config["seed"])
        count = settings["validation_batches"] * settings["batch_size"]
        loader = _data_loader(
            dataset,
            validation_indices[:count],
            settings["batch_size"],
            0,
            config["seed"] + 1,
        )
        result["validation"] = validate_autoencoder(
            model, loader, torch.device("cuda")
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


phase = os.environ.get("CAUSALCELLJEPA_AUTOENCODER_SMOKE_PHASE")
if phase:
    run_phase(phase)
else:
    config, _ = load_inputs()
    output = Path(config["cuda_smoke"]["output_directory"])
    assert not output.exists()
    for child in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment["CAUSALCELLJEPA_AUTOENCODER_SMOKE_PHASE"] = child
        subprocess.run([sys.executable, __file__], check=True, env=environment)
    print((output / "resume_report.json").read_text())
