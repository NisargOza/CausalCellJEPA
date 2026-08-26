"""Run a capped real-data CUDA check with a fresh-process checkpoint resume."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.resources import load_gmt_gene_indices
from causalcelljepa.training import _data_loader, stage1_split, train_stage1, validate


def load_inputs():
    replogle = yaml.safe_load(Path("configs/replogle.yaml").read_text())
    stage1 = yaml.safe_load(Path("configs/stage1.yaml").read_text())
    replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
    go_manifest = json.loads(Path(stage1["resource"]["manifest_path"]).read_text())
    smoke = deepcopy(stage1)
    smoke["training"]["batch_size"] = smoke["cuda_smoke"]["batch_size"]
    smoke["training"]["output_directory"] = smoke["cuda_smoke"]["output_directory"]
    smoke["training"]["checkpoint_every_steps"] = smoke["cuda_smoke"]["checkpoint_step"]
    return replogle, smoke, replogle_manifest, go_manifest


def run_phase(phase):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires an available CUDA device")
    replogle, smoke, replogle_manifest, go_manifest = load_inputs()
    output = Path(smoke["training"]["output_directory"])
    dataset = ReplogleTokenDataset()
    if phase == "resume":
        smoke["training"]["resume_from"] = str(output / "latest.pt")
        stop = smoke["cuda_smoke"]["steps"]
    else:
        stop = smoke["cuda_smoke"]["checkpoint_step"]
    torch.cuda.reset_peak_memory_stats()
    model, state, report = train_stage1(
        dataset,
        replogle,
        smoke,
        replogle_manifest,
        go_manifest,
        torch.device("cuda"),
        max_steps=stop,
    )
    result = {
        "phase": phase,
        "global_step": state["global_step"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if phase == "resume":
        train_indices, validation_indices = stage1_split(
            [sample[2] for sample in dataset.samples],
            smoke["validation"]["fraction"],
            smoke["seed"],
        )
        del train_indices
        validation_cells = (
            smoke["cuda_smoke"]["validation_batches"] * smoke["cuda_smoke"]["batch_size"]
        )
        loader = _data_loader(
            dataset,
            validation_indices[:validation_cells],
            smoke["cuda_smoke"]["batch_size"],
            smoke["training"]["num_workers"],
            smoke["seed"] + 1,
            smoke["training"]["pin_memory"],
            smoke["training"]["persistent_workers"],
            smoke["training"]["prefetch_factor"],
        )
        programs = load_gmt_gene_indices(
            go_manifest["output"]["gmt_path"], replogle_manifest["genes"]["hvg_gene_names"]
        )
        validation = validate(
            model,
            loader,
            programs,
            replogle,
            smoke["validation"]["mask_epoch"],
            smoke["seed"],
            torch.device("cuda"),
        )
        resumed_steps = stop - smoke["cuda_smoke"]["checkpoint_step"]
        result.update(
            {
                "resumed_steps": resumed_steps,
                "training_cells_per_second": resumed_steps
                * smoke["cuda_smoke"]["batch_size"]
                / report["elapsed_seconds"],
                "validation_batches": smoke["cuda_smoke"]["validation_batches"],
                "validation": validation,
            }
        )
    (output / f"{phase}_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


phase = os.environ.get("CAUSALCELLJEPA_CUDA_SMOKE_PHASE")
if phase:
    run_phase(phase)
else:
    _, smoke_config, _, _ = load_inputs()
    smoke_output = Path(smoke_config["training"]["output_directory"])
    if (smoke_output / "latest.pt").exists():
        raise RuntimeError(f"Refusing to overwrite existing CUDA smoke at {smoke_output}")
    for child_phase in ("checkpoint", "resume"):
        environment = os.environ.copy()
        environment["CAUSALCELLJEPA_CUDA_SMOKE_PHASE"] = child_phase
        subprocess.run([sys.executable, __file__], check=True, env=environment)
    print((smoke_output / "resume_report.json").read_text())
