"""Deterministic Stage 1 training, validation, checkpointing, and teacher export."""

import json
import math
import random
import subprocess
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from causalcelljepa.model import CellEncoder, CellJEPA, ema_momentum, jepa_loss, mask_gene_tokens
from causalcelljepa.resources import file_sha256, load_gmt_gene_indices


def stage1_split(cell_ids, validation_fraction, seed):
    """Hash-split only admitted cells into fixed Stage 1 train/validation subsets."""
    threshold = int(validation_fraction * (1 << 256))
    validation = np.asarray(
        [
            int.from_bytes(sha256(f"{seed}\0stage1-validation\0{cell_id}".encode()).digest(), "big")
            < threshold
            for cell_id in cell_ids
        ],
        dtype=bool,
    )
    if not validation.any() or validation.all():
        raise ValueError("Stage 1 split must contain both training and validation cells")
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def stage1_split_report(dataset, validation_fraction, seed, replogle_manifest):
    """Summarize and hash the exact admitted-cell split without storing cell identifiers."""
    cell_ids = [sample[2] for sample in dataset.samples]
    train_indices, validation_indices = stage1_split(cell_ids, validation_fraction, seed)
    splits = {}
    for name, indices in (("train", train_indices), ("validation", validation_indices)):
        selected = [dataset.samples[index] for index in indices]
        ids = sorted(sample[2] for sample in selected)
        splits[name] = {
            "cells": len(selected),
            "cell_ids_sha256": sha256("\n".join(ids).encode()).hexdigest(),
            "context_counts": dict(sorted(Counter(sample[0] for sample in selected).items())),
            "role_counts": dict(sorted(Counter(sample[3] for sample in selected).items())),
        }
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "admission_policy": dataset.config["stage1"]["admission_policy"],
        "admitted_cells": len(dataset),
        "replogle_manifest_sha256": replogle_manifest["manifest_sha256"],
        "hvg_sha256": replogle_manifest["genes"]["hvg_sha256"],
        "splits": splits,
    }


def write_stage1_split_manifest(path, dataset, stage1_config, replogle_manifest):
    payload = stage1_split_report(
        dataset, stage1_config["validation"]["fraction"], stage1_config["seed"], replogle_manifest
    )
    payload["config_sha256"] = file_sha256("configs/stage1.yaml")
    payload["manifest_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def epoch_order(indices, seed, epoch):
    """Return a replayable per-epoch order independent of global RNG state."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    return generator.permutation(np.asarray(indices)).tolist()


def seed_everything(seed, deterministic_algorithms=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def batch_views(
    batch,
    programs,
    epoch,
    seed,
    padding_id,
    device,
    mask_range=(0.30, 0.60),
    biological_probability=0.25,
    minimum_program_genes=3,
):
    """Build deterministic per-cell masks and copy tensor-only views to a device."""
    masked_rows = []
    for row, cell_id in enumerate(batch["cell_id"]):
        generator = torch.Generator().manual_seed(
            int.from_bytes(sha256(f"{seed}\0{epoch}\0{cell_id}".encode()).digest()[:8], "little")
        )
        masked_rows.append(
            mask_gene_tokens(
                batch["gene_ids"][row : row + 1],
                batch["values"][row : row + 1],
                batch["padding_mask"][row : row + 1],
                programs=programs,
                mask_range=mask_range,
                biological_probability=biological_probability,
                minimum_program_genes=minimum_program_genes,
                padding_id=padding_id,
                generator=generator,
            )
        )
    keys = ("gene_ids", "values", "padding_mask")
    student = {key: torch.cat([masked[key] for masked in masked_rows]).to(device) for key in keys}
    teacher = {key: batch[key].to(device) for key in keys}
    return student, teacher


def build_stage1_model(config):
    stage1 = config["stage1"]
    encoder = CellEncoder(
        vocab_size=config["data"]["hvg_count"],
        token_dim=stage1["token_dim"],
        latent_queries=stage1["latent_queries"],
        blocks=stage1["blocks"],
        heads=stage1["heads"],
        ffn_dim=stage1["ffn_dim"],
        dropout=stage1["dropout"],
        cell_dim=stage1["cell_dim"],
    )
    return CellJEPA(encoder, stage1["predictor_hidden"])


def learning_rate_schedule(total_steps, warmup_fraction, minimum_fraction):
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_fraction + (1.0 - minimum_fraction) * cosine

    return multiplier


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _declared_manifest_hash(path):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    actual = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != declared:
        raise ValueError(f"Manifest integrity failure for {path}: {actual} != {declared}")
    return declared


def _git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _runtime_source_hash():
    paths = [
        *Path("src").rglob("*.py"),
        *Path("scripts").glob("*.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    digest = sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_provenance(
    replogle_config_path="configs/replogle.yaml",
    stage1_config_path="configs/stage1.yaml",
    replogle_manifest_path="manifests/replogle_v1.json",
    go_manifest_path="manifests/go_bp_2026-08-05.json",
    stage1_manifest_path="manifests/stage1_v1.json",
):
    go_manifest = json.loads(Path(go_manifest_path).read_text())
    actual_gmt_hash = file_sha256(go_manifest["output"]["gmt_path"])
    if actual_gmt_hash != go_manifest["output"]["gmt_sha256"]:
        raise ValueError("GO BP GMT does not match its frozen manifest")
    return {
        "config_sha256": {
            "replogle": file_sha256(replogle_config_path),
            "stage1": file_sha256(stage1_config_path),
        },
        "manifest_sha256": {
            "replogle": _declared_manifest_hash(replogle_manifest_path),
            "go_bp": _declared_manifest_hash(go_manifest_path),
            "stage1_split": _declared_manifest_hash(stage1_manifest_path),
        },
        "hvg_sha256": json.loads(Path(replogle_manifest_path).read_text())["genes"]["hvg_sha256"],
        "go_bp_gmt_sha256": go_manifest["output"]["gmt_sha256"],
        "runtime_source_sha256": _runtime_source_hash(),
        "git": _git_state(),
    }


def initial_train_state():
    return {
        "epoch": 0,
        "batch_in_epoch": 0,
        "global_step": 0,
        "best_validation_loss": float("inf"),
        "best_validation_epoch": None,
        "epochs_without_improvement": 0,
        "complete": False,
        "completion_reason": None,
    }


def save_checkpoint(path, model, optimizer, scheduler, state, configuration, provenance):
    """Atomically save every state required for exact optimizer-step resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": deepcopy(state),
        "rng": capture_rng_state(),
        "configuration": deepcopy(configuration),
        "provenance": deepcopy(provenance),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path, model, optimizer, scheduler, expected_provenance, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format {payload['format_version']}")
    if payload["provenance"] != expected_provenance:
        raise ValueError("Checkpoint provenance does not match the current frozen inputs/code")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return payload["state"], payload["configuration"]


def _configuration_for_resume(configuration):
    normalized = deepcopy(configuration)
    normalized["stage1"]["training"]["resume_from"] = None
    return normalized


def export_canonical_teacher(path, model, state, replogle_config, provenance):
    """Export only a completed EMA teacher as the frozen canonical state encoder."""
    if not state["complete"]:
        raise ValueError("Canonical teacher export is forbidden before Stage 1 completion")
    path = Path(path)
    payload = {
        "format_version": 1,
        "frozen": True,
        "cell_dim": replogle_config["stage1"]["cell_dim"],
        "encoder_configuration": deepcopy(replogle_config["stage1"]),
        "teacher": model.teacher.state_dict(),
        "training_state": deepcopy(state),
        "provenance": deepcopy(provenance),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _json_log(path, event):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _data_loader(dataset, indices, batch_size, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
    )


@torch.no_grad()
def validate(model, loader, programs, config, epoch, seed, device):
    model.eval()
    totals, cells = {}, 0
    stage1 = config["stage1"]
    for batch in loader:
        student, teacher = batch_views(
            batch,
            programs,
            epoch,
            seed,
            model.online.padding_id,
            device,
            stage1["random_mask_range"],
            stage1["biological_mask_probability"],
            stage1["minimum_program_genes"],
        )
        prediction, target, context = model(student, teacher)
        losses = jepa_loss(
            prediction,
            target,
            context,
            stage1["variance_weight"],
            stage1["covariance_weight"],
        )
        batch_size = len(batch["cell_id"])
        cells += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
    model.train()
    return {name: value / cells for name, value in totals.items()}


def train_stage1(
    dataset,
    replogle_config,
    stage1_config,
    replogle_manifest,
    go_manifest,
    device,
    max_steps=None,
):
    """Train or exactly resume Stage 1; a max_steps cap is reserved for CPU smoke checks."""
    training = stage1_config["training"]
    seed = stage1_config["seed"]
    if training["optimizer"] != "AdamW":
        raise ValueError(f"Unsupported Stage 1 optimizer {training['optimizer']}")
    if replogle_config["stage1"]["teacher_mode"] != "eval":
        raise ValueError("The EMA teacher must remain in eval mode")
    if replogle_config["stage1"]["prediction_loss"] != "normalized_smooth_l1":
        raise ValueError("Unsupported Stage 1 prediction loss")
    seed_everything(seed, training["deterministic_algorithms"])
    train_indices, validation_indices = stage1_split(
        [sample[2] for sample in dataset.samples], stage1_config["validation"]["fraction"], seed
    )
    expected_split = json.loads(Path(stage1_config["validation"]["manifest_path"]).read_text())
    actual_split = stage1_split_report(
        dataset, stage1_config["validation"]["fraction"], seed, replogle_manifest
    )
    if actual_split != {
        key: value
        for key, value in expected_split.items()
        if key not in {"config_sha256", "manifest_sha256"}
    }:
        raise ValueError("Runtime Stage 1 admission/split does not match its frozen manifest")
    programs = load_gmt_gene_indices(
        go_manifest["output"]["gmt_path"], replogle_manifest["genes"]["hvg_gene_names"]
    )
    model = build_stage1_model(replogle_config).to(device)
    parameters = [*model.online.parameters(), *model.predictor.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(train_indices) / training["batch_size"])
    total_steps = training["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            total_steps,
            training["warmup_fraction"],
            training["minimum_learning_rate_fraction"],
        ),
    )
    provenance = build_provenance()
    configuration = {"replogle": deepcopy(replogle_config), "stage1": deepcopy(stage1_config)}
    output = Path(training["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    best = output / "best.pt"
    state = initial_train_state()
    if training["resume_from"]:
        state, saved_configuration = load_checkpoint(
            training["resume_from"], model, optimizer, scheduler, provenance, device
        )
        if _configuration_for_resume(saved_configuration) != _configuration_for_resume(
            configuration
        ):
            raise ValueError("Checkpoint configuration does not match the current run")
    validation_loader = _data_loader(
        dataset,
        validation_indices,
        training["batch_size"],
        training["num_workers"],
        seed + 1,
    )
    started = time.perf_counter()
    model.train()
    for epoch in range(state["epoch"], training["epochs"]):
        order = epoch_order(train_indices, seed, epoch)
        start = state["batch_in_epoch"] * training["batch_size"]
        loader = _data_loader(
            dataset,
            order[start:],
            training["batch_size"],
            training["num_workers"],
            seed + 2 + epoch,
        )
        for offset, batch in enumerate(loader, start=state["batch_in_epoch"]):
            student, teacher = batch_views(
                batch,
                programs,
                epoch,
                seed,
                model.online.padding_id,
                device,
                replogle_config["stage1"]["random_mask_range"],
                replogle_config["stage1"]["biological_mask_probability"],
                replogle_config["stage1"]["minimum_program_genes"],
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, target, context = model(student, teacher)
            losses = jepa_loss(
                prediction,
                target,
                context,
                replogle_config["stage1"]["variance_weight"],
                replogle_config["stage1"]["covariance_weight"],
            )
            losses["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, training["gradient_clip_norm"]
            )
            optimizer.step()
            ema_start, ema_end = replogle_config["stage1"]["ema_momentum"]
            momentum = ema_momentum(state["global_step"], total_steps, start=ema_start, end=ema_end)
            model.update_teacher(momentum)
            scheduler.step()
            state["global_step"] += 1
            state["batch_in_epoch"] = offset + 1
            event = {
                "event": "train_step",
                "epoch": epoch,
                "global_step": state["global_step"],
                "cells": len(batch["cell_id"]),
                "loss": float(losses["loss"].detach()),
                "prediction": float(losses["prediction"].detach()),
                "variance": float(losses["variance"].detach()),
                "covariance": float(losses["covariance"].detach()),
                "gradient_norm": float(gradient_norm),
                "ema_momentum": momentum,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            _json_log(output / "training.jsonl", event)
            if state["global_step"] % training["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    latest, model, optimizer, scheduler, state, configuration, provenance
                )
            if max_steps is not None and state["global_step"] >= max_steps:
                save_checkpoint(
                    latest, model, optimizer, scheduler, state, configuration, provenance
                )
                return (
                    model,
                    state,
                    {
                        "train_cells": len(train_indices),
                        "validation_cells": len(validation_indices),
                        "programs": len(programs),
                        "elapsed_seconds": time.perf_counter() - started,
                        "checkpoint": str(latest),
                    },
                )
        validation = validate(
            model,
            validation_loader,
            programs,
            replogle_config,
            stage1_config["validation"]["mask_epoch"],
            seed,
            device,
        )
        _json_log(
            output / "training.jsonl",
            {
                "event": "validation",
                "epoch": epoch,
                "global_step": state["global_step"],
                **validation,
            },
        )
        improved = validation["loss"] < state["best_validation_loss"]
        state["best_validation_loss"] = min(state["best_validation_loss"], validation["loss"])
        if improved:
            state["best_validation_epoch"] = epoch
        state["epochs_without_improvement"] = (
            0 if improved else state["epochs_without_improvement"] + 1
        )
        state["epoch"], state["batch_in_epoch"] = epoch + 1, 0
        save_checkpoint(latest, model, optimizer, scheduler, state, configuration, provenance)
        if improved:
            save_checkpoint(best, model, optimizer, scheduler, state, configuration, provenance)
        if state["epochs_without_improvement"] >= training["early_stopping_patience"]:
            state["complete"], state["completion_reason"] = True, "early_stopping"
            break
    else:
        state["complete"], state["completion_reason"] = True, "configured_epochs"
    save_checkpoint(latest, model, optimizer, scheduler, state, configuration, provenance)
    best_payload = torch.load(best, map_location=device, weights_only=False)
    best_teacher = {
        key.removeprefix("teacher."): value
        for key, value in best_payload["model"].items()
        if key.startswith("teacher.")
    }
    model.teacher.load_state_dict(best_teacher)
    export_canonical_teacher(
        output / "canonical_ema_teacher.pt", model, state, replogle_config, provenance
    )
    return (
        model,
        state,
        {
            "train_cells": len(train_indices),
            "validation_cells": len(validation_indices),
            "programs": len(programs),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(latest),
        },
    )
