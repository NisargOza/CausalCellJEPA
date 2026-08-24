# Matched PCA and reconstruction state baselines using the frozen expression/split protocol.
# Only representation-permitted cells may enter fitting, optimization, or model selection.
import json
import math
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import (
    _data_loader,
    _git_state,
    _json_log,
    _runtime_environment,
    _runtime_source_hash,
    epoch_order,
    initial_train_state,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    stage1_split,
)


class ExpressionFitDataset(Dataset):
    """Load the admitted dense expression matrix once and retain no context feature."""

    def __init__(self, expression_path, metadata_path, fit_roles):
        with h5py.File(metadata_path, "r") as metadata:
            roles = metadata["role"].asstr()[:]
            rows = np.flatnonzero(np.isin(roles, fit_roles))
            self.cell_ids = metadata["cell_id"].asstr()[rows]
            self.contexts = metadata["context"].asstr()[rows]
            self.roles = roles[rows]
            self.hvg_sha256 = metadata.attrs["hvg_sha256"]
            self.replogle_manifest_sha256 = metadata.attrs["replogle_manifest_sha256"]
        with h5py.File(expression_path, "r") as expression:
            assert expression.attrs["hvg_sha256"] == self.hvg_sha256
            self.expression = torch.from_numpy(expression["expression"][rows])

    def __len__(self):
        return len(self.expression)

    def __getitem__(self, index):
        return self.expression[index]


def expression_split_report(dataset, validation_fraction, seed):
    """Reproduce the frozen Stage 1 split report from the aligned expression cache."""
    train, validation = stage1_split(dataset.cell_ids, validation_fraction, seed)
    splits = {}
    for name, indices in (("train", train), ("validation", validation)):
        splits[name] = {
            "cells": len(indices),
            "cell_ids_sha256": sha256(
                "\n".join(sorted(dataset.cell_ids[indices])).encode()
            ).hexdigest(),
            "context_counts": dict(sorted(Counter(dataset.contexts[indices]).items())),
            "role_counts": dict(sorted(Counter(dataset.roles[indices]).items())),
        }
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "admission_policy": "controls_and_k562_dynamics_train_only",
        "admitted_cells": len(dataset),
        "replogle_manifest_sha256": dataset.replogle_manifest_sha256,
        "hvg_sha256": dataset.hvg_sha256,
        "splits": splits,
    }


class ReconstructionAutoencoder(nn.Module):
    """Symmetric MLP reconstruction baseline with a matched 256-D cell state."""

    def __init__(self, input_dim=3000, hidden_dim=1024, cell_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cell_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, expression):
        return self.encoder(expression)

    def forward(self, expression):
        return self.decoder(self.encode(expression))


def build_autoencoder(config):
    return ReconstructionAutoencoder(**config["model"])


def _manifest_sha256(path):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    assert declared == sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return declared


def autoencoder_provenance(config, config_path="configs/autoencoder_state.yaml"):
    """Bind reconstruction checkpoints to exact caches, split, code, and runtime."""
    for kind in ("expression", "metadata"):
        path = Path(config["inputs"][f"{kind}_cache_path"])
        assert (path.stat().st_size, file_sha256(path)) == (
            config["inputs"][f"{kind}_cache_bytes"],
            config["inputs"][f"{kind}_cache_sha256"],
        )
    assert _manifest_sha256(config["specification_manifest_path"]) == config[
        "specification_manifest_sha256"
    ]
    assert _manifest_sha256(config["stage1_split_manifest_path"]) == config[
        "stage1_split_manifest_sha256"
    ]
    return {
        "config_sha256": file_sha256(config_path),
        "cache_sha256": {
            "expression": config["inputs"]["expression_cache_sha256"],
            "metadata": config["inputs"]["metadata_cache_sha256"],
        },
        "manifest_sha256": {
            "specification": config["specification_manifest_sha256"],
            "stage1_split": config["stage1_split_manifest_sha256"],
        },
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }


@torch.no_grad()
def validate_autoencoder(model, loader, device, maximum_batches=None):
    """Evaluate reconstruction only on the frozen Stage 1 validation split."""
    was_training = model.training
    model.eval()
    loss, cells = 0.0, 0
    for index, expression in enumerate(loader):
        if maximum_batches is not None and index >= maximum_batches:
            break
        expression = expression.to(device, non_blocking=device.type == "cuda")
        value = F.mse_loss(model(expression), expression)
        loss += float(value) * len(expression)
        cells += len(expression)
    model.train(was_training)
    return {"loss": loss / cells, "cells": cells}


def train_autoencoder(
    dataset,
    config,
    device,
    max_steps=None,
    validation_batches=None,
    config_path="configs/autoencoder_state.yaml",
):
    """Train or exactly resume the matched reconstruction representation baseline."""
    training, seed = config["training"], config["seed"]
    assert training["optimizer"] == "AdamW"
    seed_everything(seed, training["deterministic_algorithms"])
    train_indices, validation_indices = stage1_split(dataset.cell_ids, 0.05, seed)
    expected = json.loads(Path(config["stage1_split_manifest_path"]).read_text())
    actual = expression_split_report(dataset, expected["validation_fraction"], seed)
    assert actual == {
        key: value for key, value in expected.items() if key not in {"config_sha256", "manifest_sha256"}
    }
    model = build_autoencoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(train_indices) / training["batch_size"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            training["epochs"] * steps_per_epoch,
            training["warmup_fraction"],
            training["minimum_learning_rate_fraction"],
        ),
    )
    provenance = autoencoder_provenance(config, config_path)
    configuration = deepcopy(config)
    configuration["training"]["resume_from"] = None
    output = Path(training["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    latest, best = output / "latest.pt", output / "best.pt"
    state = initial_train_state()
    if training["resume_from"]:
        state, saved = load_checkpoint(
            training["resume_from"], model, optimizer, scheduler, provenance, device
        )
        current = deepcopy(configuration)
        assert saved == current
        if state["complete"]:
            model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
            return model, state, {"checkpoint": str(latest), "best_checkpoint": str(best)}
    validation_loader = _data_loader(
        dataset,
        validation_indices,
        training["batch_size"],
        training["num_workers"],
        seed + 1,
        training["pin_memory"] and device.type == "cuda",
        training["persistent_workers"],
        training["prefetch_factor"],
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
            training["pin_memory"] and device.type == "cuda",
            training["persistent_workers"],
            training["prefetch_factor"],
        )
        for offset, expression in enumerate(loader, start=state["batch_in_epoch"]):
            expression = expression.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(expression), expression)
            assert torch.isfinite(loss)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            state["global_step"] += 1
            state["batch_in_epoch"] = offset + 1
            _json_log(
                output / "training.jsonl",
                {
                    "event": "train_step",
                    "epoch": epoch,
                    "global_step": state["global_step"],
                    "cells": len(expression),
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                },
            )
            if state["global_step"] % training["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    latest, model, optimizer, scheduler, state, configuration, provenance
                )
            if max_steps is not None and state["global_step"] >= max_steps:
                save_checkpoint(
                    latest, model, optimizer, scheduler, state, configuration, provenance
                )
                return model, state, {
                    "train_cells": len(train_indices),
                    "validation_cells": len(validation_indices),
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": str(latest),
                }
        validation = validate_autoencoder(
            model, validation_loader, device, validation_batches
        )
        _json_log(
            output / "training.jsonl",
            {"event": "validation", "epoch": epoch, "global_step": state["global_step"], **validation},
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
    model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
    return model, state, {
        "train_cells": len(train_indices),
        "validation_cells": len(validation_indices),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(latest),
        "best_checkpoint": str(best),
    }
def fit_pca_state(expression, dimensions, oversampling, power_iterations, seed):
    """Fit deterministic randomized PCA and orient every component canonically."""
    matrix = torch.from_numpy(np.asarray(expression, dtype=np.float32))
    rank = dimensions + oversampling
    assert matrix.ndim == 2 and rank <= min(matrix.shape)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, singular_values, vectors = torch.pca_lowrank(
            matrix, q=rank, center=True, niter=power_iterations
        )
    components = vectors[:, :dimensions].T.contiguous()
    anchors = components.abs().argmax(1)
    signs = torch.sign(components[torch.arange(dimensions), anchors])
    components *= signs[:, None]
    return (
        matrix.mean(0).numpy(),
        components.numpy(),
        singular_values[:dimensions].numpy(),
    )


def project_pca_state(expression, mean, components):
    """Project normalized expression without changing the fitted PCA state space."""
    values = np.asarray(expression, dtype=np.float32)
    return (values - mean) @ components.T
