# Unpaired, batch-matched population sampling and the locked Stage 2 world model.
# Neither cell-line IDs nor pairwise control/outcome correspondences enter the model or loss.
import json
import math
import time
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from geomloss import SamplesLoss
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
)


class LatentPopulationDataset(Dataset):
    """Sample independent outcome/control sets with identical batch multiplicities."""

    def __init__(
        self,
        cache_path,
        action_path,
        manifest_path,
        mode,
        population_size=32,
        seed=0,
        outcome_role=None,
        control_role=None,
        context="K562",
    ):
        self.cache_path, self.mode, self.context = Path(cache_path), mode, context
        self.population_size, self.seed, self.epoch = population_size, seed, 0
        self.manifest = json.loads(Path(manifest_path).read_text())
        with h5py.File(self.cache_path, "r") as cache:
            roles = cache["role"].asstr()[:]
            self.targets = cache["target"].asstr()[:]
            self.batches = cache["source_batch"].asstr()[:]
            contexts = cache["context"].asstr()[:]
        expected_targets = None
        if outcome_role is None:
            outcome_role = self.manifest["conditions"][mode]["role"]
            expected_targets = self.manifest["conditions"][mode]["targets"]
        control_role = control_role or self.manifest["conditions"]["controls"]["role"]
        outcome_indices = np.flatnonzero(roles == outcome_role)
        control_indices = np.flatnonzero(roles == control_role)
        assert set(contexts[outcome_indices]) == set(contexts[control_indices]) == {context}
        self.condition_targets = sorted(set(self.targets[outcome_indices]))
        assert expected_targets is None or len(self.condition_targets) == expected_targets
        self.outcomes = {
            target: outcome_indices[self.targets[outcome_indices] == target]
            for target in self.condition_targets
        }
        self.controls = {
            batch: control_indices[self.batches[control_indices] == batch]
            for batch in sorted(set(self.batches[control_indices]))
        }
        assert all(len(indices) >= population_size for indices in self.outcomes.values())
        action = torch.load(action_path, map_location="cpu", weights_only=True)
        action_index = {target: index for index, target in enumerate(action["targets"])}
        self.action = {
            target: (action["embedding"][action_index[target]], action["known"][action_index[target]])
            for target in self.condition_targets
        }
        normalization = self.manifest["normalization"]
        self.mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
        self.scale = (
            np.asarray(normalization["latent_std"], dtype=np.float32)
            * normalization["dimension_scale"]
        )
        self._cache = None

    def __len__(self):
        return len(self.condition_targets)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample_indices(self, index):
        """Expose deterministic indices for leakage and batch-matching audits."""
        target = self.condition_targets[index]
        generator = np.random.default_rng(
            int.from_bytes(
                sha256(f"{self.seed}\0{self.mode}\0{self.epoch}\0{target}".encode()).digest()[:8],
                "little",
            )
        )
        outcomes = generator.choice(
            self.outcomes[target], self.population_size, replace=False
        )
        controls = np.concatenate(
            [
                generator.choice(self.controls[batch], count, replace=False)
                for batch, count in sorted(Counter(self.batches[outcomes]).items())
            ]
        )
        generator.shuffle(outcomes)
        generator.shuffle(controls)
        return controls, outcomes, target

    def __getitem__(self, index):
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r")
        controls, outcomes, target = self.sample_indices(index)
        populations = []
        for indices in (controls, outcomes):
            order = np.argsort(indices)
            latents = self._cache["latent"][indices[order]][np.argsort(order)]
            populations.append(torch.from_numpy((latents - self.mean) / self.scale))
        action, known = self.action[target]
        return {
            "control": populations[0],
            "perturbed": populations[1],
            "action": action,
            "action_known": known,
            "target": target,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


class ConditionalTransitionBlock(nn.Module):
    """Inject the context-action interaction into attention and feed-forward paths."""

    def __init__(self, dim=256, heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.condition = nn.Linear(dim, dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, states, interaction):
        condition = self.condition(interaction).unsqueeze(1)
        attended, _ = self.attention(
            self.attention_norm(states + condition),
            self.attention_norm(states + condition),
            self.attention_norm(states + condition),
            need_weights=False,
        )
        states = states + attended
        return states + self.ffn(self.ffn_norm(states + condition))


class PopulationDynamics(nn.Module):
    """Permutation-equivariant residual dynamics conditioned on baseline context and action."""

    def __init__(
        self,
        cell_dim=256,
        action_input_dim=320,
        action_dim=256,
        context_blocks=2,
        transition_blocks=3,
        heads=8,
        ffn_dim=1024,
        dropout=0.1,
        context_mode="set_transformer",
    ):
        super().__init__()
        assert context_mode in {"set_transformer", "mean", "none"}
        self.context_mode = context_mode
        context_layer = nn.TransformerEncoderLayer(
            cell_dim,
            heads,
            ffn_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_blocks = nn.TransformerEncoder(
            context_layer,
            context_blocks,
            nn.LayerNorm(cell_dim),
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.randn(1, cell_dim) * 0.02)
        self.pool = nn.MultiheadAttention(cell_dim, heads, dropout=dropout, batch_first=True)
        self.context_output = nn.LayerNorm(cell_dim)
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_input_dim), nn.Linear(action_input_dim, action_dim)
        )
        self.unknown_action = nn.Parameter(torch.randn(action_dim) * 0.02)
        self.interaction = nn.Sequential(
            nn.LayerNorm(4 * cell_dim),
            nn.Linear(4 * cell_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, cell_dim),
        )
        self.transition = nn.ModuleList(
            ConditionalTransitionBlock(cell_dim, heads, ffn_dim, dropout)
            for _ in range(transition_blocks)
        )
        self.delta = nn.Sequential(nn.LayerNorm(cell_dim), nn.Linear(cell_dim, cell_dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, control, action_embedding, action_known):
        if self.context_mode == "set_transformer":
            encoded = self.context_blocks(control)
            query = self.pool_query.unsqueeze(0).expand(control.shape[0], -1, -1)
            context, _ = self.pool(query, encoded, encoded, need_weights=False)
            context = self.context_output(context.squeeze(1))
        elif self.context_mode == "mean":
            context = self.context_output(control.mean(1))
        else:
            context = torch.zeros_like(control[:, 0])
        projected = self.action_projection(action_embedding)
        action = torch.where(
            action_known.unsqueeze(1), projected, self.unknown_action.unsqueeze(0)
        )
        interaction = self.interaction(
            torch.cat((context, action, context * action, torch.abs(context - action)), dim=1)
        )
        states = control
        for block in self.transition:
            states = block(states, interaction)
        return control + self.delta(states)


def dynamics_loss(predicted, observed, control, config, median_distance, null_threshold):
    """Locked unpaired Sinkhorn + MMD + direction + magnitude population objective."""
    blur = config["sinkhorn_blur_ratio"] * median_distance
    sinkhorn = SamplesLoss(
        "sinkhorn",
        p=config["sinkhorn_p"],
        blur=blur,
        debias=config["sinkhorn_debias"],
        backend=config["sinkhorn_backend"],
    )(predicted, observed).mean()
    bandwidth = config["mmd_bandwidth_ratio"] * median_distance
    xx = torch.exp(-torch.cdist(predicted, predicted).square() / (2 * bandwidth**2)).mean((1, 2))
    yy = torch.exp(-torch.cdist(observed, observed).square() / (2 * bandwidth**2)).mean((1, 2))
    xy = torch.exp(-torch.cdist(predicted, observed).square() / (2 * bandwidth**2)).mean((1, 2))
    mmd = (xx + yy - 2 * xy).clamp_min(0).mean()
    true_effect = observed.mean(1) - control.mean(1)
    predicted_effect = predicted.mean(1) - control.mean(1)
    true_magnitude = torch.linalg.vector_norm(true_effect, dim=1)
    predicted_magnitude = torch.linalg.vector_norm(predicted_effect, dim=1)
    active = true_magnitude >= null_threshold
    direction = (
        (
            1
            - F.cosine_similarity(
                predicted_effect[active],
                true_effect[active],
                eps=max(null_threshold, 1e-8),
            )
        ).mean()
        if active.any()
        else predicted.sum() * 0
    )
    magnitude = F.smooth_l1_loss(predicted_magnitude, true_magnitude)
    weights = config["weights"]
    total = (
        weights["sinkhorn"] * sinkhorn
        + weights["mmd"] * mmd
        + weights["direction"] * direction
        + weights["magnitude"] * magnitude
    )
    return {
        "loss": total,
        "sinkhorn": sinkhorn,
        "mmd": mmd,
        "direction": direction,
        "magnitude": magnitude,
        "direction_active": active.float().mean(),
    }


def dynamics_objective(predicted, observed, control, config, statistics):
    """Switch only the training criterion for the locked pseudo-paired comparator."""
    objective = config.get("objective", "unpaired_distribution")
    if objective == "pseudo_paired_mse":
        pointwise = F.mse_loss(predicted, observed)
        return {"loss": pointwise, "pointwise_mse": pointwise}
    assert objective == "unpaired_distribution"
    return dynamics_loss(
        predicted,
        observed,
        control,
        config["loss"],
        statistics["normalization"]["median_training_latent_distance"],
        statistics["direction"]["null_effect_threshold"],
    )


def build_dynamics_model(config):
    """Build the fixed primary model without optional context IDs or stochastic noise."""
    model = config["model"]
    assert model["cell_dim"] == model["action_dim"]
    assert model["predicted_population_size"] == config["data"]["population_size"]
    assert model["cell_line_id"] is False and model["stochastic_noise"] is False
    return PopulationDynamics(
        model["cell_dim"],
        model["action_input_dim"],
        model["action_dim"],
        model["context_blocks"],
        model["transition_blocks"],
        model["heads"],
        model["ffn_dim"],
        model["dropout"],
        model.get("context_mode", "set_transformer"),
    )


def dynamics_ablation_configs(path="configs/ablations.yaml"):
    """Materialize the fixed matched-capacity dynamics ablations from one pinned base."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    base = yaml.safe_load(base_path.read_text())
    configs = {}
    for name, changes in specification["experiments"].items():
        config = deepcopy(base)
        config["model"]["context_mode"] = changes["context_mode"]
        config["loss"]["weights"]["direction"] = changes["direction_weight"]
        config["training"]["output_directory"] = changes["output_directory"]
        config["training"]["resume_from"] = None
        config["ablation"] = {
            "name": name,
            "hypothesis": changes["hypothesis"],
            "base_config_path": str(base_path),
            "base_config_sha256": specification["base_config_sha256"],
        }
        configs[name] = config
    return configs, specification


def learned_target_id_config(path="configs/learned_target_id.yaml"):
    """Materialize the categorical-action comparator from its pinned cache manifest."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    base = yaml.safe_load(base_path.read_text())
    cache_manifest_path = Path(specification["cache_manifest_path"])
    cache_manifest = json.loads(cache_manifest_path.read_text())
    declared = cache_manifest.pop("manifest_sha256")
    assert declared == sha256(
        json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = cache_manifest["artifact"]
    assert artifact["path"] == specification["action_cache_path"]
    assert Path(artifact["path"]).stat().st_size == artifact["bytes"]
    assert file_sha256(artifact["path"]) == artifact["sha256"]
    assert cache_manifest["source"]["config_sha256"] == file_sha256(path)
    assert (
        cache_manifest["source"]["specification_manifest_sha256"]
        == specification["specification_manifest_sha256"]
    )
    config = deepcopy(base)
    config["inputs"].update(
        {
            "action_cache_path": artifact["path"],
            "action_cache_bytes": artifact["bytes"],
            "action_cache_sha256": artifact["sha256"],
            "action_manifest_path": str(cache_manifest_path),
            "action_manifest_sha256": declared,
        }
    )
    config["model"]["action_input_dim"] = artifact["input_dim"]
    config["training"]["output_directory"] = specification["output_directory"]
    config["training"]["resume_from"] = None
    config["ablation"] = {
        "name": "learned_target_id",
        "hypothesis": "structured ESM-2 actions enable unseen-target transfer",
        "base_config_path": str(base_path),
        "base_config_sha256": specification["base_config_sha256"],
        "specification_manifest_sha256": specification["specification_manifest_sha256"],
        "cache_manifest_sha256": declared,
    }
    return config, specification, {**cache_manifest, "manifest_sha256": declared}


def dynamics_provenance(config, config_path="configs/dynamics.yaml"):
    """Bind checkpoints to exact caches, manifests, configuration, code, and runtime."""
    inputs = config["inputs"]
    for kind in ("latent", "action"):
        assert file_sha256(inputs[f"{kind}_cache_path"]) == inputs[f"{kind}_cache_sha256"]
    manifests = {}
    for kind in ("replogle", "action", "dynamics"):
        path = inputs[f"{kind}_manifest_path"]
        payload = json.loads(Path(path).read_text())
        declared = payload.pop("manifest_sha256")
        assert declared == sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifests[kind] = declared
    return {
        "config_sha256": file_sha256(config_path),
        "cache_sha256": {
            "latent": inputs["latent_cache_sha256"],
            "action": inputs["action_cache_sha256"],
        },
        "manifest_sha256": manifests,
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "geomloss": version("geomloss"),
        "git": _git_state(),
    }


def _device_batch(batch, device):
    non_blocking = device.type == "cuda"
    return {
        key: batch[key].to(device, non_blocking=non_blocking)
        for key in ("control", "perturbed", "action", "action_known")
    }


@torch.no_grad()
def validate_dynamics(model, dataset, config, device, max_batches=None):
    """Evaluate only frozen K562 perturbation-OOD validation conditions."""
    was_training = model.training
    model.eval()
    dataset.set_epoch(-1)
    training = config["training"]
    loader = _data_loader(
        dataset,
        range(len(dataset)),
        training["batch_size"],
        training["num_workers"],
        config["seed"] + 1,
        training["pin_memory"] and device.type == "cuda",
        training["persistent_workers"],
        training["prefetch_factor"],
    )
    statistics = dataset.manifest
    totals, conditions = {}, 0
    for index, raw_batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = _device_batch(raw_batch, device)
        predicted = model(batch["control"], batch["action"], batch["action_known"])
        losses = dynamics_objective(
            predicted, batch["perturbed"], batch["control"], config, statistics
        )
        size = len(raw_batch["target"])
        conditions += size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
    model.train(was_training)
    return {name: value / conditions for name, value in totals.items()}


def train_dynamics(
    config,
    device,
    max_steps=None,
    validation_batches=None,
    config_path="configs/dynamics.yaml",
):
    """Train or exactly resume condition-weighted Stage 2 dynamics."""
    training, data = config["training"], config["data"]
    assert training["optimizer"] == "AdamW"
    seed_everything(config["seed"], training["deterministic_algorithms"])
    datasets = {
        mode: LatentPopulationDataset(
            config["inputs"]["latent_cache_path"],
            config["inputs"]["action_cache_path"],
            config["inputs"]["dynamics_manifest_path"],
            mode,
            data["population_size"],
            config["seed"],
        )
        for mode in ("train", "validation")
    }
    model = build_dynamics_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(datasets["train"]) / training["batch_size"])
    total_steps = training["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            total_steps,
            training["warmup_fraction"],
            training["minimum_learning_rate_fraction"],
        ),
    )
    provenance = dynamics_provenance(config, config_path)
    configuration = deepcopy(config)
    output = Path(training["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    latest, best = output / "latest.pt", output / "best.pt"
    state = initial_train_state()
    if training["resume_from"]:
        state, saved_configuration = load_checkpoint(
            training["resume_from"], model, optimizer, scheduler, provenance, device
        )
        saved_configuration["training"]["resume_from"] = None
        current = deepcopy(configuration)
        current["training"]["resume_from"] = None
        assert saved_configuration == current
        if state["complete"]:
            model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
            return model, state, {
                "train_conditions": len(datasets["train"]),
                "validation_conditions": len(datasets["validation"]),
                "elapsed_seconds": 0.0,
                "checkpoint": str(latest),
                "best_checkpoint": str(best),
                "already_complete": True,
            }
    statistics = datasets["train"].manifest
    started = time.perf_counter()
    model.train()
    for epoch in range(state["epoch"], training["epochs"]):
        datasets["train"].set_epoch(epoch)
        order = epoch_order(range(len(datasets["train"])), config["seed"], epoch)
        start = state["batch_in_epoch"] * training["batch_size"]
        loader = _data_loader(
            datasets["train"],
            order[start:],
            training["batch_size"],
            training["num_workers"],
            config["seed"] + 2 + epoch,
            training["pin_memory"] and device.type == "cuda",
            training["persistent_workers"],
            training["prefetch_factor"],
        )
        for offset, raw_batch in enumerate(loader, start=state["batch_in_epoch"]):
            batch = _device_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch["control"], batch["action"], batch["action_known"])
            losses = dynamics_objective(
                predicted, batch["perturbed"], batch["control"], config, statistics
            )
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError(
                    f"Non-finite dynamics loss at optimizer step {state['global_step']}"
                )
            losses["loss"].backward()
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
                    "conditions": len(raw_batch["target"]),
                    **{name: float(value.detach()) for name, value in losses.items()},
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
                    "train_conditions": len(datasets["train"]),
                    "validation_conditions": len(datasets["validation"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint": str(latest),
                }
        validation = validate_dynamics(
            model, datasets["validation"], config, device, validation_batches
        )
        _json_log(
            output / "training.jsonl",
            {"event": "validation", "epoch": epoch, "global_step": state["global_step"], **validation},
        )
        improved = validation["loss"] < state["best_validation_loss"]
        state["best_validation_loss"] = min(state["best_validation_loss"], validation["loss"])
        if improved:
            state["best_validation_epoch"] = epoch
        state["epochs_without_improvement"] = 0 if improved else state["epochs_without_improvement"] + 1
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
        "train_conditions": len(datasets["train"]),
        "validation_conditions": len(datasets["validation"]),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(latest),
        "best_checkpoint": str(best),
    }
