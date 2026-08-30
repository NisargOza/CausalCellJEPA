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


class ModalityAttentiveActionProjection(nn.Module):
    """Project frozen action teachers and fuse them with sample-specific attention."""

    def __init__(
        self,
        modality_dims,
        output_dim,
        modality_dropout=0.0,
        modality_availability=False,
    ):
        super().__init__()
        assert sum(modality_dims) > 0 and 0 <= modality_dropout < 1
        self.modality_dims = tuple(modality_dims)
        self.modality_dropout = modality_dropout
        self.modality_availability = modality_availability
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
            for width in modality_dims
        )
        self.query = nn.Parameter(torch.randn(output_dim) * 0.02)
        self.output = nn.LayerNorm(output_dim)

    def projected_and_weights(self, action):
        feature_width = sum(self.modality_dims)
        expected_width = feature_width + (
            len(self.modality_dims) if self.modality_availability else 0
        )
        assert action.shape[1] == expected_width
        blocks = action[:, :feature_width].split(self.modality_dims, 1)
        projected = torch.stack(
            [
                projector(block)
                for projector, block in zip(self.projectors, blocks, strict=True)
            ],
            1,
        )
        scores = (torch.tanh(projected) * self.query).sum(2) / math.sqrt(projected.shape[2])
        visible = (
            action[:, feature_width:].bool()
            if self.modality_availability
            else torch.ones_like(scores, dtype=torch.bool)
        )
        if self.training and self.modality_dropout:
            visible &= torch.rand_like(scores) >= self.modality_dropout
        missing = ~visible.any(1)
        if missing.any():
            available = (
                action[:, feature_width:].bool()
                if self.modality_availability
                else torch.ones_like(visible)
            )
            fallback = available.float().argmax(1)
            visible[missing, fallback[missing]] = True
        weights = scores.masked_fill(~visible, -torch.inf).softmax(1)
        return projected, weights

    def forward(self, action):
        projected, weights = self.projected_and_weights(action)
        return self.output((weights.unsqueeze(2) * projected).sum(1))


class ContextConditionedModalityProjection(nn.Module):
    """Fuse available action teachers using an identity-free control-state query."""

    def __init__(self, modality_dims, output_dim, modality_dropout=0.0):
        super().__init__()
        assert sum(modality_dims) > 0 and 0 <= modality_dropout < 1
        self.modality_dims = tuple(modality_dims)
        self.modality_dropout = modality_dropout
        self.projectors = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
            for width in modality_dims
        )
        self.query = nn.Parameter(torch.randn(output_dim) * 0.02)
        self.context_query = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, output_dim))
        nn.init.zeros_(self.context_query[-1].weight)
        nn.init.zeros_(self.context_query[-1].bias)
        self.output = nn.LayerNorm(output_dim)

    def projected_and_weights(self, action, context):
        feature_width = sum(self.modality_dims)
        assert action.shape[1] == feature_width + len(self.modality_dims)
        blocks = action[:, :feature_width].split(self.modality_dims, 1)
        availability = action[:, feature_width:].bool()
        projected = torch.stack(
            [
                projector(block)
                for projector, block in zip(self.projectors, blocks, strict=True)
            ],
            1,
        )
        query = self.query.unsqueeze(0) + self.context_query(context)
        scores = (torch.tanh(projected) * query.unsqueeze(1)).sum(2) / math.sqrt(
            projected.shape[2]
        )
        visible = availability.clone()
        if self.training and self.modality_dropout:
            visible &= torch.rand_like(scores) >= self.modality_dropout
        missing = ~visible.any(1)
        if missing.any():
            fallback = availability.float().argmax(1)
            visible[missing, fallback[missing]] = True
        weights = scores.masked_fill(~visible, -torch.inf).softmax(1)
        return projected, weights

    def forward(self, action, context):
        projected, weights = self.projected_and_weights(action, context)
        return self.output((weights.unsqueeze(2) * projected).sum(1))


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
        action_modalities=None,
        action_modality_dropout=0.0,
        action_context_conditioned=False,
        action_modality_availability=False,
    ):
        super().__init__()
        assert context_mode in {"set_transformer", "mean", "none"}
        availability_width = len(action_modalities or ()) if action_modality_availability else 0
        assert action_modalities is None or sum(action_modalities) + availability_width == action_input_dim
        assert not action_context_conditioned or action_modality_availability
        self.context_mode = context_mode
        self.action_context_conditioned = action_context_conditioned
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
        if action_context_conditioned:
            self.action_projection = ContextConditionedModalityProjection(
                action_modalities, action_dim, action_modality_dropout
            )
        elif action_modalities:
            self.action_projection = ModalityAttentiveActionProjection(
                action_modalities,
                action_dim,
                action_modality_dropout,
                action_modality_availability,
            )
        else:
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
        projected = (
            self.action_projection(action_embedding, context)
            if self.action_context_conditioned
            else self.action_projection(action_embedding)
        )
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


class FrozenLowRankEffectAnchor(nn.Module):
    """Apply a frozen low-rank ESM-to-latent perturbation-effect map."""

    def __init__(self, checkpoint):
        super().__init__()
        assert checkpoint["format_version"] == 1
        assert checkpoint["architecture"] in {
            "esm2_low_rank_latent_effect_ridge",
            "multiteacher_low_rank_latent_effect_ridge",
        }
        for name in ("x_mean", "x_std", "y_mean", "components", "weights"):
            value = checkpoint[name].detach().float()
            assert torch.isfinite(value).all()
            self.register_buffer(name, value)
        input_indices = checkpoint.get("input_indices", torch.arange(len(self.x_mean)))
        input_indices = input_indices.detach().long()
        assert input_indices.ndim == 1 and len(input_indices) == len(self.x_mean)
        assert len(input_indices.unique()) == len(input_indices) and int(input_indices.min()) >= 0
        self.register_buffer("input_indices", input_indices)
        assert self.weights.shape == (len(self.x_mean), len(self.components))
        assert self.components.shape[1] == len(self.y_mean)

    def forward(self, action_embedding, action_known):
        selected = action_embedding.index_select(1, self.input_indices)
        standardized = (selected - self.x_mean) / self.x_std
        predicted = standardized @ self.weights @ self.components + self.y_mean
        return torch.where(action_known.unsqueeze(1), predicted, self.y_mean.unsqueeze(0))


class AnchoredPopulationDynamics(PopulationDynamics):
    """Separate a transferable mean action effect from population heterogeneity."""

    def __init__(
        self,
        *args,
        effect_anchor,
        anchor_gain_max=1.0,
        mean_residual_max_ratio=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert anchor_gain_max >= 1
        assert 0 <= mean_residual_max_ratio <= 1
        self.effect_anchor = FrozenLowRankEffectAnchor(effect_anchor)
        self.anchor_gain_max = float(anchor_gain_max)
        self.mean_residual_max_ratio = float(mean_residual_max_ratio)
        cell_dim = self.effect_anchor.y_mean.numel()
        ffn_dim = self.interaction[1].out_features
        dropout = self.transition[0].ffn[2].p
        self.mean_residual = nn.Sequential(
            nn.LayerNorm(cell_dim),
            nn.Linear(cell_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, cell_dim),
        )
        nn.init.zeros_(self.mean_residual[-1].weight)
        nn.init.zeros_(self.mean_residual[-1].bias)
        self.anchor_gain = nn.Sequential(nn.LayerNorm(cell_dim), nn.Linear(cell_dim, 1))
        nn.init.zeros_(self.anchor_gain[-1].weight)
        nn.init.zeros_(self.anchor_gain[-1].bias)
        self.minimum_anchor_norm = float(
            effect_anchor["report"]["training_null_effect_threshold"]
        )
        self.residual_gate_threshold = None

    def configure_residual_gate(self, checkpoint):
        """Attach a source-control-calibrated confidence gate without changing weights."""
        assert checkpoint["format_version"] == 1
        assert checkpoint["architecture"] == "control_population_residual_gate"
        for name in ("center", "scale"):
            value = checkpoint[name].detach().float()
            assert value.shape == self.effect_anchor.y_mean.shape
            self.register_buffer(f"residual_gate_{name}", value, persistent=False)
        self.residual_gate_threshold = float(checkpoint["threshold"])
        self.residual_gate_temperature = float(checkpoint["temperature"])
        assert self.residual_gate_temperature > 0

    def residual_gate_confidence(self, control):
        """Return one smooth residual-retention confidence per control population."""
        assert self.residual_gate_threshold is not None
        score = ((control.mean(1) - self.residual_gate_center) / self.residual_gate_scale).square().mean(1)
        excess = torch.relu(score - self.residual_gate_threshold)
        return torch.exp(-excess / self.residual_gate_temperature)

    def _scaled_anchor(self, action, anchor):
        if self.anchor_gain_max == 1:
            return anchor
        log_limit = math.log(self.anchor_gain_max)
        gain = torch.exp(torch.tanh(self.anchor_gain(action)) * log_limit)
        return anchor * gain

    def _bounded_mean_residual(self, action, anchor):
        if self.mean_residual_max_ratio == 0:
            return torch.zeros_like(anchor)
        residual = self.mean_residual(action)
        residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        anchor_norm = torch.linalg.vector_norm(anchor, dim=1, keepdim=True).clamp_min(
            self.minimum_anchor_norm
        )
        maximum = self.mean_residual_max_ratio * anchor_norm
        scale = torch.minimum(torch.ones_like(residual_norm), maximum / residual_norm.clamp_min(1e-12))
        return residual * scale

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
        projected = (
            self.action_projection(action_embedding, context)
            if self.action_context_conditioned
            else self.action_projection(action_embedding)
        )
        action = torch.where(
            action_known.unsqueeze(1), projected, self.unknown_action.unsqueeze(0)
        )
        interaction = self.interaction(
            torch.cat((context, action, context * action, torch.abs(context - action)), dim=1)
        )
        states = control
        for block in self.transition:
            states = block(states, interaction)
        population_residual = self.delta(states)
        population_residual = population_residual - population_residual.mean(1, keepdim=True)
        if self.residual_gate_threshold is not None:
            population_residual *= self.residual_gate_confidence(control)[:, None, None]
        anchor = self._scaled_anchor(
            action, self.effect_anchor(action_embedding, action_known)
        )
        mean_residual = self._bounded_mean_residual(action, anchor)
        return control + anchor.unsqueeze(1) + mean_residual.unsqueeze(1) + population_residual


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
    """Build the primary model or an explicitly declared post-primary revision."""
    model = config["model"]
    assert model["cell_dim"] == model["action_dim"]
    assert model["predicted_population_size"] == config["data"]["population_size"]
    assert model["cell_line_id"] is False and model["stochastic_noise"] is False
    arguments = (
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
    fusion = {
        "action_modalities": model.get("action_modalities"),
        "action_modality_dropout": model.get("action_modality_dropout", 0.0),
        "action_context_conditioned": model.get("action_context_conditioned", False),
        "action_modality_availability": model.get("action_modality_availability", False),
    }
    architecture = model.get("architecture", "population_dynamics")
    if architecture == "population_dynamics":
        return PopulationDynamics(*arguments, **fusion)
    assert architecture == "anchored_decomposed_population_dynamics"
    anchor = config["effect_anchor"]
    path = Path(anchor["output_path"])
    assert (path.stat().st_size, file_sha256(path)) == (
        anchor["output_bytes"],
        anchor["output_sha256"],
    )
    manifest = json.loads(Path(anchor["manifest_path"]).read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == anchor["manifest_sha256"] == sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["artifact"]["sha256"] == anchor["output_sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return AnchoredPopulationDynamics(
        *arguments,
        effect_anchor=checkpoint,
        anchor_gain_max=model["anchor_gain_max"],
        mean_residual_max_ratio=model["mean_residual_max_ratio"],
        **fusion,
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


def dynamics_replication_configs(path="configs/stage2_replication.yaml"):
    """Vary only Stage 2 stochastic seeds while retaining the frozen primary protocol."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    base = yaml.safe_load(base_path.read_text())
    split = json.loads(Path(base["inputs"]["replogle_manifest_path"]).read_text())["targets"][
        "split"
    ]
    assert split["seed"] == base["seed"] == specification["target_split_seed"]
    configs = {}
    for entry in specification["model_seeds"]:
        seed = entry["seed"]
        assert seed != specification["target_split_seed"] and seed not in configs
        config = deepcopy(base)
        config["seed"] = seed
        config["training"]["output_directory"] = entry["output_directory"]
        config["training"]["resume_from"] = None
        config["replication"] = {
            "model_and_sampling_seed": seed,
            "target_split_seed": specification["target_split_seed"],
            "base_config_path": str(base_path),
            "base_config_sha256": specification["base_config_sha256"],
        }
        configs[seed] = config
    return configs, specification


def _normalized_role_effects(base, roles):
    """Compute batch-matched latent effects for explicitly named K562 roles."""
    inputs = base["inputs"]
    manifest = json.loads(Path(inputs["dynamics_manifest_path"]).read_text())
    normalization = manifest["normalization"]
    mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
    scale = np.asarray(normalization["latent_std"], dtype=np.float32) * normalization[
        "dimension_scale"
    ]
    with h5py.File(inputs["latent_cache_path"], "r") as cache:
        cache_roles = cache["role"].asstr()[:]
        targets = cache["target"].asstr()[:]
        batches = cache["source_batch"].asstr()[:]
        contexts = cache["context"].asstr()[:]
        control_indices = np.flatnonzero(cache_roles == base["data"]["control_role"])
        assert set(contexts[control_indices]) == {base["data"]["context"]} == {"K562"}
        control_means = {
            batch: ((cache["latent"][control_indices[batches[control_indices] == batch]] - mean) / scale).mean(0)
            for batch in sorted(set(batches[control_indices]))
        }
        effects = {}
        target_ids = {}
        for role in roles:
            role_indices = np.flatnonzero(cache_roles == role)
            assert len(role_indices) and set(contexts[role_indices]) == {"K562"}
            target_ids[role] = sorted(set(targets[role_indices]))
            effects[role] = {}
            for target in target_ids[role]:
                selected = role_indices[targets[role_indices] == target]
                outcome = ((cache["latent"][selected] - mean) / scale).mean(0)
                counts = Counter(batches[selected])
                matched_control = sum(
                    count * control_means[batch] for batch, count in counts.items()
                ) / len(selected)
                effects[role][target] = (outcome - matched_control).astype(np.float32)
    return effects, target_ids, manifest


def _action_overridden_config(base, specification):
    """Replace only the frozen action input from a self-hashed cache manifest."""
    if "action_manifest_path" not in specification:
        return base
    manifest = json.loads(Path(specification["action_manifest_path"]).read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == specification["action_manifest_sha256"] == sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = manifest["artifact"]
    assert (Path(artifact["path"]).stat().st_size, file_sha256(artifact["path"])) == (
        artifact["bytes"], artifact["sha256"]
    )
    base["inputs"].update(
        {
            "action_cache_path": artifact["path"],
            "action_cache_bytes": artifact["bytes"],
            "action_cache_sha256": artifact["sha256"],
            "action_manifest_path": specification["action_manifest_path"],
            "action_manifest_sha256": declared,
        }
    )
    base["model"]["action_input_dim"] = artifact["input_dim"]
    base["model"]["action_modalities"] = artifact["modality_dims"]
    base["model"]["action_modality_availability"] = artifact.get(
        "modality_availability", False
    )
    return base


def effect_anchor_input_indices(action, anchor):
    """Select declared action modalities and their availability bits for an anchor."""
    modalities = action.get("modalities")
    modality_dims = action.get("modality_dims")
    selected_names = anchor.get("input_modalities")
    if selected_names is None:
        return torch.arange(action["embedding"].shape[1])
    assert modalities is not None and modality_dims is not None
    assert len(modalities) == len(modality_dims)
    assert len(selected_names) == len(set(selected_names))
    selected = set(selected_names)
    assert selected == selected.intersection(modalities)
    feature_width = sum(modality_dims)
    availability_width = len(modality_dims) if action.get("modality_availability", False) else 0
    assert action["embedding"].shape[1] == feature_width + availability_width
    indices, offset = [], 0
    selected_positions = []
    for position, (name, width) in enumerate(zip(modalities, modality_dims, strict=True)):
        if name in selected:
            indices.extend(range(offset, offset + width))
            selected_positions.append(position)
        offset += width
    if availability_width:
        indices.extend(feature_width + position for position in selected_positions)
    return torch.tensor(indices, dtype=torch.long)


def prepare_effect_anchor(path="configs/anchored_dynamics.yaml"):
    """Fit the frozen ESM-to-latent anchor without test or RPE1 outcomes."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    base = _action_overridden_config(yaml.safe_load(base_path.read_text()), specification)
    anchor = specification["effect_anchor"]
    roles = (anchor["fit_outcome_role"], anchor["selection_outcome_role"])
    assert roles == ("dynamics_train", "perturbation_ood_validation")
    effects, targets, dynamics_manifest = _normalized_role_effects(base, roles)
    action = torch.load(base["inputs"]["action_cache_path"], map_location="cpu", weights_only=True)
    input_indices = effect_anchor_input_indices(action, anchor)
    action_map = {
        target: (
            action["embedding"][index].index_select(0, input_indices).numpy(),
            bool(action["known"][index]),
        )
        for index, target in enumerate(action["targets"])
    }
    train = effects[roles[0]]
    known_targets = [target for target in targets[roles[0]] if action_map[target][1]]
    x = np.stack([action_map[target][0] for target in known_targets]).astype(np.float64)
    y = np.stack([train[target] for target in known_targets]).astype(np.float64)
    x_mean, x_std = x.mean(0), x.std(0).clip(1e-8)
    y_mean = y.mean(0)
    _, _, components = np.linalg.svd(y - y_mean, full_matrices=False)
    components = components[: min(anchor["rank"], len(known_targets))]
    standardized = (x - x_mean) / x_std
    scores = (y - y_mean) @ components.T
    gram, cross = standardized.T @ standardized, standardized.T @ scores
    validation = effects[roles[1]]
    validation_targets = targets[roles[1]]
    validation_x = np.stack([action_map[target][0] for target in validation_targets])
    validation_y = np.stack([validation[target] for target in validation_targets])
    candidates = []
    for alpha in anchor["ridge_candidates"]:
        weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        prediction = ((validation_x - x_mean) / x_std) @ weights @ components + y_mean
        correlations = [
            float(np.corrcoef(predicted, observed)[0, 1])
            for predicted, observed in zip(prediction, validation_y)
        ]
        magnitude_error = np.abs(
            np.linalg.norm(prediction, axis=1) - np.linalg.norm(validation_y, axis=1)
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "mse": float(np.mean(np.square(prediction - validation_y))),
                "mean_effect_pearson": float(np.mean(correlations)),
                "mean_magnitude_absolute_error": float(np.mean(magnitude_error)),
                "weights": weights,
            }
        )
    selected = min(candidates, key=lambda item: (item["mse"], item["alpha"]))
    report = {
        "fit_outcome_role": roles[0],
        "selection_outcome_role": roles[1],
        "fit_targets": len(train),
        "fit_targets_with_known_action": len(known_targets),
        "selection_targets": len(validation_targets),
        "rank": len(components),
        "input_modalities": anchor.get("input_modalities", action.get("modalities")),
        "input_dimensions": len(input_indices),
        "input_indices_sha256": sha256(
            json.dumps(input_indices.tolist(), separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_ridge": selected["alpha"],
        "selection_mse": selected["mse"],
        "selection_mean_effect_pearson": selected["mean_effect_pearson"],
        "selection_mean_magnitude_absolute_error": selected[
            "mean_magnitude_absolute_error"
        ],
        "ridge_candidates": [item["alpha"] for item in candidates],
        "ridge_validation_mse": [item["mse"] for item in candidates],
        "training_null_effect_threshold": dynamics_manifest["direction"][
            "null_effect_threshold"
        ],
        "target_sha256": {
            role: sha256("\n".join(targets[role]).encode()).hexdigest() for role in roles
        },
        "leakage": {
            "contexts": ["K562"],
            "sealed_test_outcomes_used": False,
            "rpe1_outcomes_used": False,
        },
    }
    checkpoint = {
        "format_version": 1,
        "architecture": anchor.get(
            "checkpoint_architecture", "esm2_low_rank_latent_effect_ridge"
        ),
        "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
        "x_std": torch.from_numpy(x_std.astype(np.float32)),
        "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
        "components": torch.from_numpy(components.astype(np.float32)),
        "weights": torch.from_numpy(selected["weights"].astype(np.float32)),
        "input_indices": input_indices,
        "report": report,
    }
    output = Path(anchor["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    artifact = {"path": str(output), "bytes": output.stat().st_size, "sha256": file_sha256(output)}
    manifest = {
        "format_version": 1,
        "architecture": checkpoint["architecture"],
        "artifact": artifact,
        "fit": report,
        "source": {
            "base_config_path": str(base_path),
            "base_config_sha256": specification["base_config_sha256"],
            "latent_cache_sha256": base["inputs"]["latent_cache_sha256"],
            "action_cache_sha256": base["inputs"]["action_cache_sha256"],
            "dynamics_manifest_sha256": dynamics_manifest["manifest_sha256"],
        },
    }
    manifest["manifest_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    Path(anchor["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def anchored_dynamics_configs(path="configs/anchored_dynamics.yaml"):
    """Materialize the preregistered validation-only anchored candidates."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    base = _action_overridden_config(yaml.safe_load(base_path.read_text()), specification)
    anchor = specification["effect_anchor"]
    assert anchor["output_sha256"] != "PENDING" and anchor["manifest_sha256"] != "PENDING"
    artifact_path = Path(anchor["output_path"])
    assert (artifact_path.stat().st_size, file_sha256(artifact_path)) == (
        anchor["output_bytes"],
        anchor["output_sha256"],
    )
    manifest = json.loads(Path(anchor["manifest_path"]).read_text())
    declared = manifest.pop("manifest_sha256")
    assert declared == anchor["manifest_sha256"] == sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    configs = {}
    for name, experiment in specification["experiments"].items():
        config = deepcopy(base)
        config["seed"] = specification["seed"]
        config["effect_anchor"] = deepcopy(anchor)
        config["model"]["architecture"] = "anchored_decomposed_population_dynamics"
        config["model"]["anchor_gain_max"] = experiment["anchor_gain_max"]
        config["model"]["mean_residual_max_ratio"] = experiment[
            "mean_residual_max_ratio"
        ]
        config["model"]["action_modality_dropout"] = experiment.get(
            "action_modality_dropout", 0.0
        )
        config["model"]["action_context_conditioned"] = experiment.get(
            "action_context_conditioned", False
        )
        config["training"]["output_directory"] = experiment["output_directory"]
        config["training"]["resume_from"] = None
        config["revision"] = {
            **deepcopy(specification["revision"]),
            "candidate": name,
            "anchor_gain_max": experiment["anchor_gain_max"],
            "mean_residual_max_ratio": experiment["mean_residual_max_ratio"],
            "action_modality_dropout": experiment.get("action_modality_dropout", 0.0),
            "action_context_conditioned": experiment.get("action_context_conditioned", False),
            "target_split_seed": base["seed"],
        }
        configs[name] = config
    return configs, specification


def anchored_selected_entry(training_manifest, selection_manifest):
    """Resolve the frozen candidate while proving selection preceded sealed evaluation."""
    assert training_manifest["protocol"]["sealed_test_outcomes_used_for_fit_or_selection"] is False
    assert training_manifest["protocol"]["rpe1_perturbed_outcomes_used_for_fit_or_selection"] is False
    assert selection_manifest["leakage"] == {
        "context": "K562",
        "outcome_role": "perturbation_ood_validation",
        "rpe1_outcomes_used": False,
        "sealed_test_outcomes_used": False,
    }
    selected = selection_manifest["selected"]
    entry = training_manifest["artifacts"]["candidates"][selected["candidate"]]
    assert entry["best_checkpoint"] == {
        key: selected[key] for key in ("bytes", "path", "sha256")
    }
    assert (selected["best_validation_epoch"], selected["best_validation_loss"]) == (
        entry["full_run"]["best_validation_epoch"],
        entry["full_run"]["best_validation_loss"],
    )
    return selected["candidate"], entry


def select_multiteacher_candidate(training_manifest, specification):
    """Apply the frozen validation-loss margin without consulting test outcomes."""
    protocol = training_manifest["protocol"]
    assert protocol["sealed_test_outcomes_used_for_fit_or_selection"] is False
    assert protocol["rpe1_perturbed_outcomes_used_for_fit_or_selection"] is False
    rule = specification["selection"]
    assert rule["viewed_test_outcomes_used"] is False
    assert rule["checkpoint_rule"] == "minimum_original_latent_validation_loss"
    fallback = rule["fallback_candidate"]
    candidates = training_manifest["artifacts"]["candidates"]
    assert fallback in candidates
    alternatives = [
        name
        for name, experiment in specification["experiments"].items()
        if experiment.get("action_modality_dropout", 0.0) > 0
    ]
    assert len(alternatives) == 1 and alternatives[0] in candidates
    alternative = alternatives[0]
    fallback_loss = candidates[fallback]["full_run"]["best_validation_loss"]
    alternative_loss = candidates[alternative]["full_run"]["best_validation_loss"]
    improvement = fallback_loss - alternative_loss
    selected = (
        alternative
        if improvement >= rule["dropout_minimum_loss_improvement"]
        else fallback
    )
    return selected, improvement


def prepare_dynamics_manifest(config, config_path="configs/dynamics.yaml"):
    """Fit state-space statistics without reading validation, test, or RPE1 outcomes."""
    inputs, data, normalization = config["inputs"], config["data"], config["normalization"]
    for kind in ("latent", "action"):
        path = Path(inputs[f"{kind}_cache_path"])
        assert (path.stat().st_size, file_sha256(path)) == (
            inputs[f"{kind}_cache_bytes"],
            inputs[f"{kind}_cache_sha256"],
        )
    for kind in ("action", "replogle"):
        manifest = json.loads(Path(inputs[f"{kind}_manifest_path"]).read_text())
        assert manifest["manifest_sha256"] == inputs[f"{kind}_manifest_sha256"]

    with h5py.File(inputs["latent_cache_path"], "r") as cache:
        roles = cache["role"].asstr()[:]
        targets = cache["target"].asstr()[:]
        contexts = cache["context"].asstr()[:]
        batches = cache["source_batch"].asstr()[:]
        cell_ids = cache["cell_id"].asstr()[:]
        fit_indices = np.flatnonzero(np.isin(roles, normalization["fit_roles"]))
        fit_latents = cache["latent"][fit_indices]
        mean = fit_latents.mean(0, dtype=np.float64)
        variance = np.square(fit_latents.astype(np.float64) - mean).mean(0)
        std = np.sqrt(variance)
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
        dimension_scale = math.sqrt(fit_latents.shape[1])
        rng = np.random.default_rng(config["seed"])
        distance_indices = np.sort(
            rng.choice(fit_indices, normalization["distance_sample_cells"], replace=False)
        )
        distance_latents = (cache["latent"][distance_indices] - mean) / std / dimension_scale
        distances = np.linalg.norm(
            distance_latents - distance_latents[rng.permutation(len(distance_latents))], axis=1
        )
        median_distance = float(np.median(distances))

        control_indices = np.flatnonzero(roles == data["control_role"])
        assert set(contexts[control_indices]) == {data["context"]}
        control_latents = (cache["latent"][control_indices] - mean) / std / dimension_scale
        control_batches = batches[control_indices]
        control_means = {
            batch: control_latents[control_batches == batch].mean(0)
            for batch in sorted(set(control_batches))
        }
        train_indices = np.flatnonzero(roles == data["train_outcome_role"])
        validation_indices = np.flatnonzero(roles == data["validation_outcome_role"])
        assert set(contexts[train_indices]) == set(contexts[validation_indices]) == {
            data["context"]
        }
        train_latents = (cache["latent"][train_indices] - mean) / std / dimension_scale
        train_targets, train_batches = targets[train_indices], batches[train_indices]
        effect_norms = {}
        for target in sorted(set(train_targets)):
            selected = train_targets == target
            batch_counts = Counter(train_batches[selected])
            matched_control = sum(
                count * control_means[batch] for batch, count in batch_counts.items()
            ) / selected.sum()
            effect_norms[target] = float(
                np.linalg.norm(train_latents[selected].mean(0) - matched_control)
            )

    threshold = float(
        np.quantile(
            list(effect_norms.values()), normalization["null_effect_quantile"], method="linear"
        )
    )
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    split = replogle["targets"]["split"]["targets"]
    assert set(effect_norms) == set(split["train"])
    assert set(targets[validation_indices]) == set(split["validation"])
    report = {
        "format_version": 1,
        "config_sha256": file_sha256(config_path),
        "inputs": {
            key: value for key, value in inputs.items() if key.endswith(("_bytes", "_sha256"))
        },
        "normalization": {
            "fit_roles": normalization["fit_roles"],
            "fit_cells": len(fit_indices),
            "method": normalization["method"],
            "latent_mean": mean.tolist(),
            "latent_std": std.tolist(),
            "dimension_scale": dimension_scale,
            "distance_sample_cells": len(distance_indices),
            "distance_sample_cell_ids_sha256": sha256(
                "\n".join(sorted(cell_ids[distance_indices])).encode()
            ).hexdigest(),
            "median_training_latent_distance": median_distance,
        },
        "conditions": {
            "train": {
                "role": data["train_outcome_role"],
                "targets": len(set(targets[train_indices])),
                "cells": len(train_indices),
                "cells_per_target": dict(sorted(Counter(targets[train_indices]).items())),
            },
            "validation": {
                "role": data["validation_outcome_role"],
                "targets": len(set(targets[validation_indices])),
                "cells": len(validation_indices),
                "cells_per_target": dict(sorted(Counter(targets[validation_indices]).items())),
            },
            "controls": {
                "role": data["control_role"],
                "cells": len(control_indices),
                "batches": dict(sorted(Counter(control_batches).items())),
            },
        },
        "direction": {
            "null_effect_quantile": normalization["null_effect_quantile"],
            "null_effect_threshold": threshold,
            "training_effect_norms": effect_norms,
            "excluded_training_targets": sum(value < threshold for value in effect_norms.values()),
        },
        "leakage": {
            "normalization_roles": sorted(set(roles[fit_indices])),
            "outcome_context": sorted(set(contexts[train_indices])),
            "validation_used_for_statistics": False,
            "rpe1_outcomes_used_for_statistics": False,
            "sealed_test_outcomes_used_for_statistics": False,
        },
        "runtime": {"numpy": np.__version__, "h5py": version("h5py")},
    }
    report["manifest_sha256"] = sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    Path(inputs["dynamics_manifest_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def state_ablation_config(path):
    """Materialize a frozen alternative state space into the locked Stage 2 model."""
    path = Path(path)
    specification = yaml.safe_load(path.read_text())
    base_path = Path(specification["base_config_path"])
    assert file_sha256(base_path) == specification["base_config_sha256"]
    protocol = json.loads(Path(specification["specification_manifest_path"]).read_text())
    protocol_sha256 = protocol.pop("manifest_sha256")
    assert protocol_sha256 == specification["specification_manifest_sha256"] == sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state_manifest_path = Path(specification["cache_manifest_path"])
    state_manifest = json.loads(state_manifest_path.read_text())
    declared = state_manifest.pop("manifest_sha256")
    assert declared == sha256(
        json.dumps(state_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact = state_manifest["artifact"]
    assert (Path(artifact["path"]).stat().st_size, file_sha256(artifact["path"])) == (
        artifact["bytes"],
        artifact["sha256"],
    )
    assert artifact["cell_dim"] == yaml.safe_load(base_path.read_text())["model"]["cell_dim"]
    assert state_manifest["source"]["config_sha256"] == file_sha256(path)
    assert (
        state_manifest["source"]["specification_manifest_sha256"]
        == specification["specification_manifest_sha256"]
    )
    dynamics_manifest = json.loads(Path(specification["dynamics_manifest_path"]).read_text())
    dynamics_sha256 = dynamics_manifest.pop("manifest_sha256")
    assert dynamics_sha256 == state_manifest["source"]["dynamics_manifest_sha256"] == sha256(
        json.dumps(dynamics_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert dynamics_manifest["config_sha256"] == file_sha256(path)
    assert dynamics_manifest["inputs"]["latent_cache_sha256"] == artifact["sha256"]
    config = yaml.safe_load(base_path.read_text())
    config["inputs"].update(
        {
            "latent_cache_path": artifact["path"],
            "latent_cache_bytes": artifact["bytes"],
            "latent_cache_sha256": artifact["sha256"],
            "latent_manifest_path": str(state_manifest_path),
            "latent_manifest_sha256": declared,
            "dynamics_manifest_path": specification["dynamics_manifest_path"],
        }
    )
    config["training"]["output_directory"] = specification["output_directory"]
    config["training"]["resume_from"] = None
    config["ablation"] = {
        "name": specification["name"],
        "hypothesis": specification["hypothesis"],
        "base_config_path": str(base_path),
        "base_config_sha256": specification["base_config_sha256"],
        "cache_manifest_sha256": declared,
    }
    return config, specification, {**state_manifest, "manifest_sha256": declared}


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
        if f"{kind}_manifest_sha256" in inputs:
            assert declared == inputs[f"{kind}_manifest_sha256"]
        manifests[kind] = declared
    if "latent_manifest_path" in inputs:
        payload = json.loads(Path(inputs["latent_manifest_path"]).read_text())
        declared = payload.pop("manifest_sha256")
        assert declared == inputs["latent_manifest_sha256"] == sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifests["latent"] = declared
    cache_sha256 = {
        "latent": inputs["latent_cache_sha256"],
        "action": inputs["action_cache_sha256"],
    }
    if "effect_anchor" in config:
        anchor = config["effect_anchor"]
        assert file_sha256(anchor["output_path"]) == anchor["output_sha256"]
        payload = json.loads(Path(anchor["manifest_path"]).read_text())
        declared = payload.pop("manifest_sha256")
        assert declared == anchor["manifest_sha256"] == sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifests["effect_anchor"] = declared
        cache_sha256["effect_anchor"] = anchor["output_sha256"]
    return {
        "config_sha256": file_sha256(config_path),
        "cache_sha256": cache_sha256,
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
