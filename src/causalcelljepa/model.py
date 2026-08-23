# From-scratch Stage 1 Cell-State JEPA fixed by RESEARCH_PROPOSAL.md.
# The encoder sees gene/value tokens only: no context, action, batch, or cell-line IDs.
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class CellEncoder(nn.Module):
    """Compact Perceiver encoder for unordered continuous-expression tokens."""

    def __init__(
        self,
        vocab_size=3_000,
        token_dim=192,
        latent_queries=32,
        blocks=3,
        heads=6,
        ffn_dim=768,
        dropout=0.10,
        cell_dim=256,
    ):
        super().__init__()
        self.padding_id = vocab_size
        self.gene_embedding = nn.Embedding(vocab_size + 1, token_dim, padding_idx=vocab_size)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.latents = nn.Parameter(torch.randn(latent_queries, token_dim) * 0.02)
        self.query_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, token_dim),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            token_dim,
            heads,
            ffn_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, blocks, nn.LayerNorm(token_dim), enable_nested_tensor=False
        )
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, cell_dim))

    def forward(self, gene_ids, values, padding_mask):
        tokens = self.token_norm(
            self.gene_embedding(gene_ids) + self.value_encoder(values.unsqueeze(-1))
        )
        latents = self.latents.unsqueeze(0).expand(gene_ids.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(latents),
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        latents = latents + attended
        latents = latents + self.cross_ffn(latents)
        return self.output(self.blocks(latents).mean(dim=1))


class CellJEPA(nn.Module):
    """Online encoder, frozen EMA teacher, and whole-cell latent predictor."""

    def __init__(self, encoder, predictor_hidden=768):
        super().__init__()
        self.online = encoder
        self.teacher = deepcopy(encoder).requires_grad_(False)
        cell_dim = encoder.output[-1].out_features
        self.predictor = nn.Sequential(
            nn.LayerNorm(cell_dim),
            nn.Linear(cell_dim, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, cell_dim),
        )
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, student_view, teacher_view):
        context = self.online(**student_view)
        with torch.no_grad():
            target = self.teacher(**teacher_view).detach()
        return self.predictor(context), target, context

    @torch.no_grad()
    def update_teacher(self, momentum):
        for teacher, online in zip(self.teacher.parameters(), self.online.parameters()):
            teacher.lerp_(online, 1.0 - momentum)


def mask_gene_tokens(
    gene_ids,
    values,
    padding_mask,
    programs=(),
    mask_range=(0.30, 0.60),
    biological_probability=0.25,
    minimum_program_genes=3,
    padding_id=3_000,
    generator=None,
):
    """Create a masked student view while leaving the complete teacher view untouched."""
    masked_genes, masked_values, masked_padding = (
        gene_ids.clone(),
        values.clone(),
        padding_mask.clone(),
    )
    for row in range(gene_ids.shape[0]):
        visible = torch.nonzero(~padding_mask[row], as_tuple=False).flatten()
        selected = None
        if programs and torch.rand((), generator=generator) < biological_probability:
            program_index = int(torch.randint(len(programs), (), generator=generator))
            positions = visible[torch.isin(gene_ids[row, visible], programs[program_index])]
            if positions.numel() >= minimum_program_genes:
                selected = positions
        if selected is None:
            ratio = torch.empty(()).uniform_(*mask_range, generator=generator).item()
            count = min(visible.numel() - 1, max(1, round(ratio * visible.numel())))
            selected = visible[torch.randperm(visible.numel(), generator=generator)[:count]]
        selected = selected[: max(0, visible.numel() - 1)]
        masked_genes[row, selected] = padding_id
        masked_values[row, selected] = 0
        masked_padding[row, selected] = True
    return {
        "gene_ids": masked_genes,
        "values": masked_values,
        "padding_mask": masked_padding,
    }


def jepa_loss(prediction, target, context, variance_weight=0.05, covariance_weight=0.005):
    """Normalized latent prediction plus light VICReg-style anti-collapse terms."""
    prediction_loss = F.smooth_l1_loss(F.normalize(prediction), F.normalize(target))
    variance_loss = F.relu(1.0 - torch.sqrt(context.var(dim=0, unbiased=False) + 1e-4)).mean()
    centered = context - context.mean(dim=0)
    covariance = centered.T @ centered / max(context.shape[0] - 1, 1)
    covariance_loss = (
        covariance[~torch.eye(covariance.shape[0], dtype=bool, device=context.device)].pow(2).mean()
    )
    total = prediction_loss + variance_weight * variance_loss + covariance_weight * covariance_loss
    return {
        "loss": total,
        "prediction": prediction_loss,
        "variance": variance_loss,
        "covariance": covariance_loss,
    }


def ema_momentum(step, total_steps, start=0.996, end=0.9999):
    """Cosine schedule the teacher momentum over optimizer updates."""
    return end - (end - start) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0
