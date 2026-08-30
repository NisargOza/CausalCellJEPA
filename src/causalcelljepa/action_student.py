# Outcome-free masked-teacher fusion and ESM-feature student distillation.
# The exported residual action preserves raw frozen ESM beside the student state.
from hashlib import sha256

import torch
from torch import nn


def salt_public_split(targets, availability, seed, fractions=(0.8, 0.1, 0.1)):
    """Hash-rank targets with both public teachers into exact disjoint splits."""
    assert len(targets) == len(availability) and sum(fractions) == 1
    eligible = [index for index, row in enumerate(availability) if bool(row.all())]
    eligible.sort(key=lambda index: sha256(f"{seed}\0salt\0{targets[index]}".encode()).digest())
    first = round(len(eligible) * fractions[0])
    second = first + round(len(eligible) * fractions[1])
    return {
        "train": torch.tensor(eligible[:first]),
        "validation": torch.tensor(eligible[first:second]),
        "test": torch.tensor(eligible[second:]),
    }


def modality_statistics(features, modality_dims, indices):
    """Fit per-dimension public-teacher statistics on the public train split only."""
    blocks = features[:, : sum(modality_dims)].split(modality_dims, 1)
    means = [block[indices].mean(0) for block in blocks]
    scales = [block[indices].std(0, correction=False).clamp_min(1e-6) for block in blocks]
    return means, scales


def standardized_modalities(features, modality_dims, means, scales):
    blocks = features[:, : sum(modality_dims)].split(modality_dims, 1)
    return [(block - mean) / scale for block, mean, scale in zip(blocks, means, scales)]


class MaskedTeacherFusion(nn.Module):
    """Learn modality projectors by reconstructing each teacher from the other."""

    def __init__(self, modality_dims, joint_dim, hidden_dim):
        super().__init__()
        self.projectors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, joint_dim),
                nn.LayerNorm(joint_dim),
            )
            for width in modality_dims
        )
        self.decoders = nn.ModuleList(
            nn.Sequential(nn.Linear(joint_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, width))
            for width in modality_dims
        )
        self.output_norm = nn.LayerNorm(joint_dim)

    def forward(self, blocks, availability):
        projected = torch.stack(
            [projector(block) for projector, block in zip(self.projectors, blocks)], 1
        )
        weights = availability.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
        return self.output_norm((projected * weights.unsqueeze(2)).sum(1))


class SaltActionStudent(nn.Module):
    """Map frozen ESM features to a student state that predicts the joint teacher."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.predictor = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        state = self.encoder(features)
        return self.predictor(state), state


def representation_stable_rank(features):
    """Return the scale-invariant stable rank after centering samples."""
    values = torch.linalg.svdvals(features.detach() - features.detach().mean(0))
    return float(values.square().sum() / values.square().max())


def teacher_neighbor_overlap(query, reference, teacher_query, teacher_reference, k=10):
    """Measure overlap with the frozen teacher's public nearest-neighbor set."""

    def neighbors(left, right):
        left = torch.nn.functional.normalize(left, dim=1)
        right = torch.nn.functional.normalize(right, dim=1)
        return (left @ right.T).topk(k, dim=1).indices

    candidate = neighbors(query, reference)
    teacher = neighbors(teacher_query, teacher_reference)
    return float(
        torch.stack(
            [
                torch.isin(candidate[index], teacher[index]).float().mean()
                for index in range(len(query))
            ]
        ).mean()
    )
