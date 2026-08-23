# Stage 1 tests pin sparse tokenization, masking, encoder invariance, and EMA behavior.
import numpy as np
import torch

from causalcelljepa.data import masking_seed, tokenize_cell
from causalcelljepa.model import (
    CellEncoder,
    CellJEPA,
    ema_momentum,
    jepa_loss,
    mask_gene_tokens,
)


def test_tokenization_normalizes_before_hvg_selection_and_breaks_ties_by_gene():
    counts = np.array([10, 5, 5, 20, 1, 0], dtype=np.float32)
    genes, values, padding = tokenize_cell(counts, np.array([0, 1, 2, 4, 5]), max_tokens=4)
    assert genes.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(values, np.log1p(counts[[0, 1, 2, 4]] * (10_000 / counts.sum())))
    assert not padding.any()
    genes, values, padding = tokenize_cell(counts, np.array([0, 5]), max_tokens=4)
    assert genes.tolist() == [0, 2, 2, 2]
    assert values[1:].tolist() == [0.0, 0.0, 0.0]
    assert padding.tolist() == [False, True, True, True]


def test_masking_is_reproducible_preserves_teacher_and_supports_programs():
    genes = torch.tensor([[0, 1, 2, 3, 4, 5, 3000, 3000], [5, 4, 3, 2, 1, 0, 3000, 3000]])
    values = torch.ones(2, 8)
    padding = genes == 3000
    first = mask_gene_tokens(genes, values, padding, generator=torch.Generator().manual_seed(7))
    second = mask_gene_tokens(genes, values, padding, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(genes[:, :6], torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]))
    assert torch.all((~first["padding_mask"]).sum(dim=1) >= 1)
    biological = mask_gene_tokens(
        genes[:1],
        values[:1],
        padding[:1],
        programs=(torch.tensor([1, 2, 3]),),
        biological_probability=1.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert biological["padding_mask"][0, 1:4].all()
    assert masking_seed("cell-1", 2, 17) == masking_seed("cell-1", 2, 17)
    assert masking_seed("cell-1", 2, 17) != masking_seed("cell-1", 3, 17)


def test_locked_encoder_is_permutation_and_padding_invariant_within_budget():
    encoder = CellEncoder()
    assert 2_000_000 <= sum(parameter.numel() for parameter in encoder.parameters()) <= 4_000_000
    encoder.eval()
    genes = torch.tensor([[2, 8, 1, 5, 3000, 3000], [4, 3, 9, 7, 6, 3000]])
    values = torch.rand(2, 6)
    padding = genes == 3000
    expected = encoder(genes, values, padding)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    actual = encoder(genes[:, permutation], values[:, permutation], padding[:, permutation])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    changed_genes, changed_values = genes.clone(), values.clone()
    changed_genes[padding], changed_values[padding] = 42, 99
    torch.testing.assert_close(encoder(changed_genes, changed_values, padding), expected)
    assert expected.shape == (2, 256)


def test_jepa_teacher_is_frozen_loss_is_finite_and_ema_is_exact():
    model = CellJEPA(CellEncoder())
    assert sum(parameter.numel() for parameter in model.predictor.parameters()) < 1_000_000
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    assert all(
        torch.equal(online, teacher)
        for online, teacher in zip(model.online.parameters(), model.teacher.parameters())
    )
    genes = torch.randint(0, 3_000, (4, 12))
    values = torch.rand(4, 12)
    padding = torch.zeros(4, 12, dtype=torch.bool)
    teacher_view = {"gene_ids": genes, "values": values, "padding_mask": padding}
    student_view = mask_gene_tokens(
        genes, values, padding, generator=torch.Generator().manual_seed(9)
    )
    prediction, target, context = model(student_view, teacher_view)
    losses = jepa_loss(prediction, target, context)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.online.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    teacher_before = next(model.teacher.parameters()).clone()
    with torch.no_grad():
        next(model.online.parameters()).add_(1)
    model.update_teacher(0.9)
    torch.testing.assert_close(next(model.teacher.parameters()), teacher_before + 0.1)
    assert ema_momentum(0, 100) == 0.996
    assert ema_momentum(100, 100) == 0.9999
