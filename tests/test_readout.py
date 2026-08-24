import numpy as np

from causalcelljepa.readout import (
    decoder_split,
    normalized_hvg_expression,
    regression_mse,
    ridge_solution,
    sufficient_statistics,
)


def test_expression_normalization_uses_full_library_before_hvg_selection():
    counts = np.asarray([[1, 3, 6], [2, 2, 4]], dtype=np.float32)
    observed = normalized_hvg_expression(counts, np.asarray([0, 2]), 10)
    expected = np.log1p(np.asarray([[1, 6], [2.5, 5]], dtype=np.float32))
    assert np.allclose(observed, expected)


def test_decoder_split_excludes_forbidden_roles_and_is_deterministic():
    cell_ids = np.asarray([f"cell-{index}" for index in range(20)])
    roles = np.asarray(["dynamics_train"] * 8 + ["control_inference"] * 8 + ["double_ood_test"] * 4)
    first = decoder_split(cell_ids, roles, ["dynamics_train", "control_inference"], 7, 0.25)
    second = decoder_split(cell_ids, roles, ["dynamics_train", "control_inference"], 7, 0.25)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    selected = np.concatenate(first)
    assert len(first[1]) == 4 and set(roles[selected]) == {"dynamics_train", "control_inference"}


def test_sufficient_statistics_recover_exact_linear_readout():
    generator = np.random.default_rng(4)
    latent = generator.normal(size=(80, 3)).astype(np.float32)
    weights = generator.normal(size=(3, 5))
    bias = generator.normal(size=5)
    expression = (latent @ weights + bias).astype(np.float32)
    train, validation = np.arange(60), np.arange(60, 80)
    train_stats = sufficient_statistics(latent, expression, train, np.zeros(3), np.ones(3), 13)
    validation_stats = sufficient_statistics(
        latent, expression, validation, np.zeros(3), np.ones(3), 11
    )
    solution = ridge_solution(train_stats, 0)
    assert regression_mse(validation_stats, solution) < 1e-12
    assert np.allclose(solution[:-1], weights, atol=1e-6)
    assert np.allclose(solution[-1], bias, atol=1e-6)
