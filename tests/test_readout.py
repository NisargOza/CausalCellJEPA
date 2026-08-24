import numpy as np
from scipy import sparse

from causalcelljepa.readout import (
    decoder_split,
    gene_effect_metrics,
    normalized_hvg_expression,
    pathway_agreement,
    regression_mse,
    ridge_solution,
    sufficient_statistics,
)
from causalcelljepa.representations import fit_pca_state, project_pca_state


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


def test_pca_state_is_deterministic_centered_and_canonically_oriented():
    expression = np.random.default_rng(8).normal(size=(24, 7)).astype(np.float32)
    first = fit_pca_state(expression, 3, 1, 2, 19)
    second = fit_pca_state(expression, 3, 1, 2, 19)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    mean, components, _ = first
    projected = project_pca_state(expression, mean, components)
    assert np.allclose(components @ components.T, np.eye(3), atol=1e-5)
    assert np.allclose(projected.mean(0), 0, atol=1e-6)
    anchors = np.abs(components).argmax(1)
    assert (components[np.arange(3), anchors] > 0).all()


def test_gene_effect_and_pathway_metrics_are_exact_for_perfect_prediction():
    observed = np.asarray([-5.0, 4.0, -3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    deg = np.asarray([True, True, True, False, False, False])
    metrics = gene_effect_metrics(observed, observed, 0, deg, (2, 3))
    assert np.isclose(metrics["all_effect_pearson"], 1.0)
    assert np.isclose(metrics["all_effect_spearman"], 1.0)
    assert np.isclose(metrics["target_excluded_effect_pearson"], 1.0)
    assert metrics["all_magnitude_absolute_error"] == 0
    assert metrics["deg_auprc"] == metrics["deg_auroc"] == metrics["deg_sign_accuracy"] == 1.0
    assert metrics["retrospective_top2_overlap"] == metrics["retrospective_top3_overlap"] == 1.0

    matrix = sparse.csr_matrix(
        np.asarray([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]])
    )
    pathways = pathway_agreement(observed, observed, matrix, (1, 2))
    assert np.isclose(pathways["pathway_nes_pearson"], 1.0)
    assert np.isclose(pathways["pathway_rank_spearman"], 1.0)
    assert pathways["pathway_nes_rmse"] == 0
    assert pathways["pathway_top1_jaccard"] == pathways["pathway_top2_jaccard"] == 1.0
