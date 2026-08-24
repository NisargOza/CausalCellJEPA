# Matched 256-D state-space comparators fitted only on representation-permitted cells.
# Projection helpers are separate from dynamics so held-out expression cannot enter training.
import numpy as np
import torch


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
