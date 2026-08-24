# Exercise the full-rank PCA fit twice on a bounded real-expression subset before materialization.
# No outcome outside the representation-permitted role mask may enter this smoke fit.
import json
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import yaml

from causalcelljepa.representations import fit_pca_state, project_pca_state

config = yaml.safe_load(Path("configs/pca_state.yaml").read_text())
fit, smoke = config["fit"], config["cpu_smoke"]
with h5py.File(config["inputs"]["expression_cache_path"], "r") as expression, h5py.File(
    config["inputs"]["metadata_cache_path"], "r"
) as metadata:
    roles = metadata["role"].asstr()[:]
    fit_indices = np.flatnonzero(np.isin(roles, fit["roles"]))[: smoke["fit_cells"]]
    values = expression["expression"][fit_indices]
    first = fit_pca_state(
        values,
        fit["dimensions"],
        fit["oversampling"],
        fit["power_iterations"],
        config["seed"],
    )
    second = fit_pca_state(
        values,
        fit["dimensions"],
        fit["oversampling"],
        fit["power_iterations"],
        config["seed"],
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    mean, components, singular_values = first
    projected = project_pca_state(
        expression["expression"][: smoke["transform_cells"]], mean, components
    )
assert np.isfinite(projected).all()
assert np.allclose(components @ components.T, np.eye(fit["dimensions"]), atol=2e-5)
assert (components[np.arange(len(components)), np.abs(components).argmax(1)] > 0).all()
print(
    json.dumps(
        {
            "fit_cells": len(fit_indices),
            "dimensions": components.shape[0],
            "transform_cells": len(projected),
            "exact_repeat": True,
            "component_sha256": sha256(components.tobytes()).hexdigest(),
            "minimum_singular_value": float(singular_values.min()),
            "maximum_absolute_projection": float(np.abs(projected).max()),
        },
        indent=2,
        sort_keys=True,
    )
)
