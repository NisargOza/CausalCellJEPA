# Fit Stage 2 latent statistics from training-visible K562 data and freeze provenance.
# Validation/test/RPE1 outcomes are never read into normalization or weak-effect thresholds.
import json
import math
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import yaml

from causalcelljepa.resources import file_sha256

config_path = Path("configs/dynamics.yaml")
config = yaml.safe_load(config_path.read_text())
inputs, data_config, normalization = config["inputs"], config["data"], config["normalization"]
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
    distances = np.linalg.norm(distance_latents - distance_latents[rng.permutation(len(distance_latents))], axis=1)
    median_distance = float(np.median(distances))

    control_indices = np.flatnonzero(roles == data_config["control_role"])
    assert set(contexts[control_indices]) == {data_config["context"]}
    control_latents = (cache["latent"][control_indices] - mean) / std / dimension_scale
    control_batches = batches[control_indices]
    control_means = {
        batch: control_latents[control_batches == batch].mean(0)
        for batch in sorted(set(control_batches))
    }
    train_indices = np.flatnonzero(roles == data_config["train_outcome_role"])
    validation_indices = np.flatnonzero(roles == data_config["validation_outcome_role"])
    assert set(contexts[train_indices]) == set(contexts[validation_indices]) == {
        data_config["context"]
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
        effect_norms[target] = float(np.linalg.norm(train_latents[selected].mean(0) - matched_control))

threshold = float(
    np.quantile(list(effect_norms.values()), normalization["null_effect_quantile"], method="linear")
)
replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
split = replogle["targets"]["split"]["targets"]
assert set(effect_norms) == set(split["train"])
assert set(targets[validation_indices]) == set(split["validation"])
report = {
    "format_version": 1,
    "config_sha256": file_sha256(config_path),
    "inputs": {
        key: value
        for key, value in inputs.items()
        if key.endswith(("_bytes", "_sha256"))
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
            "role": data_config["train_outcome_role"],
            "targets": len(set(targets[train_indices])),
            "cells": len(train_indices),
            "cells_per_target": dict(sorted(Counter(targets[train_indices]).items())),
        },
        "validation": {
            "role": data_config["validation_outcome_role"],
            "targets": len(set(targets[validation_indices])),
            "cells": len(validation_indices),
            "cells_per_target": dict(sorted(Counter(targets[validation_indices]).items())),
        },
        "controls": {
            "role": data_config["control_role"],
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
print(
    json.dumps(
        {
            "manifest_sha256": report["manifest_sha256"],
            "fit_cells": len(fit_indices),
            "train_targets": report["conditions"]["train"]["targets"],
            "validation_targets": report["conditions"]["validation"]["targets"],
            "median_training_latent_distance": median_distance,
            "null_effect_threshold": threshold,
            "direction_excluded_targets": report["direction"]["excluded_training_targets"],
        },
        indent=2,
    )
)
