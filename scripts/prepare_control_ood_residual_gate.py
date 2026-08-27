# Fit the post-hoc heterogeneity confidence gate using K562 control cells only.
# No perturbation outcomes or target identities are read by this preparation step.
import json
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config_path = Path("configs/control_ood_residual_gate.yaml")
config = yaml.safe_load(config_path.read_text())
base_path = Path(config["base_config_path"])
assert file_sha256(base_path) == config["base_config_sha256"]
base = yaml.safe_load(base_path.read_text())
calibration = config["calibration"]
latent_path = Path(base["inputs"]["latent_cache_path"])
assert file_sha256(latent_path) == base["inputs"]["latent_cache_sha256"]
manifest = json.loads(Path(base["inputs"]["dynamics_manifest_path"]).read_text())
declared = manifest["manifest_sha256"]
assert (
    declared
    == sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
)
normalization = manifest["normalization"]
mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
scale = np.asarray(normalization["latent_std"], dtype=np.float32) * normalization["dimension_scale"]
with h5py.File(latent_path, "r") as cache:
    roles, contexts = cache["role"].asstr()[:], cache["context"].asstr()[:]
    indices = np.flatnonzero((roles == calibration["role"]) & (contexts == calibration["context"]))
    assert len(indices) == manifest["conditions"]["controls"]["cells"]
    controls = (cache["latent"][indices] - mean) / scale

generator = np.random.default_rng(config["seed"])
population_means = np.stack(
    [
        controls[
            generator.choice(len(controls), calibration["population_size"], replace=False)
        ].mean(0)
        for _ in range(calibration["fit_populations"] + calibration["calibration_populations"])
    ]
)
fit, held_out = np.split(population_means, [calibration["fit_populations"]])
center, diagonal_scale = fit.mean(0), fit.std(0, ddof=1).clip(1e-8)
scores = np.square((held_out - center) / diagonal_scale).mean(1)
threshold = float(np.quantile(scores, calibration["threshold_quantile"]))
temperature = threshold - float(np.quantile(scores, calibration["temperature_quantile"]))
checkpoint = {
    "format_version": 1,
    "architecture": "control_population_residual_gate",
    "center": torch.from_numpy(center),
    "scale": torch.from_numpy(diagonal_scale),
    "threshold": threshold,
    "temperature": temperature,
}
output = Path(config["output_path"])
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, output)
report = {
    "format_version": 1,
    "architecture": checkpoint["architecture"],
    "artifact": {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
    },
    "calibration": {
        **calibration,
        "control_cells": len(controls),
        "threshold": threshold,
        "temperature": temperature,
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.95, 0.99)},
    },
    "leakage": {
        "roles_read": [calibration["role"]],
        "contexts_read": [calibration["context"]],
        "perturbed_outcomes_used": False,
        "rpe1_cells_used": False,
    },
    "source": {
        "config_sha256": file_sha256(config_path),
        "base_config_sha256": config["base_config_sha256"],
        "latent_cache_sha256": base["inputs"]["latent_cache_sha256"],
        "dynamics_manifest_sha256": manifest["manifest_sha256"],
    },
    "provenance": {
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
report["manifest_sha256"] = sha256(
    json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(config["manifest_path"]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
