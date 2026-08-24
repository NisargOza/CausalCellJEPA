# Derive Stage 2 normalization/direction statistics after the frozen autoencoder cache exists.
# Only K562 controls and dynamics-training outcomes enter these fitted statistics.
import json
from copy import deepcopy
from pathlib import Path

import yaml

from causalcelljepa.dynamics import prepare_dynamics_manifest
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

config_path = Path("configs/autoencoder_state.yaml")
config = yaml.safe_load(config_path.read_text())
assert file_sha256(config["base_config_path"]) == config["base_config_sha256"]
assert _git_state()["dirty"] is False
cache = Path(config["cache"]["output_path"])
dynamics = deepcopy(yaml.safe_load(Path(config["base_config_path"]).read_text()))
dynamics["inputs"].update(
    {
        "latent_cache_path": str(cache),
        "latent_cache_bytes": cache.stat().st_size,
        "latent_cache_sha256": file_sha256(cache),
        "dynamics_manifest_path": config["dynamics_manifest_path"],
    }
)
report = prepare_dynamics_manifest(dynamics, config_path)
print(
    json.dumps(
        {
            "cache_bytes": cache.stat().st_size,
            "cache_sha256": dynamics["inputs"]["latent_cache_sha256"],
            "dynamics_manifest": config["dynamics_manifest_path"],
            "dynamics_manifest_sha256": report["manifest_sha256"],
            "fit_cells": report["normalization"]["fit_cells"],
            "median_training_latent_distance": report["normalization"][
                "median_training_latent_distance"
            ],
        },
        indent=2,
    )
)
