# Fit Stage 2 latent statistics from training-visible K562 data and freeze provenance.
# Validation/test/RPE1 outcomes are never read into normalization or weak-effect thresholds.
import json
from pathlib import Path

import yaml

from causalcelljepa.dynamics import prepare_dynamics_manifest

config_path = Path("configs/dynamics.yaml")
report = prepare_dynamics_manifest(yaml.safe_load(config_path.read_text()), config_path)
print(
    json.dumps(
        {
            "manifest_sha256": report["manifest_sha256"],
            "fit_cells": report["normalization"]["fit_cells"],
            "train_targets": report["conditions"]["train"]["targets"],
            "validation_targets": report["conditions"]["validation"]["targets"],
            "median_training_latent_distance": report["normalization"][
                "median_training_latent_distance"
            ],
            "null_effect_threshold": report["direction"]["null_effect_threshold"],
            "direction_excluded_targets": report["direction"]["excluded_training_targets"],
        },
        indent=2,
    )
)
