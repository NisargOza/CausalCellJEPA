# Exercise the exact Adamson control-inference path on 32 cells before the full CPU run.
# The smoke artifact is ignored and cannot create the frozen prediction manifest.
import json
from pathlib import Path

import yaml

from causalcelljepa.external_evaluation import predict_adamson_external

config = yaml.safe_load(Path("configs/adamson_external_evaluation.yaml").read_text())
_, manifest = predict_adamson_external(
    maximum_controls=32,
    output_path=config["outputs"]["smoke_prediction_path"],
    write_manifest=False,
)
assert manifest["leakage"]["perturbed_outcomes_used"] is False
print(json.dumps(manifest["prediction"], indent=2, sort_keys=True))
