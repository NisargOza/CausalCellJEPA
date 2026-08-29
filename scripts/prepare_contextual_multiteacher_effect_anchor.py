# Fit the K562-only availability-aware effect anchor and enforce its frozen CPU gate.
import json
from hashlib import sha256
from pathlib import Path

import yaml

from causalcelljepa.dynamics import prepare_effect_anchor

path = "configs/contextual_multiteacher_dynamics.yaml"
specification = yaml.safe_load(Path(path).read_text())
gate = specification["anchor_gate"]
reference = json.loads(Path(gate["reference_manifest_path"]).read_text())
declared = reference.pop("manifest_sha256")
assert declared == gate["reference_manifest_sha256"] == sha256(
    json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
manifest = prepare_effect_anchor(path)
fit = manifest["fit"]
assert fit["selection_mean_effect_pearson"] >= (
    gate["reference_validation_pearson"] - gate["maximum_pearson_regression"]
)
assert fit["selection_mse"] <= gate["reference_validation_mse"] * gate["maximum_mse_ratio"]
print(manifest["artifact"])
print(fit)
