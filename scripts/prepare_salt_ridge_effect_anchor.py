# Fit the K562 train-only SALT ridge effect anchor and enforce the frozen validation gate.
import json
from hashlib import sha256
from pathlib import Path

import yaml

from causalcelljepa.dynamics import prepare_effect_anchor
from causalcelljepa.training import _git_state

PATH = Path("configs/salt_ridge_dynamics.yaml")


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    assert _git_state()["dirty"] is False, "effect-anchor fitting requires a clean protocol"
    specification = yaml.safe_load(PATH.read_text())
    action = json.loads(Path(specification["action_manifest_path"]).read_text())
    action_declared = action.pop("manifest_sha256")
    assert action_declared == specification["action_manifest_sha256"] == self_hash(action)
    assert action["artifact"]["eligible_for_downstream_anchor"] is True

    gate = specification["anchor_gate"]
    reference = json.loads(Path(gate["reference_manifest_path"]).read_text())
    reference_declared = reference.pop("manifest_sha256")
    assert reference_declared == gate["reference_manifest_sha256"] == self_hash(reference)

    manifest = prepare_effect_anchor(PATH)
    fit = manifest["fit"]
    assert fit["selection_mean_effect_pearson"] >= (
        gate["reference_validation_pearson"] - gate["maximum_pearson_regression"]
    )
    assert fit["selection_mse"] <= gate["reference_validation_mse"] * gate[
        "maximum_mse_ratio"
    ]
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
