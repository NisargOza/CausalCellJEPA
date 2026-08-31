# Compress only the audited split metadata needed for remote State inference.
# No expression matrix or held-out outcome value is read by this export.
import json
from pathlib import Path

import yaml

from causalcelljepa.state_baseline import prepare_state_prediction_metadata
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/state_baseline.yaml").read_text())
assert _git_state()["dirty"] is False, "Full metadata export requires a clean commit"
report = prepare_state_prediction_metadata(config)
assert (report["bytes"], report["sha256"]) == (
    config["prediction"]["metadata_bytes"],
    config["prediction"]["metadata_sha256"],
)
print(json.dumps(report, indent=2, sort_keys=True))
