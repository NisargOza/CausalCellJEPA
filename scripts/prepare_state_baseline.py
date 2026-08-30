# Export the complete leakage-safe State baseline inputs.
from pathlib import Path

import yaml

from causalcelljepa.state_baseline import prepare_state_baseline
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/state_baseline.yaml").read_text())
assert _git_state()["dirty"] is False, "Full State export requires a clean commit"
report = prepare_state_baseline(config)
print({"training_cells": report["artifacts"]["training"]["records"]})
