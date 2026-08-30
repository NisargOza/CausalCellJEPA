# Export a tiny real-data State input with the same leakage and schema checks as the full pass.
from pathlib import Path

import yaml

from causalcelljepa.state_baseline import prepare_state_baseline

config = yaml.safe_load(Path("configs/state_baseline.yaml").read_text())
smoke = config["smoke"]
report = prepare_state_baseline(config, smoke["output_directory"], smoke)
assert report["artifacts"]["training"]["records"] == 20
assert report["artifacts"]["k562_controls"]["records"] == 4
assert report["artifacts"]["rpe1_controls"]["records"] == 4
assert report["split"]["sealed_test_outcomes_exported_for_training"] is False
assert report["split"]["rpe1_perturbed_outcomes_exported_for_training"] is False
assert report["split"]["rpe1_controls_exported_for_training"] is False
print({name: value["records"] for name, value in report["artifacts"].items() if "records" in value})
