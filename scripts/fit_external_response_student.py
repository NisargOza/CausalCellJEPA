# Fit the exploratory multi-context student after the CPU smoke and a clean code commit.
# This script never reads a Replogle sealed target or an RPE1 perturbation outcome for fit.
import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import fit_external_response_student
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

config_path = Path("configs/external_response_pretraining.yaml")
config = yaml.safe_load(config_path.read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
assert _git_state()["dirty"] is False, "Full selection requires a clean code commit"
checkpoint = fit_external_response_student(config)
assert checkpoint["report"]["selected"]["passes_validation_gate"]
checkpoint_path = Path(config["external_response"]["checkpoint_path"])
assert not checkpoint_path.exists()
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
torch.save(checkpoint, temporary)
temporary.replace(checkpoint_path)
output = Path(config["external_response"]["output_directory"])
assert not output.exists()
output.mkdir(parents=True)
(output / "selection.json").write_text(
    json.dumps(checkpoint["report"], indent=2, sort_keys=True) + "\n"
)
print(json.dumps(checkpoint["report"]["selected"], indent=2, sort_keys=True))
