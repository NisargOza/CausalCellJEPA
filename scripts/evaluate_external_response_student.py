# Report the frozen external-response student once against immutable references.
# This script performs no fitting, selection, calibration, or checkpoint mutation.
import json
from pathlib import Path

import yaml

from causalcelljepa.readout import run_kernel_gene_evaluation
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

path = Path("configs/external_response_evaluation.yaml")
config = yaml.safe_load(path.read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
assert _git_state()["dirty"] is False, "Full evaluation requires a clean code commit"
summary, _, _, _ = run_kernel_gene_evaluation(config)
selected = [
    item
    for item in summary["condition_metrics"]
    if item["model"] == config["evaluation"]["model_name"]
    and item["metric"]
    in {"all_effect_pearson", "all_magnitude_absolute_error", "deg_auprc"}
]
print(json.dumps(selected, indent=2, sort_keys=True))
