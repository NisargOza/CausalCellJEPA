# Evaluate the frozen STRING/external-response mixture selected from controls alone.
# The gate and both components are immutable; this script performs reporting only.
import json
from pathlib import Path

import yaml

from causalcelljepa.readout import run_kernel_gene_evaluation
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

path = Path("configs/control_gated_response_evaluation.yaml")
config = yaml.safe_load(path.read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
assert _git_state()["dirty"] is False, "Full evaluation requires a clean code commit"
summary, _, _, provenance = run_kernel_gene_evaluation(config)
print(
    json.dumps(
        {
            "control_gate": provenance["control_gate"],
            "headline": [
                item
                for item in summary["condition_metrics"]
                if item["metric"]
                in {"all_effect_pearson", "all_magnitude_absolute_error", "deg_auprc"}
            ],
        },
        indent=2,
        sort_keys=True,
    )
)
