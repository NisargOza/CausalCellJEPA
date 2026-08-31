# Fit and freeze the exploratory kernel action student on CPU.
# This command reads K562 training/validation outcomes only and performs no test evaluation.
import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import fit_kernel_gene_student
from causalcelljepa.resources import file_sha256

config_path = Path("configs/kernel_gene.yaml")
config = yaml.safe_load(config_path.read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
checkpoint = fit_kernel_gene_student(config)
checkpoint_path = Path(config["kernel_gene"]["checkpoint_path"])
assert not checkpoint_path.exists()
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
torch.save(checkpoint, temporary)
temporary.replace(checkpoint_path)
output = Path(config["kernel_gene"]["output_directory"])
assert not output.exists()
output.mkdir(parents=True)
(output / "selection.json").write_text(
    json.dumps(checkpoint["report"], indent=2, sort_keys=True) + "\n"
)
print(json.dumps(checkpoint["report"], indent=2, sort_keys=True))
