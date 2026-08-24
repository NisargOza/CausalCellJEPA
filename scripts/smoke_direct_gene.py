# Fit the direct gene-space ESM baseline on bounded K562 train/validation targets.
# This gate does not read sealed outcomes or any RPE1 perturbed outcome.
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import direct_gene_predictions, fit_direct_gene_baseline

config = yaml.safe_load(Path("configs/direct_gene.yaml").read_text())
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
checkpoint = fit_direct_gene_baseline(config, maximum_targets=32)
prediction = direct_gene_predictions(
    checkpoint, config["transcriptomics"]["inputs"]["action_cache_path"]
)
assert checkpoint["report"]["fit_targets"] == checkpoint["report"]["selection_targets"] == 32
assert 1 <= checkpoint["report"]["rank"] <= 32
assert all(torch.isfinite(value).all() for value in checkpoint.values() if torch.is_tensor(value))
assert all(value.shape == (3000,) for value in prediction.values())
print(checkpoint["report"])
