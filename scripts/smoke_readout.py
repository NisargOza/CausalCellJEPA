# Exercise raw-expression caching and decoder fitting on a bounded real-data subset.
# Passing this CPU gate is required before the full 340,684-cell cache is created.
import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import fit_readout, write_expression_cache

config = yaml.safe_load(Path("configs/readout.yaml").read_text())
directory = Path("artifacts/readout_cpu_smoke")
directory.mkdir(parents=True, exist_ok=True)
expression_path = directory / "expression.h5"
if expression_path.exists():
    expression_path.unlink()
config["expression_cache"]["output_path"] = str(expression_path)
cache_report = write_expression_cache(config, maximum_cells=4096, verify_raw=False)
# The bounded prefix contains K562 only; RPE1 controls enter the full cache audit.
config["decoder"]["fit_roles"] = ["control_train", "dynamics_train"]
checkpoint = fit_readout(config, maximum_cells=1024)
assert checkpoint["report"]["selected_validation_mse"] < checkpoint["report"][
    "gene_mean_validation_mse"
]
assert checkpoint["report"]["fit_roles"] == sorted(config["decoder"]["fit_roles"])
torch.save(checkpoint, directory / "readout.pt")
print(json.dumps({"cache": cache_report, "decoder": checkpoint["report"]}, indent=2, sort_keys=True))
