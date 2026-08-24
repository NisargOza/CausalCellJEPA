# Fit the separate latent-to-expression decoder from representation-permitted cells.
# Ridge selection uses only a deterministic cell subset of those same permitted roles.
import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import fit_readout

config = yaml.safe_load(Path("configs/readout.yaml").read_text())
output = Path(config["decoder"]["output_path"])
assert not output.exists()
checkpoint = fit_readout(config)
temporary = output.with_suffix(output.suffix + ".tmp")
torch.save(checkpoint, temporary)
temporary.replace(output)
print(json.dumps({"output": str(output), **checkpoint["report"]}, indent=2, sort_keys=True))
