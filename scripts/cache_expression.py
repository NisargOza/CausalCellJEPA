# Materialize normalized 3,000-HVG values aligned one-to-one with the latent cache.
# This is a CPU preprocessing step and verifies both pinned raw H5AD files first.
import json
from pathlib import Path

import yaml

from causalcelljepa.readout import write_expression_cache

config = yaml.safe_load(Path("configs/readout.yaml").read_text())
print(json.dumps(write_expression_cache(config), indent=2, sort_keys=True))
