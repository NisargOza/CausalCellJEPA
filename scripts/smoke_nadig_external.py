# Encode 32 controls/outcomes per external context through the frozen teacher on CPU.
# This proves schema and token compatibility before any full GPU cache is authorized.
import json
from pathlib import Path

import yaml

from causalcelljepa.external import write_nadig_latent_cache

config = yaml.safe_load(Path("configs/nadig_external_validation.yaml").read_text())
print(
    json.dumps(
        write_nadig_latent_cache(
            output_path=config["cpu_smoke"]["latent_cache_path"],
            maximum_per_context=config["cpu_smoke"]["maximum_cells_per_context"],
            device="cpu",
        ),
        indent=2,
        sort_keys=True,
    )
)
