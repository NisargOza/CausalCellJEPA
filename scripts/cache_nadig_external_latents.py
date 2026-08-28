# Encode every frozen-eligible HepG2/Jurkat cell with the unchanged EMA JEPA teacher.
# Full caching requires CUDA only after the exact CPU path has passed its smoke test.
import json

from causalcelljepa.external import write_nadig_latent_cache
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full external caching requires a clean commit"
print(json.dumps(write_nadig_latent_cache(), indent=2, sort_keys=True))
