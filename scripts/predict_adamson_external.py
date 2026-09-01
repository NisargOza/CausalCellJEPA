# Freeze the final Adamson predictions from controls only on a clean implementation commit.
# Perturbed and reference-only expression rows are neither selected nor retained here.
import json

from causalcelljepa.external_evaluation import predict_adamson_external
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full prediction requires a clean implementation commit"
_, manifest = predict_adamson_external()
print(json.dumps(manifest["prediction"], indent=2, sort_keys=True))
