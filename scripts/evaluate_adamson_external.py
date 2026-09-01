# Run the sole permitted full Adamson outcome evaluation and apply the locked stop rule.
# A clean, tracked control-only prediction manifest must already exist.
import json

from causalcelljepa.external_evaluation import run_adamson_external_evaluation
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "One-shot confirmation requires a clean prediction commit"
_, _, truth, decision, _ = run_adamson_external_evaluation()
print(json.dumps({"truth": truth, "decision": decision}, indent=2, sort_keys=True))
