# Materialize the frozen ESM+GO action teacher after source and code are committed.
import json

from causalcelljepa.actions import prepare_multiteacher_action
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full action preparation requires a clean commit"
print(json.dumps(prepare_multiteacher_action(), indent=2, sort_keys=True))
