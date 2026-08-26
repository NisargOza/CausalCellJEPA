# Freeze exactly one anchored candidate using K562 validation outcomes only.
import json

import torch

from causalcelljepa.architecture import run_anchored_validation
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Architecture selection requires a clean commit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decision = run_anchored_validation(device=device)
print(json.dumps({"device": str(device), "decision": decision}, indent=2, sort_keys=True))
