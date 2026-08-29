# Fit the multimodal mean-effect anchor from K562 train/validation roles only.
import json

from causalcelljepa.dynamics import prepare_effect_anchor
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full anchor preparation requires a clean commit"
print(
    json.dumps(
        prepare_effect_anchor("configs/multiteacher_dynamics.yaml"),
        indent=2,
        sort_keys=True,
    )
)
