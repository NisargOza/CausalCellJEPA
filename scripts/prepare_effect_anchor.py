# Fit the post-primary action anchor using only K562 training/validation outcomes.
import json

from causalcelljepa.dynamics import prepare_effect_anchor

print(json.dumps(prepare_effect_anchor(), indent=2, sort_keys=True))
