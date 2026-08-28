# Run the complete preregistered external latent evaluation on CPU.
import json

from causalcelljepa.external_evaluation import run_nadig_external_latent_evaluation
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full external evaluation requires a clean commit"
summary, comparisons, provenance = run_nadig_external_latent_evaluation(device="cpu")
print(
    json.dumps(
        {
            "comparisons": len(comparisons),
            "condition_summaries": len(summary["condition_metrics"]),
            "git": provenance["git"],
        },
        indent=2,
        sort_keys=True,
    )
)
