# Run the complete preregistered external decoded-transcriptomic evaluation on CPU.
import json

from causalcelljepa.external_evaluation import run_nadig_external_transcriptomic_evaluation
from causalcelljepa.training import _git_state

assert _git_state()["dirty"] is False, "Full external evaluation requires a clean commit"
summary, comparisons, truth, provenance = run_nadig_external_transcriptomic_evaluation()
print(
    json.dumps(
        {
            "comparisons": sum(len(items) for items in comparisons.values()),
            "condition_summaries": len(summary["condition_metrics"]),
            "truth": truth,
            "git": provenance["git"],
        },
        indent=2,
        sort_keys=True,
    )
)
