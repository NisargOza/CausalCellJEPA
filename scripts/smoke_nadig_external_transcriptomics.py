# Decode two external conditions per context and exercise every frozen metric on CPU.
import json
from pathlib import Path

import yaml

from causalcelljepa.external_evaluation import run_nadig_external_transcriptomic_evaluation

config = yaml.safe_load(Path("configs/nadig_external_evaluation.yaml").read_text())
summary, comparisons, truth, provenance = run_nadig_external_transcriptomic_evaluation(
    repeats=config["cpu_smoke"]["repeats"],
    maximum_conditions=config["cpu_smoke"]["maximum_conditions_per_context"],
    output_directory=config["cpu_smoke"]["transcriptomic_output_directory"],
)
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
