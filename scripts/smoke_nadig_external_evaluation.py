# Exercise every frozen model and metric on two conditions per untouched context.
import json
from pathlib import Path

import yaml

from causalcelljepa.external_evaluation import run_nadig_external_latent_evaluation

config = yaml.safe_load(Path("configs/nadig_external_evaluation.yaml").read_text())
summary, comparisons, provenance = run_nadig_external_latent_evaluation(
    repeats=config["cpu_smoke"]["repeats"],
    maximum_conditions=config["cpu_smoke"]["maximum_conditions_per_context"],
    output_directory=config["cpu_smoke"]["output_directory"],
)
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
