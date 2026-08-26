# Exercise the sealed-data-inaccessible architecture selector on two K562 validation targets.
import json
from pathlib import Path

import torch

from causalcelljepa.architecture import run_anchored_validation
from causalcelljepa.dynamics import anchored_dynamics_configs

configs, specification = anchored_dynamics_configs()
root = Path(specification["cpu_smoke"]["output_directory"])
checkpoint_paths = {name: str(root / name / "latest.pt") for name in configs}
decision = run_anchored_validation(
    checkpoint_paths=checkpoint_paths,
    maximum_conditions=2,
    repeats=1,
    output_directory="artifacts/anchored_dynamics_validation_cpu_smoke",
    device=torch.device("cpu"),
    write_decision=False,
)
print(
    json.dumps(
        {
            "selected": decision["selected"],
            "candidate_summaries": decision["candidate_summaries"],
            "leakage": decision["leakage"],
        },
        indent=2,
        sort_keys=True,
    )
)
