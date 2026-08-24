# Exercise evaluation only on four K562 perturbation-OOD validation conditions.
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_evaluation

config = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
config["metrics"]["bootstrap_resamples"] = 100
regimes = {
    "validation_smoke": {
        "context": "K562",
        "outcome_role": "perturbation_ood_validation",
        "control_role": "control_train",
    }
}
summary, baseline, _ = run_evaluation(
    config,
    regimes=regimes,
    repeats=1,
    max_conditions=4,
    output_directory="artifacts/evaluation_cpu_smoke",
)
assert len(summary["retrieval"]) == 4 and baseline["selection_targets"] == 100
print({"validation_conditions": 4, "models": 4, "metrics": len(summary["condition_metrics"])})
