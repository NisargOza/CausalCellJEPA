# Exercise the frozen gate on two K562 and two RPE1 perturbation conditions.
# This smoke checks the complete loading, gating, metric, and provenance path on CPU.
import json
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation

config = yaml.safe_load(Path("configs/evaluation_control_ood_residual_gate.yaml").read_text())
base = yaml.safe_load(Path(config["base_evaluation_config_path"]).read_text())
smoke = config["cpu_smoke"]
regimes = {key: base["regimes"][key] for key in smoke["regimes"]}
summary, _, provenance = run_ablation_evaluation(
    config,
    base,
    regimes,
    smoke["repeats"],
    smoke["maximum_conditions_per_regime"],
    smoke["output_directory"],
)
records = [
    json.loads(line)
    for line in (Path(smoke["output_directory"]) / "condition_metrics.jsonl")
    .read_text()
    .splitlines()
]
assert len(records) == 4 and all(0 <= item["residual_gate_confidence"] <= 1 for item in records)
assert provenance["residual_gate_manifest_sha256"] == config["residual_gate"]["manifest_sha256"]
print(
    {
        item["regime"]: item["mean"]
        for item in summary["condition_metrics"]
        if item["metric"] == "residual_gate_confidence"
    }
)
