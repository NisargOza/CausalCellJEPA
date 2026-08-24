# Run a bounded CPU smoke for every frozen mechanism ablation before sealed evaluation.
import json
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation

config = yaml.safe_load(Path("configs/evaluation_ablations.yaml").read_text())
base_config = yaml.safe_load(Path(config["base_evaluation_config_path"]).read_text())
smoke = config["cpu_smoke"]
regimes = {name: base_config["regimes"][name] for name in smoke["regimes"]}
summary, comparisons, provenance = run_ablation_evaluation(
    config,
    base_config,
    regimes=regimes,
    repeats=smoke["repeats"],
    max_conditions=smoke["maximum_conditions_per_regime"],
    output_directory=smoke["output_directory"],
)
records = [
    json.loads(line)
    for line in (Path(smoke["output_directory"]) / "condition_metrics.jsonl")
    .read_text()
    .splitlines()
]
assert len(records) == len(config["models"]) * smoke["maximum_conditions_per_regime"]
assert {record["model"] for record in records} == set(config["models"])
assert provenance["maximum_conditions_per_regime"] == smoke["maximum_conditions_per_regime"]
assert summary["condition_metrics"] and summary["retrieval"] and comparisons
print(
    {
        "records": len(records),
        "models": sorted({record["model"] for record in records}),
        "paired_comparisons": len(comparisons),
    }
)
