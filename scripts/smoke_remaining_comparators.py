# Exercise all representation decoders and paired metrics on two already-evaluated IID targets.
# This bounded CPU gate must pass before the frozen four-regime comparator evaluation.
import json
from pathlib import Path

import yaml

from causalcelljepa.readout import run_remaining_comparator_evaluation

config = yaml.safe_load(Path("configs/evaluation_remaining_comparators.yaml").read_text())
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
smoke = config["cpu_smoke"]
regimes = {name: base["regimes"][name] for name in smoke["regimes"]}
summary, paired, _, provenance = run_remaining_comparator_evaluation(
    config,
    base,
    regimes=regimes,
    repeats=smoke["repeats"],
    maximum_conditions=smoke["maximum_conditions_per_regime"],
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
assert summary["condition_metrics"] and paired["condition_comparisons"]
print({"records": len(records), "models": sorted(config["models"])})
