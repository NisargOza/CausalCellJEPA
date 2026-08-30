# Exercise both v4 reporting evaluators on two K562 and two RPE1 conditions.
import json
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation
from causalcelljepa.readout import run_remaining_comparator_evaluation

config = yaml.safe_load(Path("configs/evaluation_contextual_multiteacher.yaml").read_text())
for name, runner, base_key in (
    ("latent", run_ablation_evaluation, "base_evaluation_config_path"),
    ("transcriptomic", run_remaining_comparator_evaluation, "base_transcriptomics_config_path"),
):
    section = config[name]
    base = yaml.safe_load(Path(section[base_key]).read_text())
    smoke = section["cpu_smoke"]
    regimes = {key: base["regimes"][key] for key in smoke["regimes"]}
    arguments = {
        "regimes": regimes,
        "repeats": smoke["repeats"],
        "output_directory": smoke["output_directory"],
    }
    arguments["max_conditions" if name == "latent" else "maximum_conditions"] = smoke[
        "maximum_conditions_per_regime"
    ]
    result = runner(section, base, **arguments)
    records = [
        json.loads(line)
        for line in (Path(smoke["output_directory"]) / "condition_metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(records) == len(regimes) * smoke["maximum_conditions_per_regime"]
    assert {record["model"] for record in records} == {
        "contextual_multiteacher_selected"
    }
    assert result[2 if name == "latent" else 3]["maximum_conditions_per_regime"] == 2
    print({"evaluation": name, "records": len(records)})
