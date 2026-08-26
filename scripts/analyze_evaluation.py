# Add the proposal-required paired target-level tests without rerunning model inference.
import json
from pathlib import Path

import yaml

from causalcelljepa.evaluation import paired_condition_comparisons
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
root = Path(config["output_directory"])
records_path = root / "condition_metrics.jsonl"
records = [json.loads(line) for line in records_path.read_text().splitlines()]
result = {
    "comparisons": paired_condition_comparisons(
        records, config["metrics"]["bootstrap_resamples"], config["seed"]
    ),
    "provenance": {
        "condition_metrics_sha256": file_sha256(records_path),
        "evaluation_provenance_sha256": file_sha256(root / "provenance.json"),
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    },
}
assert result["provenance"]["git"]["dirty"] is False
(root / "paired_comparisons.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print({"paired_comparisons": len(result["comparisons"])})
