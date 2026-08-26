# Run the locked four-regime evaluation from the checksum-pinned selected checkpoint.
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/evaluation.yaml").read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, baseline, provenance = run_evaluation(config)
print({"condition_summaries": len(summary["condition_metrics"]), "baseline": baseline})
