# Run the frozen mechanism ablations under the sealed four-regime evaluation protocol.
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/evaluation_ablations.yaml").read_text())
base_config = yaml.safe_load(Path(config["base_evaluation_config_path"]).read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, comparisons, _ = run_ablation_evaluation(config, base_config)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "retrieval_summaries": len(summary["retrieval"]),
        "paired_comparisons": len(comparisons),
    }
)
