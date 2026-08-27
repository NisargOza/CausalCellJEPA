# Run the one-time exploratory latent evaluation of the frozen control-OOD gate.
# The clean-commit requirement keeps the post-hoc analysis fully provenance-bound.
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/evaluation_control_ood_residual_gate.yaml").read_text())
base = yaml.safe_load(Path(config["base_evaluation_config_path"]).read_text())
assert _git_state()["dirty"] is False, "Full exploratory evaluation requires a clean commit"
summary, comparisons, _ = run_ablation_evaluation(config, base)
print({"summary_metrics": len(summary["condition_metrics"]), "comparisons": len(comparisons)})
