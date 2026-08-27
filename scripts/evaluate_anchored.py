# Run the one-time reporting-only latent and decoded evaluation of the frozen selection.
from pathlib import Path

import yaml

from causalcelljepa.evaluation import run_ablation_evaluation
from causalcelljepa.readout import run_remaining_comparator_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/evaluation_anchored.yaml").read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
for name, runner, base_key in (
    ("latent", run_ablation_evaluation, "base_evaluation_config_path"),
    ("transcriptomic", run_remaining_comparator_evaluation, "base_transcriptomics_config_path"),
):
    section = config[name]
    base = yaml.safe_load(Path(section[base_key]).read_text())
    result = runner(section, base)
    print({"evaluation": name, "summary_metrics": len(result[0]["condition_metrics"])})
