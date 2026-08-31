# Score the frozen State predictions against all four locked transcriptomic regimes.
# Sealed and RPE1 outcomes are opened here for reporting only, never for selection.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_state_baseline_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/state_baseline.yaml").read_text())
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, paired, truth, _ = run_state_baseline_evaluation(config, base)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "condition_comparisons": len(paired["condition_comparisons"]),
        "pathway_comparisons": len(paired["pathway_comparisons"]),
        "truth": truth,
    }
)
