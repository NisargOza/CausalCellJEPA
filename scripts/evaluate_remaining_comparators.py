# Run the frozen action/state comparators in the proposal's common transcriptomic endpoint.
# The sealed full evaluation requires a clean committed worktree for exact provenance.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_remaining_comparator_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/evaluation_remaining_comparators.yaml").read_text())
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, paired, truth, _ = run_remaining_comparator_evaluation(config, base)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "condition_comparisons": len(paired["condition_comparisons"]),
        "pathway_comparisons": len(paired["pathway_comparisons"]),
        "truth": truth,
    }
)
