# Run the frozen decoder on both Stage 2 replication seeds in all four regimes.
import json
from pathlib import Path

import yaml

from causalcelljepa.readout import run_remaining_comparator_evaluation
from causalcelljepa.training import _git_state

config = yaml.safe_load(
    Path("configs/evaluation_stage2_replication_transcriptomics.yaml").read_text()
)
base = yaml.safe_load(Path(config["base_transcriptomics_config_path"]).read_text())
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, paired, truth, _ = run_remaining_comparator_evaluation(config, base)
print(
    json.dumps(
        {
            "condition_summaries": len(summary["condition_metrics"]),
            "pathway_summaries": len(summary["pathway_metrics"]),
            "retrieval_summaries": len(summary["retrieval"]),
            "condition_comparisons": len(paired["condition_comparisons"]),
            "pathway_comparisons": len(paired["pathway_comparisons"]),
            "truth": truth,
        },
        indent=2,
        sort_keys=True,
    )
)
