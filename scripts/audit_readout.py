# Quantify the frozen decoder ceiling from observed outcome latents after model selection.
# This diagnostic consumes test outcomes and is never a predictive baseline or model input.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_readout_oracle

config = yaml.safe_load(Path("configs/transcriptomics.yaml").read_text())
summary, truth, _ = run_readout_oracle(config)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "truth": truth,
    }
)
