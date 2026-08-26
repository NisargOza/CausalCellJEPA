# Decode frozen latent predictions and run the fixed gene/pathway evaluation protocol.
# Every metric is aggregated at perturbation-condition level; outcomes never fit a model.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_transcriptomic_evaluation

config = yaml.safe_load(Path("configs/transcriptomics.yaml").read_text())
summary, paired, truth, _ = run_transcriptomic_evaluation(config)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "retrieval_summaries": len(summary["retrieval"]),
        "condition_comparisons": len(paired["condition_comparisons"]),
        "pathway_comparisons": len(paired["pathway_comparisons"]),
        "truth": truth,
    }
)
