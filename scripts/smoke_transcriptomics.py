# Exercise transcriptomic truth, five-model decoding, DE, pathway, and paired metrics.
# This gate uses K562 validation outcomes only and never opens a sealed evaluation role.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_transcriptomic_evaluation

config = yaml.safe_load(Path("configs/transcriptomics.yaml").read_text())
regimes = {
    "validation_smoke": {
        "context": "K562",
        "outcome_role": "perturbation_ood_validation",
        "control_role": "control_train",
    }
}
summary, paired, truth, provenance = run_transcriptomic_evaluation(
    config,
    regimes=regimes,
    repeats=1,
    maximum_conditions=4,
    output_directory="artifacts/transcriptomic_cpu_smoke",
)
assert truth["validation_smoke"]["outcome_role"] == "perturbation_ood_validation"
assert provenance["maximum_conditions_per_regime"] == 4
assert len(summary["retrieval"]) == 5
assert paired["condition_comparisons"] and paired["pathway_comparisons"]
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "retrieval_summaries": len(summary["retrieval"]),
        "truth": truth,
    }
)
