# Evaluate the selection-frozen STRING+GO student on the same four reporting regimes.
# Test and RPE1 outcomes cannot alter its already-committed checkpoint or hyperparameters.
from pathlib import Path

import yaml

from causalcelljepa.readout import run_kernel_gene_evaluation
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

config = yaml.safe_load(Path("configs/string_kernel_gene.yaml").read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
assert _git_state()["dirty"] is False, "Full sealed evaluation requires a clean commit"
summary, paired, truth, _ = run_kernel_gene_evaluation(config)
print(
    {
        "condition_summaries": len(summary["condition_metrics"]),
        "pathway_summaries": len(summary["pathway_metrics"]),
        "condition_comparisons": len(paired["condition_comparisons"]),
        "pathway_comparisons": len(paired["pathway_comparisons"]),
        "truth": truth,
    }
)
