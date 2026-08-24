# Fit and evaluate the required direct expression-space ESM baseline on CPU.
# Selection remains K562-validation-only; sealed and RPE1 outcomes are metrics only.
import json
from pathlib import Path

import torch
import yaml

from causalcelljepa.readout import fit_direct_gene_baseline, run_direct_gene_evaluation
from causalcelljepa.resources import file_sha256

config_path = Path("configs/direct_gene.yaml")
config = yaml.safe_load(config_path.read_text())
assert config["transcriptomics_config_sha256"] == file_sha256(
    config["transcriptomics_config_path"]
)
config["transcriptomics"] = yaml.safe_load(
    Path(config["transcriptomics_config_path"]).read_text()
)
checkpoint_path = Path(config["direct_gene"]["checkpoint_path"])
assert not checkpoint_path.exists()
checkpoint = fit_direct_gene_baseline(config)
temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
torch.save(checkpoint, temporary)
temporary.replace(checkpoint_path)
summary, paired, truth, _ = run_direct_gene_evaluation(config, checkpoint)
print(
    json.dumps(
        {
            "fit": checkpoint["report"],
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
