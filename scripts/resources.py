"""Download and derive the pinned GO BP masking collection."""

import json
from pathlib import Path

import yaml

from causalcelljepa.resources import build_go_bp_resource

stage1_config = yaml.safe_load(Path("configs/stage1.yaml").read_text())
replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
manifest = build_go_bp_resource(stage1_config, replogle_manifest["genes"]["hvg_gene_names"])
print(json.dumps(manifest["output"], indent=2))
