"""Freeze hashes and role counts for the admitted-cell Stage 1 validation split."""

import json
from pathlib import Path

import yaml

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.training import write_stage1_split_manifest

stage1_config = yaml.safe_load(Path("configs/stage1.yaml").read_text())
replogle_manifest = json.loads(Path("manifests/replogle_v1.json").read_text())
manifest = write_stage1_split_manifest(
    stage1_config["validation"]["manifest_path"],
    ReplogleTokenDataset(),
    stage1_config,
    replogle_manifest,
)
print(json.dumps(manifest, indent=2))
