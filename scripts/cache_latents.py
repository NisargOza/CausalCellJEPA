# Embed every required Replogle cell exactly once with the frozen canonical EMA teacher.
# The cache keeps split/condition metadata beside each latent to preserve leakage boundaries.
"""Create the fixed Stage 2 cell-state cache after a completed Stage 1 GPU run."""

import json
from collections import Counter
from pathlib import Path

import h5py
import torch
import yaml
from torch.utils.data import DataLoader

from causalcelljepa.data import ReplogleTokenDataset
from causalcelljepa.model import load_frozen_teacher
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

stage2_config = yaml.safe_load(Path("configs/stage2.yaml").read_text())
config = stage2_config["latent_cache"]
teacher_path, output = Path(config["teacher_path"]), Path(config["output_path"])
assert torch.cuda.is_available(), "Full latent caching requires CUDA; refusing CPU fallback"
assert file_sha256(teacher_path) == config["teacher_sha256"]
assert config["dtype"] == "float32" and not output.exists()

dataset = ReplogleTokenDataset(all_required=True)
assert len(dataset) == config["expected_cells"]
device = torch.device("cuda")
teacher, teacher_payload = load_frozen_teacher(
    teacher_path, dataset.config["data"]["hvg_count"], device
)
options = {}
if config["num_workers"]:
    options = {
        "persistent_workers": config["persistent_workers"],
        "prefetch_factor": config["prefetch_factor"],
    }
loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"],
    generator=torch.Generator().manual_seed(stage2_config["seed"]),
    **options,
)

output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
assert not temporary.exists()
string = h5py.string_dtype("utf-8")
role_counts = Counter(sample[3] for sample in dataset.samples)
with h5py.File(temporary, "w") as cache:
    cache.attrs.update(
        {
            "format_version": 1,
            "cells": len(dataset),
            "cell_dim": teacher_payload["cell_dim"],
            "dtype": config["dtype"],
            "teacher_sha256": config["teacher_sha256"],
            "replogle_manifest_sha256": dataset.manifest["manifest_sha256"],
            "hvg_sha256": dataset.manifest["genes"]["hvg_sha256"],
            "role_counts_json": json.dumps(dict(sorted(role_counts.items())), sort_keys=True),
            "teacher_provenance_json": json.dumps(teacher_payload["provenance"], sort_keys=True),
            "cache_provenance_json": json.dumps(
                {
                    "config_sha256": file_sha256("configs/stage2.yaml"),
                    "runtime_source_sha256": _runtime_source_hash(),
                    "runtime_environment": _runtime_environment(),
                    "git": _git_state(),
                },
                sort_keys=True,
            ),
        }
    )
    latents = cache.create_dataset(
        "latent",
        (len(dataset), teacher_payload["cell_dim"]),
        dtype="f4",
        chunks=(config["chunk_cells"], teacher_payload["cell_dim"]),
    )
    metadata = {
        name: cache.create_dataset(name, (len(dataset),), dtype=string)
        for name in ("cell_id", "context", "target", "role", "source_batch")
    }
    source_rows = cache.create_dataset("source_row", (len(dataset),), dtype="i8")
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            end = offset + len(batch["cell_id"])
            embedding = teacher(
                batch["gene_ids"].to(device, non_blocking=True),
                batch["values"].to(device, non_blocking=True),
                batch["padding_mask"].to(device, non_blocking=True),
            )
            assert torch.isfinite(embedding).all()
            latents[offset:end] = embedding.cpu().numpy()
            for name, destination in metadata.items():
                destination[offset:end] = batch[name]
            source_rows[offset:end] = batch["source_row"].numpy()
            offset = end
    assert offset == len(dataset)
    cache.flush()
temporary.replace(output)
print(
    json.dumps(
        {
            "output": str(output),
            "sha256": file_sha256(output),
            "cells": len(dataset),
            "cell_dim": teacher_payload["cell_dim"],
            "role_counts": dict(sorted(role_counts.items())),
            "device": torch.cuda.get_device_name(),
        },
        indent=2,
    )
)
