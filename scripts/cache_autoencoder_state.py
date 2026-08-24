# Encode every required cell with the selected frozen reconstruction encoder.
# Test outcomes are projected only after training/selection is complete and cannot update weights.
import json
from collections import Counter
from pathlib import Path

import h5py
import torch
import yaml

from causalcelljepa.representations import autoencoder_provenance, build_autoencoder
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config_path = "configs/autoencoder_state.yaml"
config = yaml.safe_load(Path(config_path).read_text())
assert torch.cuda.is_available(), "Full autoencoder latent caching requires CUDA"
assert _git_state()["dirty"] is False
output_directory = Path(config["training"]["output_directory"])
latest_path, best_path = output_directory / "latest.pt", output_directory / "best.pt"
latest = torch.load(latest_path, map_location="cpu", weights_only=False)
best = torch.load(best_path, map_location="cpu", weights_only=False)
expected_provenance = autoencoder_provenance(config, config_path)
assert latest["state"]["complete"] is True
assert latest["provenance"] == best["provenance"] == expected_provenance
assert latest["configuration"] == best["configuration"] == config
assert best["state"]["best_validation_epoch"] == latest["state"]["best_validation_epoch"]

model = build_autoencoder(config).cuda().eval()
model.load_state_dict(best["model"])
cache_config = config["cache"]
output = Path(cache_config["output_path"])
assert cache_config["dtype"] == "float32" and not output.exists()
temporary = output.with_suffix(output.suffix + ".tmp")
assert not temporary.exists()
output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(config["inputs"]["expression_cache_path"], "r") as expression, h5py.File(
    config["inputs"]["metadata_cache_path"], "r"
) as metadata, h5py.File(temporary, "w") as destination:
    cells = int(expression.attrs["cells"])
    assert cells == cache_config["expected_cells"] == int(metadata.attrs["cells"])
    roles = metadata["role"].asstr()[:]
    destination.attrs.update(
        {
            "format_version": 1,
            "cells": cells,
            "cell_dim": config["model"]["cell_dim"],
            "dtype": cache_config["dtype"],
            "representation": "reconstruction_autoencoder",
            "hvg_sha256": expression.attrs["hvg_sha256"],
            "replogle_manifest_sha256": metadata.attrs["replogle_manifest_sha256"],
            "role_counts_json": json.dumps(dict(sorted(Counter(roles).items()))),
            "cache_provenance_json": json.dumps(
                {
                    "config_sha256": expected_provenance["config_sha256"],
                    "selected_checkpoint_sha256": file_sha256(best_path),
                    "selected_validation_epoch": latest["state"]["best_validation_epoch"],
                    "selected_validation_loss": latest["state"]["best_validation_loss"],
                    "training_completion_reason": latest["state"]["completion_reason"],
                    "runtime_source_sha256": _runtime_source_hash(),
                    "runtime_environment": _runtime_environment(),
                    "git": _git_state(),
                },
                sort_keys=True,
            ),
        }
    )
    latent = destination.create_dataset(
        "latent",
        (cells, config["model"]["cell_dim"]),
        dtype="f4",
        chunks=(cache_config["chunk_cells"], config["model"]["cell_dim"]),
    )
    string = h5py.string_dtype("utf-8")
    copied = {
        name: destination.create_dataset(name, (cells,), dtype=string)
        for name in ("cell_id", "context", "target", "role", "source_batch")
    }
    source_rows = destination.create_dataset("source_row", (cells,), dtype="i8")
    with torch.inference_mode():
        for start in range(0, cells, cache_config["batch_size"]):
            stop = min(start + cache_config["batch_size"], cells)
            values = torch.from_numpy(expression["expression"][start:stop]).cuda()
            embedding = model.encode(values)
            assert torch.isfinite(embedding).all()
            latent[start:stop] = embedding.cpu().numpy()
            for name, target in copied.items():
                target[start:stop] = metadata[name][start:stop]
            source_rows[start:stop] = metadata["source_row"][start:stop]
    destination.flush()
temporary.replace(output)
print(
    json.dumps(
        {
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "cells": cells,
            "cell_dim": config["model"]["cell_dim"],
            "selected_checkpoint": str(best_path),
            "selected_checkpoint_sha256": file_sha256(best_path),
        },
        indent=2,
    )
)
