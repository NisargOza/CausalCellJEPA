# Fit PCA only on representation-permitted cells, then project every frozen evaluation cell.
# Metadata is copied from the pinned JEPA cache solely to preserve exact row/split alignment.
import json
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import yaml

from causalcelljepa.dynamics import prepare_dynamics_manifest
from causalcelljepa.representations import fit_pca_state, project_pca_state
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config_path = Path("configs/pca_state.yaml")
config = yaml.safe_load(config_path.read_text())
assert file_sha256(config["base_config_path"]) == config["base_config_sha256"]
specification = json.loads(Path(config["specification_manifest_path"]).read_text())
declared = specification.pop("manifest_sha256")
assert declared == config["specification_manifest_sha256"] == sha256(
    json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
for kind in ("expression", "metadata"):
    path = Path(config["inputs"][f"{kind}_cache_path"])
    assert (path.stat().st_size, file_sha256(path)) == (
        config["inputs"][f"{kind}_cache_bytes"],
        config["inputs"][f"{kind}_cache_sha256"],
    )
git = _git_state()
assert git["dirty"] is False, "Full PCA materialization requires a clean commit"

fit = config["fit"]
output = Path(config["inputs"]["output_path"])
assert fit["dtype"] == "float32" and not output.exists()
temporary = output.with_suffix(output.suffix + ".tmp")
assert not temporary.exists()
output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(config["inputs"]["expression_cache_path"], "r") as expression, h5py.File(
    config["inputs"]["metadata_cache_path"], "r"
) as metadata:
    assert int(expression.attrs["cells"]) == int(metadata.attrs["cells"])
    assert expression.attrs["hvg_sha256"] == metadata.attrs["hvg_sha256"]
    roles = metadata["role"].asstr()[:]
    cell_ids = metadata["cell_id"].asstr()[:]
    fit_indices = np.flatnonzero(np.isin(roles, fit["roles"]))
    assert len(fit_indices) == fit["cells"]
    assert Counter(roles[fit_indices]) == Counter(specification["policy"]["fit_role_counts"])
    assert sha256("\n".join(sorted(cell_ids[fit_indices])).encode()).hexdigest() == fit[
        "cell_ids_sha256"
    ]
    fit_expression = expression["expression"][fit_indices]
    mean, components, singular_values = fit_pca_state(
        fit_expression,
        fit["dimensions"],
        fit["oversampling"],
        fit["power_iterations"],
        config["seed"],
    )
    del fit_expression
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update(
            {
                "format_version": 1,
                "cells": int(expression.attrs["cells"]),
                "cell_dim": fit["dimensions"],
                "dtype": fit["dtype"],
                "representation": "centered_randomized_pca",
                "hvg_sha256": expression.attrs["hvg_sha256"],
                "replogle_manifest_sha256": metadata.attrs["replogle_manifest_sha256"],
                "role_counts_json": metadata.attrs["role_counts_json"],
                "cache_provenance_json": json.dumps(
                    {
                        "config_sha256": file_sha256(config_path),
                        "specification_manifest_sha256": declared,
                        "expression_cache_sha256": config["inputs"][
                            "expression_cache_sha256"
                        ],
                        "metadata_cache_sha256": config["inputs"]["metadata_cache_sha256"],
                        "fit_cell_ids_sha256": fit["cell_ids_sha256"],
                        "runtime_source_sha256": _runtime_source_hash(),
                        "runtime_environment": _runtime_environment(),
                        "git": git,
                    },
                    sort_keys=True,
                ),
            }
        )
        destination.create_dataset("expression_mean", data=mean)
        destination.create_dataset("component", data=components)
        destination.create_dataset("singular_value", data=singular_values)
        latent = destination.create_dataset(
            "latent",
            (int(expression.attrs["cells"]), fit["dimensions"]),
            dtype="f4",
            chunks=(fit["block_size"], fit["dimensions"]),
        )
        string = h5py.string_dtype("utf-8")
        copied = {
            name: destination.create_dataset(name, (len(roles),), dtype=string)
            for name in ("cell_id", "context", "target", "role", "source_batch")
        }
        source_rows = destination.create_dataset("source_row", (len(roles),), dtype="i8")
        for start in range(0, len(roles), fit["block_size"]):
            stop = min(start + fit["block_size"], len(roles))
            projected = project_pca_state(expression["expression"][start:stop], mean, components)
            assert np.isfinite(projected).all()
            latent[start:stop] = projected
            for name, values in copied.items():
                values[start:stop] = metadata[name][start:stop]
            source_rows[start:stop] = metadata["source_row"][start:stop]
        destination.flush()
temporary.replace(output)

cache_sha256 = file_sha256(output)
dynamics = deepcopy(yaml.safe_load(Path(config["base_config_path"]).read_text()))
dynamics["inputs"].update(
    {
        "latent_cache_path": str(output),
        "latent_cache_bytes": output.stat().st_size,
        "latent_cache_sha256": cache_sha256,
        "dynamics_manifest_path": config["dynamics_manifest_path"],
    }
)
report = prepare_dynamics_manifest(dynamics, config_path)
print(
    json.dumps(
        {
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": cache_sha256,
            "cells": len(roles),
            "fit_cells": len(fit_indices),
            "dimensions": fit["dimensions"],
            "dynamics_manifest": config["dynamics_manifest_path"],
            "dynamics_manifest_sha256": report["manifest_sha256"],
            "median_training_latent_distance": report["normalization"][
                "median_training_latent_distance"
            ],
        },
        indent=2,
    )
)
