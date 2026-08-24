# Leakage-safe normalized-expression caching and latent-to-transcriptome readout.
# The decoder is separate from—and never backpropagates into—the frozen world model.
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
import yaml

from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


def normalized_hvg_expression(counts, hvg_columns, library_size=10_000):
    """Normalize by each full measured library before selecting the frozen HVGs."""
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(1)
    assert np.isfinite(counts).all() and (counts >= 0).all() and (totals > 0).all()
    return np.log1p(
        counts[:, hvg_columns] * (np.float32(library_size) / totals)[:, None]
    ).astype(np.float32, copy=False)


def write_expression_cache(config, output_path=None, maximum_cells=None, verify_raw=True):
    """Stream raw H5AD rows into a latent-aligned normalized 3,000-HVG cache."""
    inputs, cache_config = config["inputs"], config["expression_cache"]
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    assert replogle["manifest_sha256"] == inputs["replogle_manifest_sha256"]
    replogle_config = yaml.safe_load(Path(inputs["replogle_config_path"]).read_text())
    assert file_sha256(inputs["replogle_config_path"]) == replogle["runtime"]["config_sha256"]
    latent_path = Path(inputs["latent_cache_path"])
    assert (latent_path.stat().st_size, file_sha256(latent_path)) == (
        inputs["latent_cache_bytes"],
        inputs["latent_cache_sha256"],
    )
    output = Path(output_path or cache_config["output_path"])
    assert cache_config["dtype"] == "float32" and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    assert not temporary.exists()
    with h5py.File(latent_path, "r") as latent:
        cells = min(int(latent.attrs["cells"]), maximum_cells or int(latent.attrs["cells"]))
        assert maximum_cells is not None or cells == cache_config["expected_cells"]
        contexts = latent["context"].asstr()[:cells]
        source_rows = latent["source_row"][:cells]
        roles = latent["role"].asstr()[:cells]
        with h5py.File(temporary, "w") as destination:
            destination.attrs.update(
                {
                    "format_version": 1,
                    "cells": cells,
                    "hvg_count": cache_config["hvg_count"],
                    "dtype": cache_config["dtype"],
                    "library_size": replogle_config["data"]["library_size"],
                    "latent_cache_sha256": inputs["latent_cache_sha256"],
                    "replogle_manifest_sha256": inputs["replogle_manifest_sha256"],
                    "hvg_sha256": replogle["genes"]["hvg_sha256"],
                    "role_counts_json": json.dumps(dict(sorted(Counter(roles).items()))),
                    "provenance_json": json.dumps(
                        {
                            "config_sha256": file_sha256("configs/readout.yaml"),
                            "runtime_source_sha256": _runtime_source_hash(),
                            "runtime_environment": _runtime_environment(),
                            "git": _git_state(),
                        },
                        sort_keys=True,
                    ),
                }
            )
            expression = destination.create_dataset(
                "expression",
                (cells, cache_config["hvg_count"]),
                dtype="f4",
                chunks=(cache_config["block_size"], cache_config["hvg_count"]),
            )
            hvg_ids = replogle["genes"]["hvg_gene_ids"]
            for context in replogle_config["data"]["contexts"]:
                source = replogle_config["data"]["files"][context]
                path = Path(inputs["raw_directory"]) / source["filename"]
                assert path.stat().st_size == source["bytes"]
                if verify_raw:
                    assert file_sha256(path) == source["sha256"]
                positions = np.flatnonzero(contexts == context)
                if not len(positions):
                    continue
                assert np.array_equal(positions, np.arange(positions[0], positions[-1] + 1))
                data = ad.read_h5ad(path, backed="r")
                columns = np.asarray([data.var_names.get_loc(gene) for gene in hvg_ids])
                rows = source_rows[positions]
                assert np.all(rows[1:] > rows[:-1])
                for start in range(0, len(positions), cache_config["block_size"]):
                    selected = positions[start : start + cache_config["block_size"]]
                    counts = np.asarray(data.X[source_rows[selected]])
                    expression[selected] = normalized_hvg_expression(
                        counts, columns, replogle_config["data"]["library_size"]
                    )
                data.file.close()
            destination.flush()
    temporary.replace(output)
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "cells": cells,
        "hvg_count": cache_config["hvg_count"],
        "role_counts": dict(sorted(Counter(roles).items())),
    }


def decoder_split(cell_ids, roles, fit_roles, seed, validation_fraction, maximum_cells=None):
    """Create an exact deterministic cell split restricted to representation-visible roles."""
    eligible = np.flatnonzero(np.isin(roles, fit_roles))
    ranked = sorted(
        eligible,
        key=lambda index: sha256(f"{seed}\0readout\0{cell_ids[index]}".encode()).digest(),
    )
    if maximum_cells is not None:
        ranked = ranked[:maximum_cells]
    validation_cells = max(1, round(validation_fraction * len(ranked)))
    return np.sort(ranked[validation_cells:]), np.sort(ranked[:validation_cells])


def sufficient_statistics(latent, expression, indices, mean, scale, block_size):
    """Accumulate the exact multivariate linear-regression sufficient statistics."""
    width, genes = latent.shape[1] + 1, expression.shape[1]
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros((width, genes), dtype=np.float64)
    response_square = 0.0
    for start in range(0, len(indices), block_size):
        selected = indices[start : start + block_size]
        design = np.empty((len(selected), width), dtype=np.float32)
        design[:, :-1] = (latent[selected] - mean) / scale
        design[:, -1] = 1
        response = expression[selected].astype(np.float32)
        gram += design.T @ design
        cross += design.T @ response
        response_square += float(np.square(response.astype(np.float64)).sum())
    return {"cells": len(indices), "gram": gram, "cross": cross, "response_square": response_square}


def ridge_solution(statistics, alpha):
    """Solve the linear decoder while leaving the intercept unpenalized."""
    penalty = np.eye(statistics["gram"].shape[0], dtype=np.float64) * alpha
    penalty[-1, -1] = 0
    return np.linalg.solve(statistics["gram"] + penalty, statistics["cross"])


def regression_mse(statistics, solution):
    """Evaluate an un-clipped linear decoder exactly from sufficient statistics."""
    residual = (
        statistics["response_square"]
        - 2 * np.sum(solution * statistics["cross"])
        + np.sum(solution * (statistics["gram"] @ solution))
    )
    return max(0.0, float(residual)) / (statistics["cells"] * statistics["cross"].shape[1])


def fit_readout(config, maximum_cells=None):
    """Select and refit the decoder using only explicitly permitted cell roles."""
    inputs, decoder = config["inputs"], config["decoder"]
    assert decoder["architecture"] == "linear"
    dynamics = json.loads(Path(inputs["dynamics_manifest_path"]).read_text())
    assert dynamics["manifest_sha256"] == inputs["dynamics_manifest_sha256"]
    latent_mean = np.asarray(dynamics["normalization"]["latent_mean"], dtype=np.float32)
    latent_scale = (
        np.asarray(dynamics["normalization"]["latent_std"], dtype=np.float32)
        * dynamics["normalization"]["dimension_scale"]
    )
    with h5py.File(inputs["latent_cache_path"], "r") as latent, h5py.File(
        config["expression_cache"]["output_path"], "r"
    ) as expression_cache:
        cells = int(expression_cache.attrs["cells"])
        assert expression_cache.attrs["latent_cache_sha256"] == inputs["latent_cache_sha256"]
        cell_ids, roles = latent["cell_id"].asstr()[:cells], latent["role"].asstr()[:cells]
        train, validation = decoder_split(
            cell_ids,
            roles,
            decoder["fit_roles"],
            config["seed"],
            decoder["validation_fraction"],
            maximum_cells,
        )
        train_stats = sufficient_statistics(
            latent["latent"],
            expression_cache["expression"],
            train,
            latent_mean,
            latent_scale,
            decoder["block_size"],
        )
        validation_stats = sufficient_statistics(
            latent["latent"],
            expression_cache["expression"],
            validation,
            latent_mean,
            latent_scale,
            decoder["block_size"],
        )
        candidates = []
        for alpha in decoder["ridge_candidates"]:
            solution = ridge_solution(train_stats, alpha)
            candidates.append((regression_mse(validation_stats, solution), alpha))
        validation_mse, selected_alpha = min(candidates)
        combined = {
            key: train_stats[key] + validation_stats[key]
            for key in ("cells", "gram", "cross", "response_square")
        }
        solution = ridge_solution(combined, selected_alpha)
        baseline_mean = train_stats["cross"][-1] / train_stats["cells"]
        baseline = np.vstack(
            (np.zeros((solution.shape[0] - 1, solution.shape[1])), baseline_mean)
        )
        baseline_mse = regression_mse(validation_stats, baseline)
        used_roles = sorted(set(roles[np.concatenate((train, validation))]))
        assert used_roles == sorted(decoder["fit_roles"])
        report = {
            "architecture": decoder["architecture"],
            "fit_roles": used_roles,
            "fit_cells": combined["cells"],
            "train_cells": train_stats["cells"],
            "validation_cells": validation_stats["cells"],
            "validation_fraction": decoder["validation_fraction"],
            "ridge_candidates": decoder["ridge_candidates"],
            "ridge_validation_mse": [value for value, _ in candidates],
            "selected_ridge": selected_alpha,
            "selected_validation_mse": validation_mse,
            "gene_mean_validation_mse": baseline_mse,
            "validation_explained_fraction": 1 - validation_mse / baseline_mse,
            "split_cell_ids_sha256": {
                "train": sha256("\n".join(sorted(cell_ids[train])).encode()).hexdigest(),
                "validation": sha256("\n".join(sorted(cell_ids[validation])).encode()).hexdigest(),
            },
        }
    checkpoint = {
        "format_version": 1,
        "weights": torch.from_numpy(solution[:-1].astype(np.float32)),
        "bias": torch.from_numpy(solution[-1].astype(np.float32)),
        "latent_mean": torch.from_numpy(latent_mean),
        "latent_scale": torch.from_numpy(latent_scale),
        "output_clamp_min": decoder["output_clamp_min"],
        "report": report,
        "provenance": {
            "config_sha256": file_sha256("configs/readout.yaml"),
            "latent_cache_sha256": inputs["latent_cache_sha256"],
            "expression_cache_sha256": file_sha256(config["expression_cache"]["output_path"]),
            "replogle_manifest_sha256": inputs["replogle_manifest_sha256"],
            "dynamics_manifest_sha256": inputs["dynamics_manifest_sha256"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    assert torch.isfinite(checkpoint["weights"]).all() and torch.isfinite(checkpoint["bias"]).all()
    return checkpoint


def decode_normalized_latents(latents, checkpoint):
    """Decode dynamics-space latents to nonnegative normalized log expression."""
    return (latents @ checkpoint["weights"] + checkpoint["bias"]).clamp_min(
        checkpoint["output_clamp_min"]
    )
