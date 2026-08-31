"""Leakage-safe AnnData export for the official State transition baseline."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .dynamics import LatentPopulationDataset
from .resources import file_sha256
from .training import _git_state, _runtime_environment, _runtime_source_hash


def state_export_indices(
    roles,
    targets,
    allowed_roles,
    maximum_conditions_per_role=None,
    maximum_cells_per_condition=None,
):
    """Select only declared fit roles, optionally retaining a bounded condition sample."""
    roles, targets = np.asarray(roles), np.asarray(targets)
    assert roles.shape == targets.shape and len(set(allowed_roles)) == len(allowed_roles)
    if maximum_conditions_per_role is None:
        assert maximum_cells_per_condition is None
        return np.flatnonzero(np.isin(roles, allowed_roles))
    assert maximum_conditions_per_role > 0 and maximum_cells_per_condition > 0
    selected = []
    for role in allowed_roles:
        candidates = np.flatnonzero(roles == role)
        assert len(candidates)
        condition_names = sorted(set(targets[candidates]))[:maximum_conditions_per_role]
        for target in condition_names:
            condition = candidates[targets[candidates] == target]
            selected.extend(condition[:maximum_cells_per_condition])
    return np.asarray(sorted(selected), dtype=np.int64)


def _self_hashed_manifest(path, expected):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    assert declared == expected == sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _validate_artifact(path, size, digest):
    path = Path(path)
    assert (path.stat().st_size, file_sha256(path)) == (size, digest)
    return path


def _state_adata(expression, indices, metadata, gene_names, gene_ids):
    values = np.asarray(expression[indices], dtype=np.float32)
    assert values.shape == (len(indices), len(gene_names)) and np.isfinite(values).all()
    # cell-load 0.10.4 reads var/gene_name directly as an HDF5 dataset. Pandas 3
    # infers its nullable StringDtype by default, which AnnData 0.13 serializes as
    # a group instead. Keep the legacy object-string representation State expects.
    with pd.option_context("future.infer_string", False):
        obs = pd.DataFrame(
            {
                "gene": pd.Categorical(metadata["target"][indices]),
                "cell_type": pd.Categorical(metadata["context"][indices]),
                "gem_group": pd.Categorical(metadata["source_batch"][indices]),
                "role": pd.Categorical(metadata["role"][indices]),
                "source_row": metadata["source_row"][indices],
            },
            index=pd.Index(metadata["cell_id"][indices], name="cell_id"),
        )
        var = pd.DataFrame(
            {"gene_id": np.asarray(gene_ids, dtype=object)},
            index=pd.Index(np.asarray(gene_names, dtype=object), name="gene_name"),
        )
        result = ad.AnnData(
            X=sp.csr_matrix((len(indices), len(gene_names)), dtype=np.float32),
            obs=obs,
            var=var,
            obsm={"X_hvg": values},
        )
    result.uns["log1p"] = {"base": None}
    return result


def _write_state_var_schema(path, gene_names, gene_ids):
    """Normalize var to the flat H5AD strings required by cell-load 0.10.4."""
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "r+") as handle:
        del handle["var"]
        var = handle.create_group("var")
        var.attrs["_index"] = "gene_name"
        var.attrs["column-order"] = np.asarray(["gene_id"], dtype=string)
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        for name, values in (("gene_name", gene_names), ("gene_id", gene_ids)):
            dataset = var.create_dataset(name, data=np.asarray(values, dtype=object), dtype=string)
            dataset.attrs["encoding-type"] = "string-array"
            dataset.attrs["encoding-version"] = "0.2.0"


def _write_state_uns_schema(path):
    """Restore the log1p marker that concat_on_disk omits from its output."""
    with h5py.File(path, "r+") as handle:
        if "uns" in handle:
            del handle["uns"]
        uns = handle.create_group("uns")
        uns.attrs["encoding-type"] = "dict"
        uns.attrs["encoding-version"] = "0.1.0"
        log1p = uns.create_group("log1p")
        log1p.attrs["encoding-type"] = "dict"
        log1p.attrs["encoding-version"] = "0.1.0"
        base = log1p.create_dataset("base", shape=None, dtype=np.float32)
        base.attrs["encoding-type"] = "null"
        base.attrs["encoding-version"] = "0.1.0"


def _write_state_h5ad(path, expression, indices, metadata, gene_names, gene_ids, chunk_cells):
    path = Path(path)
    assert len(indices) and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(indices) <= chunk_cells:
        _state_adata(expression, indices, metadata, gene_names, gene_ids).write_h5ad(
            path, compression="lzf"
        )
    else:
        with TemporaryDirectory(dir=path.parent) as temporary:
            chunks = []
            for start in range(0, len(indices), chunk_cells):
                chunk_path = Path(temporary) / f"{start:09d}.h5ad"
                _state_adata(
                    expression,
                    indices[start : start + chunk_cells],
                    metadata,
                    gene_names,
                    gene_ids,
                ).write_h5ad(chunk_path, compression="lzf")
                chunks.append(chunk_path)
            ad.experimental.concat_on_disk(chunks, path, uns_merge="same")
    # concat_on_disk currently rewrites the shared var index as a nullable-string
    # group and drops columns. Normalize both code paths to the frozen State schema.
    _write_state_var_schema(path, gene_names, gene_ids)
    _write_state_uns_schema(path)
    exported = ad.read_h5ad(path, backed="r")
    try:
        assert exported.shape == (len(indices), len(gene_names))
        assert exported.obsm["X_hvg"].shape == (len(indices), len(gene_names))
        assert list(exported.var_names) == list(gene_names)
    finally:
        exported.file.close()
    with h5py.File(path, "r") as handle:
        assert isinstance(handle["var/gene_name"], h5py.Dataset)
        assert isinstance(handle["var/gene_id"], h5py.Dataset)
        assert isinstance(handle["uns/log1p"], h5py.Group)


def _write_action_features(path, action):
    assert action["embedding"].shape[0] == len(action["targets"])
    assert action["embedding"].shape[1] == sum(action["modality_dims"]) + len(
        action["modality_dims"]
    )
    features = {
        target: embedding.detach().float().clone()
        for target, embedding in zip(action["targets"], action["embedding"], strict=True)
    }
    features["non-targeting"] = torch.zeros(action["embedding"].shape[1])
    assert len(features) == len(action["targets"]) + 1
    assert all(torch.isfinite(value).all() for value in features.values())
    torch.save(features, path)


def _write_training_toml(path, training_directory, validation_targets):
    lines = [
        "[datasets]",
        f"causalcelljepa = {json.dumps(str(training_directory))}",
        "",
        "[training]",
        'causalcelljepa = "train"',
        "",
        "[zeroshot]",
        "",
        "[fewshot]",
        '[fewshot."causalcelljepa.K562"]',
        f"val = {json.dumps(validation_targets)}",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def _artifact(path, records=None):
    path = Path(path)
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if records is not None:
        result["records"] = records
    return result


def prepare_state_baseline(config, output_directory=None, smoke=None):
    """Export fit-only expression, control templates, and shared biological actions."""
    inputs, export = config["inputs"], config["export"]
    expression_path = _validate_artifact(
        inputs["expression_cache_path"],
        inputs["expression_cache_bytes"],
        inputs["expression_cache_sha256"],
    )
    latent_path = _validate_artifact(
        inputs["latent_cache_path"], inputs["latent_cache_bytes"], inputs["latent_cache_sha256"]
    )
    action_path = _validate_artifact(
        inputs["action_cache_path"], inputs["action_cache_bytes"], inputs["action_cache_sha256"]
    )
    replogle = _self_hashed_manifest(
        inputs["replogle_manifest_path"], inputs["replogle_manifest_sha256"]
    )
    action_manifest = _self_hashed_manifest(
        inputs["action_manifest_path"], inputs["action_manifest_sha256"]
    )
    assert action_manifest["artifact"]["sha256"] == inputs["action_cache_sha256"]
    gene_names = replogle["genes"]["hvg_gene_names"]
    gene_ids = replogle["genes"]["hvg_gene_ids"]
    assert len(gene_names) == len(gene_ids) == export["hvg_count"]
    output = Path(output_directory or export["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)

    limits = smoke or {}
    maximum_conditions = limits.get("maximum_conditions_per_role")
    maximum_cells = limits.get("maximum_cells_per_condition")
    with h5py.File(latent_path, "r") as latent, h5py.File(expression_path, "r") as expression:
        assert latent["latent"].shape[0] == expression["expression"].shape[0]
        assert expression["expression"].shape[1] == export["hvg_count"]
        metadata = {
            name: latent[name].asstr()[:]
            for name in ("cell_id", "context", "role", "source_batch", "target")
        }
        metadata["source_row"] = latent["source_row"][:]
        fit_indices = state_export_indices(
            metadata["role"],
            metadata["target"],
            export["training_roles"],
            maximum_conditions,
            maximum_cells,
        )
        forbidden = set(export["forbidden_training_roles"])
        assert not forbidden.intersection(metadata["role"][fit_indices])
        assert set(metadata["role"][fit_indices]) == set(export["training_roles"])
        assert not np.any(metadata["context"][fit_indices] == "RPE1")
        control_limit = maximum_cells if smoke else None
        control_indices = {}
        for context, role in (("K562", "control_train"), ("RPE1", "control_inference")):
            indices = np.flatnonzero(
                (metadata["context"] == context) & (metadata["role"] == role)
            )
            control_indices[context] = indices[:control_limit]
            assert len(control_indices[context])
        files = {
            "training": (export["training_file"], fit_indices),
            "k562_controls": (export["k562_control_file"], control_indices["K562"]),
            "rpe1_controls": (export["rpe1_control_file"], control_indices["RPE1"]),
        }
        file_reports = {}
        for name, (relative, indices) in files.items():
            path = output / relative
            _write_state_h5ad(
                path,
                expression["expression"],
                indices,
                metadata,
                gene_names,
                gene_ids,
                export["chunk_cells"],
            )
            file_reports[name] = {
                **_artifact(path, len(indices)),
                "role_counts": dict(sorted(Counter(metadata["role"][indices]).items())),
                "context_counts": dict(sorted(Counter(metadata["context"][indices]).items())),
                "source_indices_sha256": sha256(indices.tobytes()).hexdigest(),
            }

    action = torch.load(action_path, map_location="cpu", weights_only=True)
    feature_path = output / export["perturbation_features_file"]
    _write_action_features(feature_path, action)
    validation_targets = replogle["targets"]["split"]["targets"]["validation"]
    toml_path = output / export["training_toml_file"]
    _write_training_toml(toml_path, output / Path(export["training_file"]).parent, validation_targets)
    report = {
        "format_version": 1,
        "source": config["source"],
        "artifacts": {
            **file_reports,
            "perturbation_features": _artifact(feature_path, len(action["targets"]) + 1),
            "training_toml": _artifact(toml_path),
        },
        "split": {
            "training_roles": export["training_roles"],
            "forbidden_training_roles": export["forbidden_training_roles"],
            "validation_targets": len(validation_targets),
            "sealed_test_outcomes_exported_for_training": False,
            "rpe1_controls_exported_for_training": False,
            "rpe1_perturbed_outcomes_exported_for_training": False,
            "statistical_unit": "perturbation-condition",
        },
        "features": {
            "dimensions": int(action["embedding"].shape[1]),
            "targets": len(action["targets"]),
            "control_is_zero": True,
            "outcomes_read": False,
        },
        "smoke_limits": smoke,
        "provenance": {
            "inputs": {
                name: inputs[f"{name}_sha256"]
                for name in ("expression_cache", "latent_cache", "action_cache")
            },
            "replogle_manifest_sha256": inputs["replogle_manifest_sha256"],
            "action_manifest_sha256": inputs["action_manifest_sha256"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    report_path = output / export["report_file"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def prepare_state_prediction_metadata(config, output_path=None):
    """Compress the audited split fields needed for remote control-population sampling."""
    inputs, prediction = config["inputs"], config["prediction"]
    latent_path = _validate_artifact(
        inputs["latent_cache_path"], inputs["latent_cache_bytes"], inputs["latent_cache_sha256"]
    )
    output = Path(output_path or prediction["metadata_path"])
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(latent_path, "r") as source, h5py.File(output, "w") as destination:
        for name in ("context", "role", "source_batch", "target"):
            values = source[name].asstr()[:]
            destination.create_dataset(
                name,
                data=np.asarray(values, dtype=f"S{max(map(len, values))}"),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
        destination.create_dataset(
            "source_row",
            data=source["source_row"][:],
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        destination.attrs["source_latent_sha256"] = inputs["latent_cache_sha256"]
    return _artifact(output, len(values))


@torch.inference_mode()
def predict_state_baseline(
    config,
    base_config,
    model,
    output_path=None,
    device=None,
    regimes=None,
    repeats=None,
    maximum_conditions=None,
):
    """Generate test effects from controls/actions only with a validation-frozen State model."""
    device = device or torch.device("cpu")
    inputs, prediction = config["inputs"], config["prediction"]
    assert file_sha256(config["base_transcriptomics_config_path"]) == config[
        "base_transcriptomics_config_sha256"
    ]
    for name in ("action_cache", "state_features"):
        assert file_sha256(inputs[f"{name}_path"]) == inputs[f"{name}_sha256"]
    metadata_path = _validate_artifact(
        prediction["metadata_path"], prediction["metadata_bytes"], prediction["metadata_sha256"]
    )
    for artifact in prediction["control_files"].values():
        path = Path(artifact["path"])
        assert (path.stat().st_size, file_sha256(path)) == (
            artifact["bytes"],
            artifact["sha256"],
        )
    assert (
        model.input_dim,
        model.output_dim,
        model.pert_dim,
        model.cell_sentence_len,
        model.batch_encoder,
        model.use_batch_token,
    ) == (3000, 3000, prediction["action_dimensions"], prediction["population_size"], None, False)
    model = model.to(device).eval()
    regimes = regimes or base_config["regimes"]
    repeats = repeats or prediction["repeats"]
    features = torch.load(inputs["state_features_path"], map_location="cpu", weights_only=True)
    output = Path(output_path or prediction["output_path"])
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_controls = sha256()
    rows = expected = 0
    string = h5py.string_dtype("utf-8")
    with h5py.File(metadata_path, "r") as metadata, h5py.File(
        prediction["control_files"]["K562"]["path"], "r"
    ) as k562_controls, h5py.File(
        prediction["control_files"]["RPE1"]["path"], "r"
    ) as rpe1_controls, h5py.File(output, "w") as destination:
        roles = metadata["role"].asstr()[:]
        assert metadata.attrs["source_latent_sha256"] == inputs["latent_cache_sha256"]
        control_files = {"K562": k562_controls, "RPE1": rpe1_controls}
        control_source_rows = {
            context: handle["obs/source_row"][:] for context, handle in control_files.items()
        }
        assert all(
            len(rows) == len(np.unique(rows)) and np.all(rows[1:] > rows[:-1])
            for rows in control_source_rows.values()
        )
        effect = destination.create_dataset(
            "predicted_effect",
            (0, model.output_dim),
            maxshape=(None, model.output_dim),
            dtype="f4",
            chunks=(prediction["batch_size"], model.output_dim),
            compression="lzf",
        )
        regime_values = destination.create_dataset("regime", (0,), maxshape=(None,), dtype=string)
        target_values = destination.create_dataset("target", (0,), maxshape=(None,), dtype=string)
        repeat_values = destination.create_dataset("repeat", (0,), maxshape=(None,), dtype="i2")
        for regime, specification in regimes.items():
            dataset = LatentPopulationDataset(
                metadata_path,
                inputs["action_cache_path"],
                inputs["dynamics_manifest_path"],
                regime,
                prediction["population_size"],
                base_config["seed"],
                specification["outcome_role"],
                specification["control_role"],
                specification["context"],
            )
            indices = list(range(min(len(dataset), maximum_conditions or len(dataset))))
            expected += repeats * len(indices)
            for repeat in range(repeats):
                dataset.set_epoch(repeat)
                for start in range(0, len(indices), prediction["batch_size"]):
                    plans = [
                        dataset.sample_indices(index)
                        for index in indices[start : start + prediction["batch_size"]]
                    ]
                    controls, targets = [], []
                    for control_indices, _, target in plans:
                        assert set(roles[control_indices]) == {specification["control_role"]}
                        index_order = np.argsort(control_indices)
                        source_rows = metadata["source_row"][control_indices[index_order]][
                            np.argsort(index_order)
                        ]
                        control_rows = control_source_rows[specification["context"]]
                        positions = np.searchsorted(control_rows, source_rows)
                        assert np.array_equal(control_rows[positions], source_rows)
                        order = np.argsort(positions)
                        values = control_files[specification["context"]]["obsm/X_hvg"][
                            positions[order]
                        ][np.argsort(order)]
                        controls.append(values)
                        targets.append(target)
                        selected_controls.update(
                            f"{regime}\0{repeat}\0{target}\0".encode() + control_indices.tobytes()
                        )
                    control = torch.from_numpy(np.stack(controls)).to(device)
                    action = torch.stack([features[target] for target in targets]).to(device)
                    action = action[:, None].expand(-1, prediction["population_size"], -1)
                    predicted = model(
                        {
                            "ctrl_cell_emb": control.flatten(0, 1),
                            "pert_emb": action.flatten(0, 1),
                        },
                        padded=True,
                    )
                    predicted = predicted.reshape(len(targets), prediction["population_size"], -1)
                    values = (predicted.mean(1) - control.mean(1)).float().cpu().numpy()
                    assert values.shape == (len(targets), model.output_dim) and np.isfinite(values).all()
                    stop = rows + len(targets)
                    for dataset_value in (effect, regime_values, target_values, repeat_values):
                        dataset_value.resize(stop, axis=0)
                    effect[rows:stop] = values
                    regime_values[rows:stop] = [regime] * len(targets)
                    target_values[rows:stop] = targets
                    repeat_values[rows:stop] = repeat
                    rows = stop
        destination.attrs.update(
            {
                "format_version": 1,
                "model": prediction["model_name"],
                "checkpoint_sha256": inputs["checkpoint_sha256"],
                "metadata_sha256": prediction["metadata_sha256"],
                "controls_sha256": selected_controls.hexdigest(),
                "test_outcome_expression_read": False,
            }
        )
    assert rows == expected
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "records": rows,
        "repeats": repeats,
        "controls_sha256": selected_controls.hexdigest(),
        "metadata_sha256": prediction["metadata_sha256"],
        "test_outcome_expression_read": False,
    }
