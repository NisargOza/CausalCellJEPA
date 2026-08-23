# Validate the official raw AnnData files and freeze the primary split/HVG manifest.
# The dense count matrices stay backed on disk so preprocessing does not duplicate them.
import json
import platform
from collections import Counter
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import yaml

from causalcelljepa.data import (
    assign_roles,
    eligible_targets,
    fit_hvgs_stream,
    representation_fit_mask,
    target_manifest,
)

OBS_COLUMNS = [
    "gem_group",
    "gene",
    "gene_id",
    "transcript",
    "gene_transcript",
    "sgID_AB",
    "mitopercent",
    "UMI_count",
    "z_gemgroup_UMI",
    "core_scale_factor",
    "core_adjusted_UMI_count",
]
VAR_COLUMNS = [
    "gene_name",
    "chr",
    "start",
    "end",
    "class",
    "strand",
    "length",
    "in_matrix",
    "mean",
    "std",
    "cv",
    "fano",
]
EXPECTED = {"K562": (10_691, 2_057, 48, 0.20), "RPE1": (11_485, 2_393, 56, 0.11)}
config_path = Path("configs/replogle.yaml")
config = yaml.safe_load(config_path.read_text())
datasets, source_report = {}, {}
for context, source in config["data"]["files"].items():
    path = Path("data/raw") / source["filename"]
    assert path.stat().st_size == source["bytes"]
    with path.open("rb") as handle:
        local_sha256 = file_digest(handle, "sha256").hexdigest()
    assert local_sha256 == source["sha256"]
    data = ad.read_h5ad(path, backed="r")
    assert data.shape == tuple(source["shape"])
    assert data.obs.index.name == "cell_barcode" and data.obs_names.is_unique
    assert data.var.index.name == "gene_id" and data.var_names.is_unique
    assert list(data.obs.columns) == OBS_COLUMNS and list(data.var.columns) == VAR_COLUMNS
    targets = data.obs["gene"].astype(str).to_numpy()
    target_ids = data.obs["gene_id"].astype(str).to_numpy()
    controls = targets == "non-targeting"
    assert controls.any() and not data.obs["gene"].isna().any()
    gems = data.obs["gem_group"].to_numpy()
    control_batches = Counter(gems[controls].tolist())
    expected_controls, expected_targets, expected_gems, mitochondrial_limit = EXPECTED[context]
    assert int(controls.sum()) == expected_controls
    assert np.unique(targets[~controls]).size == expected_targets
    assert set(gems) == set(range(1, expected_gems + 1))
    assert set(gems) == set(control_batches)
    assert min(control_batches.values()) >= config["split"]["population_size"]
    assert data.obs["core_adjusted_UMI_count"].min() > 3_000
    assert data.obs["mitopercent"].max() < mitochondrial_limit
    datasets[context] = {
        "data": data,
        "targets": targets,
        "target_ids": target_ids,
        "controls": controls,
        "gems": gems,
    }
    source_report[context] = {
        "file_id": source["file_id"],
        "filename": source["filename"],
        "experiment": source["experiment"],
        "timepoint_days": source["timepoint_days"],
        "bytes": path.stat().st_size,
        "upstream_md5": source["md5"],
        "local_sha256": local_sha256,
        "shape": list(data.shape),
        "controls": int(controls.sum()),
        "perturbed": int((~controls).sum()),
        "target_labels": int(np.unique(targets[~controls]).size),
        "gem_groups": len(control_batches),
        "qc": {
            "minimum_core_adjusted_UMI_count": float(data.obs["core_adjusted_UMI_count"].min()),
            "maximum_mitopercent": float(data.obs["mitopercent"].max()),
        },
        "controls_per_gem_group": {
            "minimum": min(control_batches.values()),
            "median": float(np.median(list(control_batches.values()))),
            "maximum": max(control_batches.values()),
            "counts": {str(gem): control_batches[gem] for gem in sorted(control_batches)},
        },
    }

targets = np.concatenate([datasets[context]["targets"] for context in config["data"]["contexts"]])
contexts = np.concatenate(
    [
        np.full(datasets[context]["targets"].size, context, dtype=object)
        for context in config["data"]["contexts"]
    ]
)
controls = np.concatenate([datasets[context]["controls"] for context in config["data"]["contexts"]])
eligible = eligible_targets(
    targets,
    contexts,
    controls,
    tuple(config["data"]["contexts"]),
    config["data"]["min_condition_cells"],
)
split = target_manifest(eligible, config["seed"], config["split"]["fractions"])
condition_counts = Counter(zip(targets[~controls].tolist(), contexts[~controls].tolist()))
target_id_maps = {
    context: dict(zip(datasets[context]["targets"], datasets[context]["target_ids"]))
    for context in config["data"]["contexts"]
}
assert all(
    target_id_maps["K562"][target] == target_id_maps["RPE1"][target] != "nan" for target in eligible
)

common_gene_ids = sorted(
    set.intersection(*(set(entry["data"].var_names) for entry in datasets.values()))
)
assert len(common_gene_ids) >= config["data"]["hvg_count"]
common_gene_names, sources, role_report = None, [], {}
for context in config["data"]["contexts"]:
    entry = datasets[context]
    cell_ids = np.asarray(
        [
            f"{context}|{gem}|{barcode}"
            for gem, barcode in zip(entry["gems"], entry["data"].obs_names)
        ]
    )
    roles = assign_roles(
        cell_ids,
        entry["targets"],
        np.full(cell_ids.size, context),
        entry["controls"],
        split,
        config["split"]["iid_test_fraction"],
        config["split"]["population_size"],
    )
    columns = np.asarray([entry["data"].var_names.get_loc(gene_id) for gene_id in common_gene_ids])
    names = entry["data"].var.iloc[columns]["gene_name"].astype(str).tolist()
    common_gene_names = names if common_gene_names is None else common_gene_names
    assert names == common_gene_names
    sources.append(
        (
            entry["data"].X,
            representation_fit_mask(roles),
            columns,
            entry["data"].obs["UMI_count"].to_numpy(),
        )
    )
    role_report[context] = dict(sorted(Counter(roles).items()))

hvg_positions, count_diagnostics, fit_cells = fit_hvgs_stream(
    sources, config["data"]["hvg_count"], config["data"]["library_size"]
)
assert all(item["zero_library_cells"] == 0 for item in count_diagnostics)
hvg_gene_ids = [common_gene_ids[index] for index in hvg_positions]
hvg_gene_names = [common_gene_names[index] for index in hvg_positions]
report = {
    "dataset": {
        "article": config["data"]["source_article"],
        "version": config["data"]["source_version"],
    },
    "runtime": {
        "python": platform.python_version(),
        "anndata": version("anndata"),
        "numpy": np.__version__,
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
    },
    "sources": source_report,
    "schema": {
        "obs": OBS_COLUMNS,
        "var": VAR_COLUMNS,
        "target": "gene",
        "control": "non-targeting",
        "batch": "gem_group",
    },
    "targets": {
        "minimum_cells_per_context": config["data"]["min_condition_cells"],
        "eligible": len(eligible),
        "condition_counts": {
            target: {
                context: condition_counts[target, context] for context in config["data"]["contexts"]
            }
            for target in eligible
        },
        "split": split,
    },
    "roles": role_report,
    "genes": {
        "common": len(common_gene_ids),
        "hvg_method": config["data"]["hvg_method"],
        "hvg_count": len(hvg_gene_ids),
        "hvg_gene_ids": hvg_gene_ids,
        "hvg_gene_names": hvg_gene_names,
        "hvg_sha256": sha256("\n".join(hvg_gene_ids).encode()).hexdigest(),
        "fit_cells": fit_cells,
    },
    "raw_count_diagnostics": dict(zip(config["data"]["contexts"], count_diagnostics)),
}
report["manifest_sha256"] = sha256(
    json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path("manifests/replogle_v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
for entry in datasets.values():
    entry["data"].file.close()
print(
    "eligible_targets",
    len(eligible),
    "split",
    {name: len(group) for name, group in split["targets"].items()},
)
print("manifest_sha256", report["manifest_sha256"])
