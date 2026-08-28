# Frozen adapters for test-only cross-dataset validation on normalized Perturb-seq files.
# Preparation reads metadata only; expression values enter solely during frozen inference.
import json
from collections import Counter
from hashlib import file_digest, sha256
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
import yaml
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from causalcelljepa.model import load_frozen_teacher
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


def tokenize_normalized_cell(values, vocab_positions, vocab_size=3000, max_tokens=512):
    """Tokenize already-log-normalized values while retaining frozen vocabulary IDs."""
    values = np.asarray(values, dtype=np.float32).ravel()
    vocab_positions = np.asarray(vocab_positions, dtype=np.int64)
    assert (
        values.shape == vocab_positions.shape and np.isfinite(values).all() and (values >= 0).all()
    )
    nonzero = np.flatnonzero(values)
    assert nonzero.size
    selected = nonzero[np.lexsort((vocab_positions[nonzero], -values[nonzero]))[:max_tokens]]
    gene_ids = np.full(max_tokens, vocab_size, dtype=np.int64)
    expression = np.zeros(max_tokens, dtype=np.float32)
    padding_mask = np.ones(max_tokens, dtype=bool)
    gene_ids[: len(selected)] = vocab_positions[selected]
    expression[: len(selected)] = values[selected]
    padding_mask[: len(selected)] = False
    return gene_ids, expression, padding_mask


def prepare_nadig_external(config_path="configs/nadig_external_validation.yaml"):
    """Audit source metadata and freeze eligible targets without reading expression values."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    inputs = config["inputs"]
    action_path = Path(inputs["action_cache_path"])
    assert (action_path.stat().st_size, file_sha256(action_path)) == (
        inputs["action_cache_bytes"],
        inputs["action_cache_sha256"],
    )
    action = torch.load(action_path, map_location="cpu", weights_only=True)
    known_targets = {
        target for target, known in zip(action["targets"], action["known"]) if bool(known)
    }
    replogle = json.loads(Path(inputs["replogle_manifest_path"]).read_text())
    declared = replogle.pop("manifest_sha256")
    assert (
        declared
        == inputs["replogle_manifest_sha256"]
        == sha256(json.dumps(replogle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    hvg_names = replogle["genes"]["hvg_gene_names"]
    contexts = {}
    for context, source in config["source"]["files"].items():
        path = Path(config["source"]["raw_directory"]) / source["filename"]
        assert path.stat().st_size == source["bytes"]
        with path.open("rb") as handle:
            assert file_digest(handle, "md5").hexdigest() == source["md5"]
        data = ad.read_h5ad(path, backed="r")
        assert data.n_obs == source["expected_cells"] and list(data.obs.columns) == [
            config["source"]["perturbation_column"]
        ]
        assert data.obs_names.is_unique and data.var_names.is_unique and not list(data.var.columns)
        labels = data.obs[config["source"]["perturbation_column"]].astype(str).to_numpy()
        counts = Counter(labels)
        assert counts[config["source"]["control_label"]] == source["expected_controls"]
        assert len(counts) - 1 == source["expected_perturbations"]
        eligible = sorted(
            target
            for target, count in counts.items()
            if target != config["source"]["control_label"]
            and count >= config["protocol"]["population_size"]
            and target in known_targets
        )
        present_hvgs = [gene for gene in hvg_names if gene in data.var_names]
        contexts[context] = {
            "cells": data.n_obs,
            "genes": data.n_vars,
            "controls": counts[config["source"]["control_label"]],
            "perturbations": len(counts) - 1,
            "eligible_targets": eligible,
            "eligible_target_sha256": sha256("\n".join(eligible).encode()).hexdigest(),
            "eligible_outcome_cells": sum(counts[target] for target in eligible),
            "admitted_cells": counts[config["source"]["control_label"]]
            + sum(counts[target] for target in eligible),
            "frozen_hvg_overlap": len(present_hvgs),
            "frozen_hvg_overlap_sha256": sha256("\n".join(present_hvgs).encode()).hexdigest(),
            "source_sha256": file_sha256(path),
        }
        data.file.close()
    report = {
        "format_version": 1,
        "dataset": config["source"],
        "protocol": config["protocol"],
        "contexts": contexts,
        "leakage": {
            "expression_outcomes_read_during_preparation": False,
            "external_cells_used_for_fit_or_selection": False,
            "roles": ["external_control", "external_test"],
        },
        "source": {
            "config_sha256": file_sha256(config_path),
            "replogle_manifest_sha256": declared,
            "action_cache_sha256": inputs["action_cache_sha256"],
        },
        "provenance": {
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    report["manifest_sha256"] = sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    Path(config["source"]["manifest_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


class NadigExternalTokenDataset(Dataset):
    """Expose only frozen-eligible external cells as normalized JEPA tokens."""

    def __init__(
        self, config_path="configs/nadig_external_validation.yaml", maximum_per_context=None
    ):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.manifest = json.loads(Path(self.config["source"]["manifest_path"]).read_text())
        declared = self.manifest.pop("manifest_sha256")
        assert (
            declared
            == sha256(
                json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        replogle = json.loads(Path(self.config["inputs"]["replogle_manifest_path"]).read_text())
        hvg_names = replogle["genes"]["hvg_gene_names"]
        self.paths, self.source_columns, self.vocab_positions, self.samples, self._backed = (
            {},
            {},
            {},
            [],
            {},
        )
        for context in self.config["protocol"]["contexts"]:
            source = self.config["source"]["files"][context]
            path = Path(self.config["source"]["raw_directory"]) / source["filename"]
            data = ad.read_h5ad(path, backed="r")
            labels = data.obs[self.config["source"]["perturbation_column"]].astype(str).to_numpy()
            eligible = set(self.manifest["contexts"][context]["eligible_targets"])
            controls = np.flatnonzero(labels == self.config["source"]["control_label"])
            outcomes = np.flatnonzero(np.isin(labels, sorted(eligible)))
            admitted = np.concatenate((controls, outcomes))
            if maximum_per_context is not None:
                half = maximum_per_context // 2
                admitted = np.concatenate((controls[:half], outcomes[: maximum_per_context - half]))
            for row in np.sort(admitted):
                target = str(labels[row])
                role = (
                    "external_control"
                    if target == self.config["source"]["control_label"]
                    else "external_test"
                )
                self.samples.append(
                    (context, int(row), f"{context}|{data.obs_names[row]}", role, target)
                )
            pairs = [
                (data.var_names.get_loc(gene), index)
                for index, gene in enumerate(hvg_names)
                if gene in data.var_names
            ]
            pairs.sort()
            self.paths[context] = path
            self.source_columns[context] = np.asarray([pair[0] for pair in pairs])
            self.vocab_positions[context] = np.asarray([pair[1] for pair in pairs])
            data.file.close()
        if maximum_per_context is None:
            assert len(self.samples) == sum(
                item["admitted_cells"] for item in self.manifest["contexts"].values()
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, row, cell_id, role, target = self.samples[index]
        if context not in self._backed:
            self._backed[context] = ad.read_h5ad(self.paths[context], backed="r")
        # Slice one backed CSR row first; mixed row/column indexing materializes the full matrix.
        values = self._backed[context].X[row][:, self.source_columns[context]]
        values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
        gene_ids, expression, padding_mask = tokenize_normalized_cell(
            values, self.vocab_positions[context]
        )
        return {
            "gene_ids": torch.from_numpy(gene_ids),
            "values": torch.from_numpy(expression),
            "padding_mask": torch.from_numpy(padding_mask),
            "cell_id": cell_id,
            "context": context,
            "target": target,
            "role": role,
            "source_batch": "unavailable",
            "source_row": row,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_backed"] = {}
        return state


def write_nadig_latent_cache(
    config_path="configs/nadig_external_validation.yaml",
    output_path=None,
    maximum_per_context=None,
    device="cuda",
):
    """Encode frozen-eligible external cells without updating any model parameter."""
    config = yaml.safe_load(Path(config_path).read_text())
    cache_config = config["latent_cache"]
    device = torch.device(device)
    if device.type == "cuda":
        assert torch.cuda.is_available()
    dataset = NadigExternalTokenDataset(config_path, maximum_per_context)
    teacher_path = Path(cache_config["teacher_path"])
    assert file_sha256(teacher_path) == cache_config["teacher_sha256"]
    teacher, teacher_payload = load_frozen_teacher(teacher_path, 3000, device)
    workers = 0 if maximum_per_context is not None else cache_config["num_workers"]
    batch_size = min(cache_config["batch_size"], 16) if maximum_per_context else cache_config["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    output = Path(output_path or cache_config["output_path"])
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    assert not temporary.exists()
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as cache:
        cache.attrs.update(
            {
                "format_version": 1,
                "cells": len(dataset),
                "cell_dim": teacher_payload["cell_dim"],
                "dtype": cache_config["dtype"],
                "teacher_sha256": cache_config["teacher_sha256"],
                "external_manifest_sha256": json.loads(
                    Path(config["source"]["manifest_path"]).read_text()
                )["manifest_sha256"],
                "provenance_json": json.dumps(
                    {
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
            chunks=(min(cache_config["chunk_cells"], len(dataset)), teacher_payload["cell_dim"]),
        )
        metadata = {
            name: cache.create_dataset(name, (len(dataset),), dtype=string)
            for name in ("cell_id", "context", "target", "role", "source_batch")
        }
        rows = cache.create_dataset("source_row", (len(dataset),), dtype="i8")
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                end = offset + len(batch["cell_id"])
                latent = teacher(
                    batch["gene_ids"].to(device),
                    batch["values"].to(device),
                    batch["padding_mask"].to(device),
                )
                assert torch.isfinite(latent).all()
                latents[offset:end] = latent.cpu().numpy()
                for name, destination in metadata.items():
                    destination[offset:end] = batch[name]
                rows[offset:end] = batch["source_row"].numpy()
                offset = end
        assert offset == len(dataset)
        cache.flush()
    temporary.replace(output)
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
        "cells": len(dataset),
        "device": str(device),
    }
