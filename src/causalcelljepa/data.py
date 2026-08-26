# Leakage-resistant data primitives for the fixed Replogle experiment.
# Model code comes later and must remain separate from download/preprocessing code.
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import yaml
from scipy import sparse
from torch.utils.data import Dataset


def normalize_log1p(counts, library_size=10_000):
    """Library-size normalize raw cell-by-gene counts and apply log1p."""
    normalized = counts.astype(np.float32, copy=True)
    totals = np.asarray(normalized.sum(axis=1)).ravel()
    with np.errstate(divide="raise", invalid="raise"):
        scale = library_size / totals
    if sparse.issparse(normalized):
        normalized = normalized.multiply(scale[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
    else:
        normalized = np.log1p(normalized * scale[:, None])
    return normalized


def fit_hvgs(counts, fit_mask, n_genes=3_000, library_size=10_000):
    """Fit a fixed vocabulary by log-expression variance on allowed cells only."""
    expression = normalize_log1p(counts[np.asarray(fit_mask)], library_size)
    if sparse.issparse(expression):
        mean = np.asarray(expression.mean(axis=0)).ravel()
        variance = np.asarray(expression.power(2).mean(axis=0)).ravel() - mean**2
    else:
        variance = expression.var(axis=0)
    ranked = np.lexsort((np.arange(variance.size), -variance))[:n_genes]
    return np.sort(ranked)


def fit_hvgs_stream(sources, n_genes=3_000, library_size=10_000, block_size=1_024):
    """Fit aligned HVGs from backed matrices while validating every raw count."""
    width = len(sources[0][2])
    total = 0
    sums = np.zeros(width, dtype=np.float64)
    squares = np.zeros(width, dtype=np.float64)
    diagnostics = []
    for counts, fit_mask, columns, observed_totals in sources:
        minimum, maximum, max_excluded, zero_cells = np.inf, -np.inf, 0.0, 0
        for start in range(0, counts.shape[0], block_size):
            block = np.asarray(counts[start : start + block_size])
            assert (
                np.isfinite(block).all()
                and (block >= 0).all()
                and np.equal(block, np.floor(block)).all()
            )
            totals = block.sum(axis=1)
            recorded = np.asarray(observed_totals[start : start + block_size])
            assert (totals <= recorded + 1e-4).all()
            max_excluded = max(max_excluded, float(np.max(recorded - totals)))
            zero_cells += int(np.count_nonzero(totals == 0))
            minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
            selected = block[np.asarray(fit_mask[start : start + block_size])]
            if selected.size:
                with np.errstate(divide="raise", invalid="raise"):
                    expression = np.log1p(
                        selected[:, columns] * (library_size / selected.sum(axis=1))[:, None]
                    )
                sums += expression.sum(axis=0, dtype=np.float64)
                squares += np.square(expression).sum(axis=0, dtype=np.float64)
                total += selected.shape[0]
        diagnostics.append(
            {
                "min_count": minimum,
                "max_count": maximum,
                "zero_library_cells": zero_cells,
                "max_umi_excluded_by_gene_filter": max_excluded,
            }
        )
    variance = squares / total - np.square(sums / total)
    ranked = np.lexsort((np.arange(width), -variance))[:n_genes]
    return np.sort(ranked), diagnostics, total


def eligible_targets(
    targets, contexts, is_control, required_contexts=("K562", "RPE1"), min_cells=64
):
    """Return targets meeting the cell-count threshold in every primary context."""
    targets, contexts, is_control = map(np.asarray, (targets, contexts, is_control))
    counts = Counter(zip(targets[~is_control].tolist(), contexts[~is_control].tolist()))
    return tuple(
        sorted(
            target
            for target in set(targets[~is_control])
            if all(counts[target, context] >= min_cells for context in required_contexts)
        )
    )


def target_manifest(targets, seed, fractions=(0.7, 0.1, 0.2)):
    """Create deterministic train/validation/sealed-test target lists and hash."""
    names = ("train", "validation", "test")
    ordered = sorted(set(targets), key=lambda target: sha256(f"{seed}\0{target}".encode()).digest())
    exact = np.asarray(fractions) * len(ordered)
    sizes = exact.astype(int)
    for index in sorted(range(3), key=lambda i: (exact[i] - sizes[i], -i), reverse=True)[
        : len(ordered) - sizes.sum()
    ]:
        sizes[index] += 1
    stops = np.cumsum(sizes)
    split = {
        name: sorted(ordered[start:stop])
        for name, start, stop in zip(names, (0, *stops[:-1]), stops)
    }
    payload = {"seed": seed, "fractions": list(fractions), "targets": split}
    payload["sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def assign_roles(
    cell_ids, targets, contexts, is_control, manifest, iid_fraction=0.2, population_size=32
):
    """Assign cells to auditable training/evaluation roles without outcome leakage."""
    cell_ids, targets, contexts, is_control = map(
        np.asarray, (cell_ids, targets, contexts, is_control)
    )
    roles = np.full(cell_ids.size, "excluded", dtype=object)
    roles[is_control & (contexts == "K562")] = "control_train"
    roles[is_control & (contexts == "RPE1")] = "control_inference"
    split_of = {target: name for name, members in manifest["targets"].items() for target in members}
    heldout = {
        ("K562", "validation"): "perturbation_ood_validation",
        ("K562", "test"): "perturbation_ood_test",
        ("RPE1", "train"): "context_ood_test",
        ("RPE1", "validation"): "double_ood_validation_locked",
        ("RPE1", "test"): "double_ood_test",
    }
    for index in np.flatnonzero(~is_control):
        split = split_of.get(targets[index])
        if (contexts[index], split) in heldout:
            roles[index] = heldout[contexts[index], split]
    groups = {target: [] for target in manifest["targets"]["train"]}
    for index in np.flatnonzero((~is_control) & (contexts == "K562")):
        if targets[index] in groups:
            groups[targets[index]].append(index)
    for indices in groups.values():
        ranked = sorted(
            indices,
            key=lambda i: sha256(f"{manifest['seed']}\0iid\0{cell_ids[i]}".encode()).digest(),
        )
        n_test = min(
            max(population_size, round(iid_fraction * len(ranked))), len(ranked) - population_size
        )
        roles[ranked[:n_test]], roles[ranked[n_test:]] = "iid_test", "dynamics_train"
    return roles


def representation_fit_mask(roles):
    """Allow controls and K562 dynamics outcomes, never held-out/RPE1 outcomes."""
    return np.isin(roles, ("control_train", "control_inference", "dynamics_train"))


def required_embedding_mask(roles):
    """Admit every frozen train/evaluation role, but no ineligible target outcomes."""
    return np.asarray(roles) != "excluded"


def tokenize_cell(counts, hvg_columns, max_tokens=512, library_size=10_000):
    """Convert one raw cell into deterministic sparse gene/value tokens."""
    counts = np.asarray(counts, dtype=np.float32)
    with np.errstate(divide="raise", invalid="raise"):
        values = np.log1p(counts[hvg_columns] * (library_size / counts.sum()))
    nonzero = np.flatnonzero(values)
    assert nonzero.size
    selected = nonzero[np.lexsort((nonzero, -values[nonzero]))[:max_tokens]]
    gene_ids = np.full(max_tokens, len(hvg_columns), dtype=np.int64)
    expression = np.zeros(max_tokens, dtype=np.float32)
    padding_mask = np.ones(max_tokens, dtype=bool)
    gene_ids[: selected.size] = selected
    expression[: selected.size] = values[selected]
    padding_mask[: selected.size] = False
    return gene_ids, expression, padding_mask


class ReplogleTokenDataset(Dataset):
    """Role-filtered Replogle cells backed by the untouched raw H5AD files."""

    def __init__(
        self,
        config_path="configs/replogle.yaml",
        manifest_path="manifests/replogle_v1.json",
        all_required=False,
    ):
        config_path = Path(config_path)
        self.config = yaml.safe_load(config_path.read_text())
        self.manifest = json.loads(Path(manifest_path).read_text())
        assert (
            sha256(config_path.read_bytes()).hexdigest()
            == self.manifest["runtime"]["config_sha256"]
        )
        assert self.config["stage1"]["admission_policy"] == (
            "controls_and_k562_dynamics_train_only"
        )
        self.paths, self.columns, self.samples, self._backed = {}, {}, [], {}
        hvg_ids = self.manifest["genes"]["hvg_gene_ids"]
        split = self.manifest["targets"]["split"]
        for context in self.config["data"]["contexts"]:
            source = self.config["data"]["files"][context]
            path = Path("data/raw") / source["filename"]
            data = ad.read_h5ad(path, backed="r")
            targets = data.obs["gene"].astype(str).to_numpy()
            controls = targets == "non-targeting"
            gems = data.obs["gem_group"].to_numpy()
            cell_ids = np.asarray(
                [f"{context}|{gem}|{barcode}" for gem, barcode in zip(gems, data.obs_names)]
            )
            roles = assign_roles(
                cell_ids,
                targets,
                np.full(data.n_obs, context),
                controls,
                split,
                self.config["split"]["iid_test_fraction"],
                self.config["split"]["population_size"],
            )
            admitted = (
                required_embedding_mask(roles) if all_required else representation_fit_mask(roles)
            )
            self.samples.extend(
                (
                    context,
                    int(row),
                    str(cell_ids[row]),
                    roles[row],
                    str(targets[row]),
                    str(gems[row]),
                )
                for row in np.flatnonzero(admitted)
            )
            self.paths[context] = path
            self.columns[context] = np.asarray(
                [data.var_names.get_loc(gene_id) for gene_id in hvg_ids]
            )
            data.file.close()
        if all_required:
            expected_roles = Counter()
            for context in self.manifest["roles"].values():
                expected_roles.update(
                    {role: count for role, count in context.items() if role != "excluded"}
                )
            assert Counter(sample[3] for sample in self.samples) == expected_roles
        else:
            assert len(self.samples) == self.manifest["genes"]["fit_cells"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        context, row, cell_id, role, target, source_batch = self.samples[index]
        if context not in self._backed:
            self._backed[context] = ad.read_h5ad(self.paths[context], backed="r")
        counts = np.asarray(self._backed[context].X[row])
        gene_ids, values, padding_mask = tokenize_cell(
            counts,
            self.columns[context],
            self.config["stage1"]["max_tokens"],
            self.config["data"]["library_size"],
        )
        return {
            "gene_ids": torch.from_numpy(gene_ids),
            "values": torch.from_numpy(values),
            "padding_mask": torch.from_numpy(padding_mask),
            "cell_id": cell_id,
            "context": context,
            "target": target,
            "role": role,
            "source_batch": source_batch,
            "source_row": row,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_backed"] = {}
        return state


def masking_seed(cell_id, epoch, seed):
    """Derive a stable per-cell masking seed independent of loader order/workers."""
    return int.from_bytes(sha256(f"{seed}\0{epoch}\0{cell_id}".encode()).digest()[:8], "little")
