# Frozen condition-level evaluation on untouched Nadig HepG2 and Jurkat screens.
# External controls and outcomes are sampled independently; no cell pairing is constructed.
import csv
import gzip
import json
from array import array
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import torch
import yaml
from scipy import sparse
from scipy.stats import t
from torch.utils.data import DataLoader, Dataset

from causalcelljepa.dynamics import anchored_selected_entry, build_dynamics_model
from causalcelljepa.evaluation import (
    _condition_metric_summaries,
    _retrieval_summaries,
    _self_hashed_manifest,
    fit_linear_baseline,
    paired_model_comparisons,
    population_metrics,
)
from causalcelljepa.external import tokenize_normalized_cell
from causalcelljepa.model import load_frozen_teacher
from causalcelljepa.readout import (
    _bh_adjust,
    _condition_summary,
    _correlation,
    _load_pathway_matrix,
    _paired_transcriptomic_models,
    decode_normalized_latents,
    direct_gene_predictions,
    gene_effect_metrics,
    kernel_gene_predictions,
    pathway_agreement,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


class NadigLatentPopulationDataset(Dataset):
    """Sample unpaired external control/outcome populations in frozen JEPA space."""

    def __init__(
        self,
        cache_path,
        action_path,
        dynamics_manifest_path,
        context,
        population_size=32,
        seed=0,
        expected_targets=None,
    ):
        self.cache_path = Path(cache_path)
        self.context = context
        self.population_size = population_size
        self.seed = seed
        self.epoch = 0
        manifest = json.loads(Path(dynamics_manifest_path).read_text())
        normalization = manifest["normalization"]
        self.mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
        self.scale = (
            np.asarray(normalization["latent_std"], dtype=np.float32)
            * normalization["dimension_scale"]
        )
        assert self.mean.shape == self.scale.shape and np.all(self.scale > 0)
        with h5py.File(self.cache_path, "r") as cache:
            roles = cache["role"].asstr()[:]
            contexts = cache["context"].asstr()[:]
            targets = cache["target"].asstr()[:]
            batches = cache["source_batch"].asstr()[:]
            cell_ids = cache["cell_id"].asstr()[:]
            context_indices = np.flatnonzero(contexts == context)
            self.latent = (cache["latent"][context_indices] - self.mean) / self.scale
        local_index = np.full(len(contexts), -1, dtype=np.int64)
        local_index[context_indices] = np.arange(len(context_indices))
        selected_context = contexts == context
        control_global = np.flatnonzero(selected_context & (roles == "external_control"))
        outcome_global = np.flatnonzero(selected_context & (roles == "external_test"))
        admitted_global = np.concatenate((control_global, outcome_global))
        assert len(control_global) >= population_size and len(outcome_global)
        assert set(batches[admitted_global]) == {"unavailable"}
        assert len(np.unique(cell_ids[admitted_global])) == (
            len(control_global) + len(outcome_global)
        )
        self.controls = local_index[control_global]
        self.condition_targets = sorted(set(targets[outcome_global]))
        if expected_targets is not None:
            assert self.condition_targets == sorted(expected_targets)
        self.outcomes = {
            target: local_index[outcome_global[targets[outcome_global] == target]]
            for target in self.condition_targets
        }
        assert all(len(indices) >= population_size for indices in self.outcomes.values())

        action = torch.load(action_path, map_location="cpu", weights_only=True)
        action_index = {target: index for index, target in enumerate(action["targets"])}
        self.action = {
            target: (
                action["embedding"][action_index[target]],
                action["known"][action_index[target]],
            )
            for target in self.condition_targets
        }
        assert all(bool(known) for _embedding, known in self.action.values())

    def __len__(self):
        return len(self.condition_targets)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample_indices(self, index):
        """Return deterministic, independently sampled control and outcome indices."""
        target = self.condition_targets[index]
        key = f"{self.seed}\0{self.context}\0{self.epoch}\0{target}"
        generator = np.random.default_rng(
            int.from_bytes(sha256(key.encode()).digest()[:8], "little")
        )
        controls = generator.choice(self.controls, self.population_size, replace=False)
        outcomes = generator.choice(self.outcomes[target], self.population_size, replace=False)
        generator.shuffle(controls)
        generator.shuffle(outcomes)
        return controls, outcomes, target

    def __getitem__(self, index):
        controls, outcomes, target = self.sample_indices(index)
        action, known = self.action[target]
        return {
            "control": torch.from_numpy(self.latent[controls]),
            "perturbed": torch.from_numpy(self.latent[outcomes]),
            "action": action,
            "action_known": known,
            "target": target,
        }


def _load_external_models(config, base_config, device):
    inputs = config["inputs"]
    primary_path = Path(inputs["primary_checkpoint_path"])
    assert (primary_path.stat().st_size, file_sha256(primary_path)) == (
        inputs["primary_checkpoint_bytes"],
        inputs["primary_checkpoint_sha256"],
    )
    primary_checkpoint = torch.load(primary_path, map_location="cpu", weights_only=False)
    primary = build_dynamics_model(primary_checkpoint["configuration"]).to(device).eval()
    primary.load_state_dict(primary_checkpoint["model"])

    training, training_sha256 = _self_hashed_manifest(
        inputs["anchored_training_manifest_path"],
        inputs["anchored_training_manifest_sha256"],
    )
    selection, selection_sha256 = _self_hashed_manifest(
        inputs["anchored_selection_manifest_path"],
        inputs["anchored_selection_manifest_sha256"],
    )
    selected_candidate, selected_entry = anchored_selected_entry(training, selection)
    checkpoint_artifact = selected_entry["best_checkpoint"]
    checkpoint_path = Path(checkpoint_artifact["path"])
    assert (checkpoint_path.stat().st_size, file_sha256(checkpoint_path)) == (
        checkpoint_artifact["bytes"],
        checkpoint_artifact["sha256"],
    )
    anchored_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert anchored_checkpoint["configuration"]["revision"]["candidate"] == selected_candidate

    anchored = build_dynamics_model(anchored_checkpoint["configuration"]).to(device).eval()
    anchored.load_state_dict(anchored_checkpoint["model"])
    gated = build_dynamics_model(anchored_checkpoint["configuration"]).eval()
    gated.load_state_dict(anchored_checkpoint["model"])

    gate_manifest, gate_manifest_sha256 = _self_hashed_manifest(
        inputs["residual_gate_manifest_path"],
        inputs["residual_gate_manifest_sha256"],
    )
    gate_artifact = gate_manifest["artifact"]
    assert gate_artifact == {
        "path": inputs["residual_gate_path"],
        "bytes": inputs["residual_gate_bytes"],
        "sha256": inputs["residual_gate_sha256"],
    }
    gate_path = Path(gate_artifact["path"])
    assert (gate_path.stat().st_size, file_sha256(gate_path)) == (
        gate_artifact["bytes"],
        gate_artifact["sha256"],
    )
    gated.configure_residual_gate(torch.load(gate_path, map_location="cpu", weights_only=True))
    gated.to(device)

    linear_effect, mean_effect, baseline_report = fit_linear_baseline(base_config)
    return (
        {
            "anchored_control_ood_gated": gated,
            "anchored_selected": anchored,
            "causalcelljepa": primary,
        },
        linear_effect,
        mean_effect,
        baseline_report,
        {
            "anchored_training_manifest_sha256": training_sha256,
            "anchored_selection_manifest_sha256": selection_sha256,
            "residual_gate_manifest_sha256": gate_manifest_sha256,
            "selected_anchored_candidate": selected_candidate,
            "primary_checkpoint_provenance": primary_checkpoint["provenance"],
            "anchored_checkpoint_provenance": anchored_checkpoint["provenance"],
        },
    )


def _gate_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "populations": len(values),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(
                (0.01, 0.05, 0.5, 0.95, 0.99),
                np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99)),
                strict=True,
            )
        },
    }


def grouped_expression_moments(matrix, labels, groups, columns, block_size=4096):
    """Stream per-condition moments without treating cells as paired observations."""
    group_index = {name: index for index, name in enumerate(groups)}
    sums = np.zeros((len(groups), len(columns)), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(groups), dtype=np.int64)
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        membership = np.asarray([group_index.get(label, -1) for label in labels[start:stop]])
        selected = np.flatnonzero(membership >= 0)
        if not len(selected):
            continue
        block = matrix[start:stop][selected][:, columns]
        block = sparse.csr_matrix(block)
        assignment = sparse.csr_matrix(
            (np.ones(len(selected)), (membership[selected], np.arange(len(selected)))),
            shape=(len(groups), len(selected)),
        )
        sums += (assignment @ block).toarray()
        squares += (assignment @ block.multiply(block)).toarray()
        counts += np.bincount(membership[selected], minlength=len(groups))
    means = sums / counts[:, None]
    variances = np.maximum(
        0,
        (squares - np.square(sums) / counts[:, None]) / np.maximum(counts[:, None] - 1, 1),
    )
    return means, variances, counts


def _external_expression_truth(preregistered, manifest, transcriptomics, maximum_conditions=None):
    """Compute external pseudobulk effects and explicitly limited cell-level DE labels."""
    replogle = json.loads(Path(transcriptomics["inputs"]["replogle_manifest_path"]).read_text())
    hvg_genes = replogle["genes"]["hvg_gene_names"]
    truth, report, pathway_matrices, hvg_indices = {}, {}, {}, {}
    for context in preregistered["protocol"]["contexts"]:
        source = preregistered["source"]["files"][context]
        path = Path(preregistered["source"]["raw_directory"]) / source["filename"]
        assert (path.stat().st_size, file_sha256(path)) == (
            source["bytes"],
            manifest["contexts"][context]["source_sha256"],
        )
        data = ad.read_h5ad(path, backed="r")
        labels = data.obs[preregistered["source"]["perturbation_column"]].astype(str).to_numpy()
        targets = manifest["contexts"][context]["eligible_targets"]
        targets = targets[: maximum_conditions or len(targets)]
        overlap = [
            (index, data.var_names.get_loc(gene))
            for index, gene in enumerate(hvg_genes)
            if gene in data.var_names
        ]
        hvg_indices[context] = np.asarray([item[0] for item in overlap])
        source_columns = np.asarray([item[1] for item in overlap])
        overlap_genes = [hvg_genes[index] for index in hvg_indices[context]]
        groups = [preregistered["source"]["control_label"], *targets]
        means, variances, counts = grouped_expression_moments(
            data.X, labels, groups, source_columns
        )
        data.file.close()
        assert counts[0] == source["expected_controls"] and np.all(counts[1:] >= 32)
        matrix, _ = _load_pathway_matrix(transcriptomics["inputs"]["go_gmt_path"], overlap_genes)
        sizes = np.asarray(matrix.sum(1)).ravel()
        pathway_matrices[context] = matrix[(sizes > 0) & (sizes < len(overlap_genes))]
        report[context] = {
            "targets": len(targets),
            "frozen_hvg_overlap": len(overlap_genes),
            "pathways_with_observed_genes": pathway_matrices[context].shape[0],
            "control_cells": int(counts[0]),
            "conditions_with_no_degs": 0,
            "de_method": "cell_level_welch_t_with_bh_fdr_and_effect_threshold",
            "batch_limitation": "experimental_batch_labels_unavailable",
        }
        for index, target in enumerate(targets, 1):
            effect = means[index] - means[0]
            standard_error_squared = variances[index] / counts[index] + variances[0] / counts[0]
            statistic = np.divide(
                np.abs(effect),
                np.sqrt(standard_error_squared),
                out=np.zeros_like(effect),
                where=standard_error_squared > 0,
            )
            degrees = np.square(standard_error_squared) / (
                np.square(variances[index] / counts[index]) / (counts[index] - 1)
                + np.square(variances[0] / counts[0]) / (counts[0] - 1)
            ).clip(1e-30)
            q_value = _bh_adjust(2 * t.sf(statistic, degrees))
            deg = (q_value <= transcriptomics["metrics"]["deg_batch_fdr"]) & (
                np.abs(effect) >= transcriptomics["metrics"]["deg_min_abs_effect"]
            )
            report[context]["conditions_with_no_degs"] += int(not deg.any())
            truth[context, target] = {
                "effect": effect.astype(np.float32),
                "deg": deg,
                "cells": int(counts[index]),
            }
    return truth, report, pathway_matrices, hvg_indices


@torch.inference_mode()
def run_nadig_external_latent_evaluation(
    config_path="configs/nadig_external_evaluation.yaml",
    repeats=None,
    maximum_conditions=None,
    output_directory=None,
    device="cpu",
):
    """Evaluate all frozen models without fitting or selecting on external outcomes."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    device = torch.device(device)
    if device.type == "mps":
        assert torch.backends.mps.is_available()
    inputs = config["inputs"]
    assert file_sha256(inputs["preregistered_config_path"]) == inputs["preregistered_config_sha256"]
    preregistered = yaml.safe_load(Path(inputs["preregistered_config_path"]).read_text())
    assert preregistered["protocol"]["role"] == "external_test_only"
    assert preregistered["protocol"]["no_model_selection_or_tuning"] is True
    assert preregistered["protocol"]["batch_matching"] == "unavailable_in_trimmed_source"
    external_manifest, external_manifest_sha256 = _self_hashed_manifest(
        inputs["external_manifest_path"], inputs["external_manifest_sha256"]
    )
    assert external_manifest["protocol"] == preregistered["protocol"]

    for kind in ("external_latent_cache", "action_cache"):
        path = Path(inputs[f"{kind}_path"])
        assert (path.stat().st_size, file_sha256(path)) == (
            inputs[f"{kind}_bytes"],
            inputs[f"{kind}_sha256"],
        )
    with h5py.File(inputs["external_latent_cache_path"], "r") as cache:
        assert cache.attrs["external_manifest_sha256"] == external_manifest_sha256
        assert cache.attrs["teacher_sha256"] == preregistered["latent_cache"]["teacher_sha256"]
        external_cache_provenance = json.loads(cache.attrs["provenance_json"])
        assert external_cache_provenance["git"]["dirty"] is False

    assert (
        file_sha256(inputs["base_evaluation_config_path"])
        == inputs["base_evaluation_config_sha256"]
    )
    base_config = yaml.safe_load(Path(inputs["base_evaluation_config_path"]).read_text())
    assert (
        base_config["sampling"]["population_size"] == preregistered["protocol"]["population_size"]
    )
    assert base_config["sampling"]["repeats"] == preregistered["protocol"]["repeats"]
    assert set(config["models"]) == set(preregistered["protocol"]["models"])
    assert config["latent_metrics"] == preregistered["protocol"]["metrics"]["latent"]

    models, linear_effect, mean_effect, baseline_report, model_provenance = _load_external_models(
        config, base_config, device
    )
    dynamics_manifest = json.loads(
        Path(base_config["inputs"]["dynamics_manifest_path"]).read_text()
    )
    median_distance = dynamics_manifest["normalization"]["median_training_latent_distance"]
    repeats = repeats or preregistered["protocol"]["repeats"]
    output = Path(output_directory or config["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)

    records = []
    signatures = defaultdict(lambda: [0, None])
    gate_confidences = defaultdict(list)
    contexts = preregistered["protocol"]["contexts"]
    for context in contexts:
        dataset = NadigLatentPopulationDataset(
            inputs["external_latent_cache_path"],
            inputs["action_cache_path"],
            base_config["inputs"]["dynamics_manifest_path"],
            context,
            preregistered["protocol"]["population_size"],
            preregistered["seed"],
            external_manifest["contexts"][context]["eligible_targets"],
        )
        indices = range(min(len(dataset), maximum_conditions or len(dataset)))
        for repeat in range(repeats):
            dataset.set_epoch(repeat)
            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                sampler=list(indices),
                num_workers=config["num_workers"],
            )
            for batch in loader:
                control = batch["control"].to(device)
                observed = batch["perturbed"].to(device)
                action = batch["action"].to(device)
                action_known = batch["action_known"].to(device)
                learned = {
                    name: model(control, action, action_known) for name, model in models.items()
                }
                confidence = models["anchored_control_ood_gated"].residual_gate_confidence(control)
                gate_confidences[context].extend(confidence.cpu().tolist())
                predictions = {
                    **learned,
                    "linear_esm": control
                    + torch.from_numpy(
                        np.stack([linear_effect[target] for target in batch["target"]])
                    ).to(device)[:, None],
                    "mean_effect": control + torch.from_numpy(mean_effect).to(device)[None, None],
                    "no_change": control,
                }
                assert set(predictions) == set(config["models"])
                assert all(value.dtype == observed.dtype for value in predictions.values())
                true_effect = observed.mean(1) - control.mean(1)
                for index, target in enumerate(batch["target"]):
                    key = (context, "true", target)
                    signatures[key][0] += 1
                    value = true_effect[index].cpu().numpy()
                    signatures[key][1] = (
                        value if signatures[key][1] is None else signatures[key][1] + value
                    )
                for name, predicted in predictions.items():
                    metrics = population_metrics(
                        predicted,
                        observed,
                        control,
                        median_distance,
                        base_config["metrics"],
                    )
                    assert set(metrics) == set(config["latent_metrics"])
                    predicted_effect = (predicted.mean(1) - control.mean(1)).detach()
                    for index, target in enumerate(batch["target"]):
                        record = {
                            "regime": context,
                            "context": context,
                            "outcome_role": "external_test",
                            "target": target,
                            "repeat": repeat,
                            "model": name,
                        }
                        for metric, values in metrics.items():
                            value = float(values[index].cpu())
                            record[metric] = value if np.isfinite(value) else None
                        records.append(record)
                        key = (context, name, target)
                        signatures[key][0] += 1
                        value = predicted_effect[index].cpu().numpy()
                        signatures[key][1] = (
                            value if signatures[key][1] is None else signatures[key][1] + value
                        )

    metrics_path = output / "condition_metrics.jsonl"
    with metrics_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "condition_metrics": _condition_metric_summaries(
            records, base_config["metrics"]["bootstrap_resamples"], preregistered["seed"]
        ),
        "retrieval": _retrieval_summaries(signatures, contexts, config["models"]),
        "residual_gate_confidence": {
            context: _gate_summary(values) for context, values in gate_confidences.items()
        },
    }
    comparisons = paired_model_comparisons(
        records,
        config["comparisons"],
        base_config["metrics"]["bootstrap_resamples"],
        preregistered["seed"],
    )
    provenance = {
        "config": deepcopy(config),
        "preregistered_config": deepcopy(preregistered),
        "executed_contexts": contexts,
        "executed_repeats": repeats,
        "device": str(device),
        "maximum_conditions_per_context": maximum_conditions,
        "statistical_unit": "perturbation_condition",
        "batch_matching": "unavailable_in_trimmed_source",
        "external_cells_used_for_fit_or_selection": False,
        "external_cache_provenance": external_cache_provenance,
        "baseline_report": baseline_report,
        "model_provenance": model_provenance,
        "file_sha256": {
            "config": file_sha256(config_path),
            "condition_metrics": file_sha256(metrics_path),
            "external_latent_cache": inputs["external_latent_cache_sha256"],
            "external_manifest": external_manifest_sha256,
            "preregistered_config": inputs["preregistered_config_sha256"],
        },
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n"
    )
    (output / "baseline_report.json").write_text(
        json.dumps(baseline_report, indent=2, sort_keys=True) + "\n"
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return summary, comparisons, provenance


@torch.inference_mode()
def run_nadig_external_transcriptomic_evaluation(
    config_path="configs/nadig_external_evaluation.yaml",
    repeats=None,
    maximum_conditions=None,
    output_directory=None,
    device="cpu",
):
    """Decode frozen external predictions and score shared-gene perturbation effects."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    inputs = config["inputs"]
    device = torch.device(device)
    assert file_sha256(inputs["preregistered_config_path"]) == inputs["preregistered_config_sha256"]
    preregistered = yaml.safe_load(Path(inputs["preregistered_config_path"]).read_text())
    assert (
        config["transcriptomic_metrics"] == preregistered["protocol"]["metrics"]["transcriptomic"]
    )
    manifest, manifest_sha256 = _self_hashed_manifest(
        inputs["external_manifest_path"], inputs["external_manifest_sha256"]
    )
    assert (
        file_sha256(inputs["base_transcriptomics_config_path"])
        == inputs["base_transcriptomics_config_sha256"]
    )
    transcriptomics = yaml.safe_load(Path(inputs["base_transcriptomics_config_path"]).read_text())
    replogle, _ = _self_hashed_manifest(
        transcriptomics["inputs"]["replogle_manifest_path"],
        preregistered["inputs"]["replogle_manifest_sha256"],
    )
    for kind in ("readout_checkpoint", "go_gmt"):
        assert (
            file_sha256(transcriptomics["inputs"][f"{kind}_path"])
            == transcriptomics["inputs"][f"{kind}_sha256"]
        )
    readout = torch.load(
        transcriptomics["inputs"]["readout_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    readout["weights"] = readout["weights"].to(device)
    readout["bias"] = readout["bias"].to(device)
    base = yaml.safe_load(Path(inputs["base_evaluation_config_path"]).read_text())
    models, linear_effect, mean_effect, baseline_report, model_provenance = _load_external_models(
        config, base, device
    )
    truth, truth_report, pathway_matrices, hvg_indices = _external_expression_truth(
        preregistered, manifest, transcriptomics, maximum_conditions
    )
    hvg_index = {gene: index for index, gene in enumerate(replogle["genes"]["hvg_gene_names"])}
    overlap_positions = {
        context: {global_index: local_index for local_index, global_index in enumerate(indices)}
        for context, indices in hvg_indices.items()
    }
    repeats = repeats or preregistered["protocol"]["repeats"]
    output = Path(output_directory or config["transcriptomic_output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    records, signatures = [], defaultdict(lambda: [0, None])
    for context in preregistered["protocol"]["contexts"]:
        dataset = NadigLatentPopulationDataset(
            inputs["external_latent_cache_path"],
            inputs["action_cache_path"],
            base["inputs"]["dynamics_manifest_path"],
            context,
            preregistered["protocol"]["population_size"],
            preregistered["seed"],
            manifest["contexts"][context]["eligible_targets"],
        )
        indices = range(min(len(dataset), maximum_conditions or len(dataset)))
        for repeat in range(repeats):
            dataset.set_epoch(repeat)
            loader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                sampler=list(indices),
                num_workers=config["num_workers"],
            )
            for batch in loader:
                control = batch["control"].to(device)
                action = batch["action"].to(device)
                known = batch["action_known"].to(device)
                predictions = {
                    **{name: model(control, action, known) for name, model in models.items()},
                    "linear_esm": control
                    + torch.from_numpy(
                        np.stack([linear_effect[target] for target in batch["target"]])
                    ).to(device)[:, None],
                    "mean_effect": control + torch.from_numpy(mean_effect).to(device)[None, None],
                    "no_change": control,
                }
                control_expression = decode_normalized_latents(control.mean(1), readout)
                for name, predicted in predictions.items():
                    decoded = (
                        (decode_normalized_latents(predicted.mean(1), readout) - control_expression)
                        .cpu()
                        .numpy()[:, hvg_indices[context]]
                    )
                    for index, target in enumerate(batch["target"]):
                        observed = truth[context, target]
                        target_index = overlap_positions[context].get(hvg_index.get(target))
                        records.append(
                            {
                                "regime": context,
                                "context": context,
                                "outcome_role": "external_test",
                                "target": target,
                                "repeat": repeat,
                                "model": name,
                                "truth_batches": 0,
                                "truth_cells": observed["cells"],
                                "true_deg_count": int(observed["deg"].sum()),
                                "target_in_hvg": target_index is not None,
                                **gene_effect_metrics(
                                    decoded[index],
                                    observed["effect"],
                                    target_index,
                                    observed["deg"],
                                    transcriptomics["metrics"]["retrospective_top_genes"],
                                ),
                            }
                        )
                        key = (context, name, target)
                        signatures[key][0] += 1
                        signatures[key][1] = (
                            decoded[index]
                            if signatures[key][1] is None
                            else signatures[key][1] + decoded[index]
                        )
    pathway_records, retrieval = [], []
    for context in preregistered["protocol"]["contexts"]:
        targets = sorted(target for regime, target in truth if regime == context)
        observed = np.stack([truth[context, target]["effect"] for target in targets])
        normalized_observed = observed / np.linalg.norm(observed, axis=1, keepdims=True).clip(1e-12)
        for name in config["models"]:
            predicted = np.stack(
                [
                    signatures[context, name, target][1] / signatures[context, name, target][0]
                    for target in targets
                ]
            )
            similarity = predicted / np.linalg.norm(predicted, axis=1, keepdims=True).clip(1e-12)
            similarity = similarity @ normalized_observed.T
            ranks = np.asarray(
                [
                    1
                    + np.count_nonzero(row > row[index])
                    + 0.5 * (np.count_nonzero(row == row[index]) - 1)
                    for index, row in enumerate(similarity)
                ]
            )
            retrieval.append(
                {
                    "regime": context,
                    "model": name,
                    "targets": len(targets),
                    "top_1": float(np.mean(ranks <= 1)),
                    "top_5": float(np.mean(ranks <= 5)),
                    "mean_reciprocal_rank": float(np.mean(1 / ranks)),
                    "median_rank": float(np.median(ranks)),
                }
            )
            for index, target in enumerate(targets):
                pathway_records.append(
                    {
                        "regime": context,
                        "context": context,
                        "outcome_role": "external_test",
                        "target": target,
                        "repeat": 0,
                        "model": name,
                        "truth_batches": 0,
                        "truth_cells": truth[context, target]["cells"],
                        "true_deg_count": int(truth[context, target]["deg"].sum()),
                        "target_in_hvg": hvg_index.get(target) in overlap_positions[context],
                        **pathway_agreement(
                            predicted[index],
                            observed[index],
                            pathway_matrices[context],
                            transcriptomics["metrics"]["pathway_top_k"],
                        ),
                    }
                )
    resamples = transcriptomics["metrics"]["bootstrap_resamples"]
    summary = {
        "condition_metrics": _condition_summary(records, resamples, preregistered["seed"]),
        "pathway_metrics": _condition_summary(pathway_records, resamples, preregistered["seed"]),
        "retrieval": retrieval,
    }
    comparisons = {
        "condition_comparisons": _paired_transcriptomic_models(
            records, config["comparisons"], resamples, preregistered["seed"]
        ),
        "pathway_comparisons": _paired_transcriptomic_models(
            pathway_records, config["comparisons"], resamples, preregistered["seed"]
        ),
    }
    provenance = {
        "config": deepcopy(config),
        "executed_repeats": repeats,
        "maximum_conditions_per_context": maximum_conditions,
        "statistical_unit": "perturbation_condition",
        "batch_matching": "unavailable_in_trimmed_source",
        "external_cells_used_for_fit_or_selection": False,
        "truth_report": truth_report,
        "baseline_report": baseline_report,
        "model_provenance": model_provenance,
        "file_sha256": {
            "config": file_sha256(config_path),
            "external_manifest": manifest_sha256,
            "base_transcriptomics_config": inputs["base_transcriptomics_config_sha256"],
        },
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    for filename, payload in (
        ("summary.json", summary),
        ("paired_comparisons.json", comparisons),
        ("truth_report.json", truth_report),
        ("baseline_report.json", baseline_report),
        ("provenance.json", provenance),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    return summary, comparisons, truth_report, provenance


def load_adamson_expression(preregistered, hvg_gene_ids, roles, maximum_cells=None, grouped=False):
    """Read only admitted roles into the frozen vocabulary with exact Ensembl matching."""
    root, files = Path(preregistered["source"]["raw_directory"]), preregistered["source"]["files"]
    for specification in files.values():
        path = root / specification["filename"]
        assert (path.stat().st_size, file_sha256(path)) == (
            specification["bytes"],
            specification["sha256"],
        )
    with gzip.open(root / files["barcodes"]["filename"], "rt") as handle:
        barcodes = [line.rstrip("\n") for line in handle]
    with gzip.open(root / files["genes"]["filename"], "rt") as handle:
        genes = list(csv.reader(handle, delimiter="\t"))
    with gzip.open(root / files["identities"]["filename"], "rt") as handle:
        identities = list(csv.DictReader(handle))
    barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
    filtering, targets = preregistered["filtering"], preregistered["targets"]
    controls, scored, reference = (
        set(filtering["controls"]),
        set(targets["scored"]),
        set(targets["systematic_reference_only"]),
    )
    samples = []
    for row in identities:
        if not (
            row["good coverage"] == str(filtering["good_coverage"]).upper()
            and int(row["number of cells"]) == filtering["number_of_cells"]
        ):
            continue
        guide = row["guide identity"]
        target = "control" if guide in controls else guide.rsplit("_", 1)[0]
        role = (
            "control"
            if guide in controls
            else "scored"
            if target in scored
            else "reference"
            if target in reference
            else "excluded"
        )
        if guide != filtering["unknown_identity"] and role in roles:
            barcode = row["cell BC"]
            samples.append(
                (barcode_index[barcode], barcode, target, barcode.rsplit("-", 1)[1], role)
            )
    samples.sort()
    samples = samples[: maximum_cells or len(samples)]
    selected_columns = {sample[0]: index for index, sample in enumerate(samples)}
    hvg_index = {gene: index for index, gene in enumerate(hvg_gene_ids)}
    assert len(hvg_index) == len(hvg_gene_ids)
    source_rows = {
        index: hvg_index[row[0]] for index, row in enumerate(genes) if row[0] in hvg_index
    }
    rows, columns, values = array("i"), array("i"), array("f")
    library_size = np.zeros(len(samples), dtype=np.float64)
    groups = sorted({(sample[4], sample[2], sample[3]) for sample in samples})
    group_index = {group: index for index, group in enumerate(groups)}
    group_sums = np.zeros((len(groups), len(hvg_gene_ids)), dtype=np.float64)
    with gzip.open(root / files["matrix"]["filename"], "rt") as handle:
        line = next(handle)
        while line.startswith("%"):
            line = next(handle)
        dimensions = tuple(map(int, line.split()))
        assert dimensions[:2] == (len(genes), len(barcodes))
        previous_column, local_column, cell_rows, cell_values = -1, None, [], []
        for line in handle:
            source_row, source_column, value = line.split()
            source_column = int(source_column) - 1
            assert source_column >= previous_column
            if source_column != previous_column:
                if local_column is not None:
                    normalized = np.log1p(
                        10_000
                        * np.asarray(cell_values, dtype=np.float64)
                        / library_size[local_column]
                    )
                    if grouped:
                        sample = samples[local_column]
                        group_sums[group_index[sample[4], sample[2], sample[3]], cell_rows] += (
                            normalized
                        )
                    else:
                        rows.extend([local_column] * len(cell_rows))
                        columns.extend(cell_rows)
                        values.extend(normalized)
                previous_column, local_column = source_column, selected_columns.get(source_column)
                cell_rows, cell_values = [], []
            if local_column is None:
                continue
            value = float(value)
            library_size[local_column] += value
            vocab_position = source_rows.get(int(source_row) - 1)
            if vocab_position is not None:
                cell_rows.append(vocab_position)
                cell_values.append(value)
        if local_column is not None:
            normalized = np.log1p(
                10_000 * np.asarray(cell_values, dtype=np.float64) / library_size[local_column]
            )
            if grouped:
                sample = samples[local_column]
                group_sums[group_index[sample[4], sample[2], sample[3]], cell_rows] += normalized
            else:
                rows.extend([local_column] * len(cell_rows))
                columns.extend(cell_rows)
                values.extend(normalized)
    assert np.all(library_size > 0)
    metadata = {
        "cell_id": np.asarray([sample[1] for sample in samples]),
        "target": np.asarray([sample[2] for sample in samples]),
        "batch": np.asarray([sample[3] for sample in samples]),
        "role": np.asarray([sample[4] for sample in samples]),
        "library_size": library_size,
    }
    if grouped:
        cell_groups = np.asarray(
            [group_index[sample[4], sample[2], sample[3]] for sample in samples]
        )
        expression = {
            "groups": groups,
            "counts": np.bincount(cell_groups, minlength=len(groups)),
            "sums": group_sums,
        }
    else:
        expression = sparse.csr_matrix(
            (
                np.frombuffer(values, dtype=np.float32),
                (
                    np.frombuffer(rows, dtype=np.int32),
                    np.frombuffer(columns, dtype=np.int32),
                ),
            ),
            shape=(len(samples), len(hvg_gene_ids)),
        )
    return expression, metadata, np.asarray(sorted(source_rows.values()), dtype=np.int64)


def _adamson_runtime(config_path):
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    inputs = config["inputs"]
    assert file_sha256(inputs["preregistered_config_path"]) == inputs["preregistered_config_sha256"]
    preregistered = yaml.safe_load(Path(inputs["preregistered_config_path"]).read_text())
    preparation, preparation_sha256 = _self_hashed_manifest(
        inputs["preparation_manifest_path"], inputs["preparation_manifest_sha256"]
    )
    assert preparation["source"]["config_sha256"] == inputs["preregistered_config_sha256"]
    assert (
        file_sha256(inputs["transcriptomics_config_path"])
        == inputs["transcriptomics_config_sha256"]
    )
    transcriptomics = yaml.safe_load(Path(inputs["transcriptomics_config_path"]).read_text())
    replogle, replogle_sha256 = _self_hashed_manifest(
        transcriptomics["inputs"]["replogle_manifest_path"],
        preregistered["frozen_candidate"]["replogle_manifest_sha256"],
    )
    return (
        config_path,
        config,
        preregistered,
        transcriptomics,
        replogle,
        {"preparation": preparation_sha256, "replogle": replogle_sha256},
    )


@torch.inference_mode()
def predict_adamson_external(
    config_path="configs/adamson_external_evaluation.yaml",
    maximum_controls=None,
    output_path=None,
    write_manifest=True,
):
    """Freeze Adamson predictions using controls only, before any perturbed outcome scoring."""
    config_path, config, preregistered, _, replogle, source_manifests = _adamson_runtime(
        config_path
    )
    inputs, frozen = config["inputs"], preregistered["frozen_candidate"]
    action_path = Path(frozen["action_path"])
    assert (action_path.stat().st_size, file_sha256(action_path)) == (
        frozen["action_bytes"],
        frozen["action_sha256"],
    )
    expression, metadata, observed = load_adamson_expression(
        preregistered, replogle["genes"]["hvg_gene_ids"], {"control"}, maximum_controls
    )
    assert set(metadata["role"]) == {"control"}
    teacher_path = Path(inputs["teacher_path"])
    assert (teacher_path.stat().st_size, file_sha256(teacher_path)) == (
        inputs["teacher_bytes"],
        inputs["teacher_sha256"],
    )
    device = torch.device(config["prediction"]["device"])
    teacher, teacher_payload = load_frozen_teacher(
        teacher_path, len(replogle["genes"]["hvg_gene_ids"]), device
    )
    latents = []
    batch_size = config["prediction"]["batch_size"]
    for start in range(0, expression.shape[0], batch_size):
        tokenized = [
            tokenize_normalized_cell(
                expression.data[expression.indptr[row] : expression.indptr[row + 1]],
                expression.indices[expression.indptr[row] : expression.indptr[row + 1]],
                max_tokens=config["prediction"]["max_tokens"],
            )
            for row in range(start, min(start + batch_size, expression.shape[0]))
        ]
        gene_ids, values, padding = (
            torch.from_numpy(np.stack(items)).to(device) for items in zip(*tokenized, strict=True)
        )
        latents.append(teacher(gene_ids, values, padding).cpu())
    control_latent = torch.cat(latents).mean(0).numpy()
    dynamics, dynamics_sha256 = _self_hashed_manifest(
        inputs["dynamics_manifest_path"], inputs["dynamics_manifest_sha256"]
    )
    normalization = dynamics["normalization"]
    control_latent = (control_latent - np.asarray(normalization["latent_mean"])) / (
        np.asarray(normalization["latent_std"]) * normalization["dimension_scale"]
    )
    gate_manifest, gate_sha256 = _self_hashed_manifest(
        inputs["gate_manifest_path"], inputs["gate_manifest_sha256"]
    )
    gate_artifact = gate_manifest["artifact"]
    gate_path = Path(gate_artifact["path"])
    assert (gate_path.stat().st_size, file_sha256(gate_path)) == (
        gate_artifact["bytes"],
        gate_artifact["sha256"],
    )
    gate = torch.load(gate_path, map_location="cpu", weights_only=True)
    score = float(
        np.square((control_latent - gate["center"].numpy()) / gate["scale"].numpy()).mean()
    )
    confidence = float(np.exp(-max(0, score - gate["threshold"]) / gate["temperature"]))
    selections = {}
    for name in ("external", "string"):
        selection, declared = _self_hashed_manifest(
            inputs[f"{name}_selection_manifest_path"],
            inputs[f"{name}_selection_manifest_sha256"],
        )
        artifact = selection["artifacts"]["checkpoint"]
        path = Path(artifact["path"])
        assert (path.stat().st_size, file_sha256(path)) == (artifact["bytes"], artifact["sha256"])
        selections[name] = (
            kernel_gene_predictions(
                torch.load(path, map_location="cpu", weights_only=True), frozen["action_path"]
            ),
            declared,
        )
    _self_hashed_manifest(inputs["direct_manifest_path"], inputs["direct_manifest_sha256"])
    direct_path = Path(inputs["direct_checkpoint_path"])
    assert (direct_path.stat().st_size, file_sha256(direct_path)) == (
        inputs["direct_checkpoint_bytes"],
        inputs["direct_checkpoint_sha256"],
    )
    assert file_sha256(inputs["direct_action_path"]) == inputs["direct_action_sha256"]
    direct_checkpoint = torch.load(direct_path, map_location="cpu", weights_only=True)
    direct = direct_gene_predictions(direct_checkpoint, inputs["direct_action_path"])
    targets = preregistered["targets"]["scored"]
    external = np.stack([selections["external"][0][target] for target in targets])
    string = np.stack([selections["string"][0][target] for target in targets])
    models = {
        "control_gated_external_response": confidence * string + (1 - confidence) * external,
        "external_response_multiview_rbf": external,
        "string_kernel_gene_go_rbf": string,
        "direct_gene_esm": np.stack([direct[target] for target in targets]),
        "mean_effect": np.repeat(direct_checkpoint["y_mean"].numpy()[None], len(targets), axis=0),
        "no_change": np.zeros_like(external),
    }
    output = Path(output_path or config["outputs"]["prediction_path"])
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "targets": targets,
        "hvg_gene_ids": replogle["genes"]["hvg_gene_ids"],
        "hvg_gene_names": replogle["genes"]["hvg_gene_names"],
        "observed_hvg_positions": torch.from_numpy(observed),
        "effects": {
            name: torch.from_numpy(value.astype(np.float32)) for name, value in models.items()
        },
        "control_gate": {"score": score, "reference_confidence": confidence},
        "controls": {"cells": expression.shape[0], "batches": sorted(set(metadata["batch"]))},
        "leakage": {"roles_read": ["control"], "perturbed_outcomes_used": False},
    }
    torch.save(payload, output)
    manifest = {
        "format_version": 1,
        "artifact": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
        },
        "prediction": {
            "targets": len(targets),
            "genes": len(replogle["genes"]["hvg_gene_ids"]),
            "models": sorted(models),
            "control_gate": payload["control_gate"],
        },
        "leakage": payload["leakage"],
        "source": {
            "config_sha256": file_sha256(config_path),
            "teacher_sha256": inputs["teacher_sha256"],
            "dynamics_manifest_sha256": dynamics_sha256,
            "gate_manifest_sha256": gate_sha256,
            "external_selection_manifest_sha256": selections["external"][1],
            "string_selection_manifest_sha256": selections["string"][1],
            **{f"{name}_manifest_sha256": value for name, value in source_manifests.items()},
        },
        "provenance": {
            "teacher": teacher_payload["provenance"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    manifest["manifest_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if write_manifest:
        manifest_path = Path(config["outputs"]["prediction_manifest_path"])
        assert maximum_controls is None and not manifest_path.exists()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload, manifest


def _centroid_accuracy(predicted, target_index, truth_centroids):
    distances = np.linalg.norm(truth_centroids - predicted, axis=1)
    return float(np.mean(distances[target_index] < np.delete(distances, target_index)))


def run_adamson_external_evaluation(
    config_path="configs/adamson_external_evaluation.yaml",
):
    """Open Adamson perturbation outcomes once and apply the locked terminal decision rule."""
    config_path, config, preregistered, transcriptomics, replogle, source_manifests = (
        _adamson_runtime(config_path)
    )
    outputs, metrics = config["outputs"], config["metrics"]
    prediction_manifest = json.loads(Path(outputs["prediction_manifest_path"]).read_text())
    prediction_declared = prediction_manifest.pop("manifest_sha256")
    assert (
        prediction_declared
        == sha256(
            json.dumps(prediction_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert prediction_manifest["source"]["config_sha256"] == file_sha256(config_path)
    prediction_artifact = prediction_manifest["artifact"]
    prediction_path = Path(prediction_artifact["path"])
    assert (prediction_path.stat().st_size, file_sha256(prediction_path)) == (
        prediction_artifact["bytes"],
        prediction_artifact["sha256"],
    )
    prediction = torch.load(prediction_path, map_location="cpu", weights_only=True)
    assert prediction["leakage"] == {"roles_read": ["control"], "perturbed_outcomes_used": False}
    assert prediction["targets"] == preregistered["targets"]["scored"]
    assert prediction["hvg_gene_ids"] == replogle["genes"]["hvg_gene_ids"]
    expression, metadata, observed = load_adamson_expression(
        preregistered,
        replogle["genes"]["hvg_gene_ids"],
        {"control", "scored", "reference"},
        grouped=True,
    )
    assert np.array_equal(observed, prediction["observed_hvg_positions"].numpy())
    gene_names = np.asarray(replogle["genes"]["hvg_gene_names"])[observed]
    group_sums = {
        group: expression["sums"][index, observed]
        for index, group in enumerate(expression["groups"])
    }
    group_counts = {
        group: expression["counts"][index] for index, group in enumerate(expression["groups"])
    }
    roles, batches = metadata["role"], metadata["batch"]
    control_means = {
        batch: group_sums["control", "control", batch] / group_counts["control", "control", batch]
        for batch in sorted(set(batches[roles == "control"]))
    }
    scored_targets = preregistered["targets"]["scored"]
    reference_targets = preregistered["targets"]["systematic_reference_only"]
    truth_centroids = np.stack(
        [
            sum(
                values
                for (role, group_target, _), values in group_sums.items()
                if role == "scored" and group_target == target
            )
            / sum(
                count
                for (role, group_target, _), count in group_counts.items()
                if role == "scored" and group_target == target
            )
            for target in scored_targets
        ]
    )
    reference_centroids = np.stack(
        [
            sum(
                values
                for (role, group_target, _), values in group_sums.items()
                if role == "reference" and group_target == target
            )
            / sum(
                count
                for (role, group_target, _), count in group_counts.items()
                if role == "reference" and group_target == target
            )
            for target in reference_targets
        ]
    )
    perturbed_reference = reference_centroids.mean(0)
    perturbed_mean = sum(
        values for (role, _, _), values in group_sums.items() if role == "reference"
    ) / sum(count for (role, _, _), count in group_counts.items() if role == "reference")
    matched_controls, truth = {}, {}
    for target, centroid in zip(scored_targets, truth_centroids, strict=True):
        target_groups = {
            batch: group_counts["scored", target, batch]
            for role, group_target, batch in group_counts
            if role == "scored" and group_target == target
        }
        matched_controls[target] = sum(
            count * control_means[batch] for batch, count in target_groups.items()
        ) / sum(target_groups.values())
        batch_effects = np.stack(
            [
                group_sums["scored", target, batch] / group_counts["scored", target, batch]
                - control_means[batch]
                for batch in sorted(target_groups)
            ]
        )
        standard_error = batch_effects.std(0, ddof=1) / np.sqrt(len(batch_effects))
        statistic = np.divide(
            np.abs(batch_effects.mean(0)),
            standard_error,
            out=np.where(batch_effects.mean(0) == 0, 0.0, np.inf),
            where=standard_error > 0,
        )
        q_value = _bh_adjust(2 * t.sf(statistic, len(batch_effects) - 1))
        effect = centroid - matched_controls[target]
        truth[target] = {
            "centroid": centroid,
            "effect": effect,
            "deg": (q_value <= metrics["deg_batch_fdr"])
            & (np.abs(effect) >= metrics["deg_min_abs_effect"]),
            "batches": len(batch_effects),
            "cells": int(sum(target_groups.values())),
        }
    pathway_matrix, pathway_labels = _load_pathway_matrix(
        transcriptomics["inputs"]["go_gmt_path"], gene_names
    )
    pathway_sizes = np.asarray(pathway_matrix.sum(1)).ravel()
    pathway_matrix = pathway_matrix[(pathway_sizes > 0) & (pathway_sizes < len(gene_names))]
    assert len(pathway_matrix.data) and len(pathway_labels) == 4328
    model_effects = {
        name: value.numpy()[:, observed] for name, value in prediction["effects"].items()
    }
    predicted_centroids = {
        name: np.stack(
            [
                matched_controls[target] + effects[index]
                for index, target in enumerate(scored_targets)
            ]
        )
        for name, effects in model_effects.items()
    }
    predicted_centroids["perturbed_mean"] = np.repeat(
        perturbed_mean[None], len(scored_targets), axis=0
    )
    name_to_position = {}
    for index, name in enumerate(gene_names):
        name_to_position.setdefault(name, index)
    records, pathway_records = [], []
    seen = set(preregistered["targets"]["outcome_fit_seen"])
    for target_index, target in enumerate(scored_targets):
        observed_truth = truth[target]
        target_position = name_to_position.get(target)
        target_excluded = (
            np.arange(len(gene_names))
            if target_position is None
            else np.delete(np.arange(len(gene_names)), target_position)
        )
        regimes = ["all_scored", "outcome_fit_seen" if target in seen else "outcome_fit_unseen"]
        for model, post_profiles in predicted_centroids.items():
            post = post_profiles[target_index]
            predicted_effect = post - matched_controls[target]
            effect_metrics = gene_effect_metrics(
                predicted_effect,
                observed_truth["effect"],
                target_position,
                observed_truth["deg"],
                metrics["retrospective_top_genes"],
            )
            effect_metrics.update(
                {
                    "systema_all_gene_pearson_delta": _correlation(
                        post - perturbed_reference,
                        observed_truth["centroid"] - perturbed_reference,
                    ),
                    "systema_target_excluded_pearson_delta": _correlation(
                        (post - perturbed_reference)[target_excluded],
                        (observed_truth["centroid"] - perturbed_reference)[target_excluded],
                    ),
                    "centroid_accuracy": _centroid_accuracy(post, target_index, truth_centroids),
                    "control_reference_effect_pearson": effect_metrics["all_effect_pearson"],
                    "magnitude_absolute_error": effect_metrics["all_magnitude_absolute_error"],
                }
            )
            pathway_metrics = pathway_agreement(
                predicted_effect,
                observed_truth["effect"],
                pathway_matrix,
                metrics["pathway_top_k"],
            )
            metadata_record = {
                "context": preregistered["source"]["cell_context"],
                "outcome_role": "external_confirmation",
                "target": target,
                "repeat": 0,
                "model": model,
                "truth_batches": observed_truth["batches"],
                "truth_cells": observed_truth["cells"],
                "true_deg_count": int(observed_truth["deg"].sum()),
                "target_in_hvg": target_position is not None,
            }
            for regime in regimes:
                records.append({"regime": regime, **metadata_record, **effect_metrics})
                pathway_records.append({"regime": regime, **metadata_record, **pathway_metrics})
    resamples, seed = metrics["bootstrap_resamples"], preregistered["seed"]
    summary = {
        "condition_metrics": _condition_summary(records, resamples, seed),
        "pathway_metrics": _condition_summary(pathway_records, resamples, seed),
    }
    comparisons = {
        "condition_comparisons": _paired_transcriptomic_models(
            records, config["comparisons"], resamples, seed
        ),
        "pathway_comparisons": _paired_transcriptomic_models(
            pathway_records, config["comparisons"], resamples, seed
        ),
    }
    means = {
        (item["regime"], item["model"], item["metric"]): item["mean"]
        for item in summary["condition_metrics"]
    }
    comparison_index = {
        (item["regime"], item["candidate"], item["reference"], item["metric"]): item
        for item in comparisons["condition_comparisons"]
    }
    candidate, components = (
        "control_gated_external_response",
        [
            "external_response_multiview_rbf",
            "string_kernel_gene_go_rbf",
        ],
    )
    systema_metrics = [
        "systema_all_gene_pearson_delta",
        "systema_target_excluded_pearson_delta",
    ]
    candidate_systema = np.mean(
        [means["all_scored", candidate, metric] for metric in systema_metrics]
    )
    component_systema = {
        model: np.mean([means["all_scored", model, metric] for metric in systema_metrics])
        for model in components
    }
    systematic_comparisons = [
        comparison_index["all_scored", candidate, "perturbed_mean", metric]
        for metric in systema_metrics
    ]
    criteria = {
        "systema_all_gene_ci_lower_above_zero_vs_perturbed_mean": systematic_comparisons[0][
            "mean_improvement_bootstrap_95ci"
        ][0]
        > 0,
        "systema_target_excluded_ci_lower_above_zero_vs_perturbed_mean": systematic_comparisons[1][
            "mean_improvement_bootstrap_95ci"
        ][0]
        > 0,
        "centroid_accuracy_above_perturbed_mean": means[
            "all_scored", candidate, "centroid_accuracy"
        ]
        > means["all_scored", "perturbed_mean", "centroid_accuracy"],
        "systema_loss_vs_best_component_at_most_0.01": max(component_systema.values())
        - candidate_systema
        <= 0.01,
        "centroid_accuracy_loss_vs_best_component_at_most_0.02": max(
            means["all_scored", model, "centroid_accuracy"] for model in components
        )
        - means["all_scored", candidate, "centroid_accuracy"]
        <= 0.02,
        "magnitude_error_degradation_vs_best_component_at_most_2_percent": means[
            "all_scored", candidate, "magnitude_absolute_error"
        ]
        <= 1.02
        * min(means["all_scored", model, "magnitude_absolute_error"] for model in components),
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    decision = {
        "external_confirmation_passes": all(criteria.values()),
        "criteria": criteria,
        "terminal_next_step": (
            "run_one_locked_systema_frontier_comparison_then_finalize"
            if all(criteria.values())
            else "stop_architecture_search_and_finalize_mixed_or_negative_result"
        ),
        "architecture_or_threshold_changes_after_outcomes": "forbidden",
        "global_state_of_the_art_supported_by_adamson_alone": False,
    }
    output = Path(outputs["evaluation_directory"])
    assert not output.exists() and not Path(outputs["evaluation_manifest_path"]).exists()
    output.mkdir(parents=True)
    truth_report = {
        "scored_targets": len(scored_targets),
        "outcome_fit_seen": len(seen),
        "outcome_fit_unseen": len(scored_targets) - len(seen),
        "reference_only_targets": len(reference_targets),
        "observed_frozen_hvg": len(observed),
        "conditions_with_no_degs": sum(not value["deg"].any() for value in truth.values()),
        "batch_matching": "barcode_lane_matched_unpaired_centroids",
        "systema_reference": "equal_condition_mean_of_reference_only_target_centroids",
        "systematic_baseline": "cell_weighted_mean_of_reference_only_target_cells",
    }
    provenance = {
        "config": deepcopy(config),
        "statistical_unit": "perturbation_condition",
        "adamson_outcomes_used_for_fit_selection_or_gate": False,
        "predictions_frozen_before_perturbed_truth_scoring": True,
        "full_adamson_evaluation_index": 1,
        "additional_full_adamson_evaluations_allowed": 0,
        "source_manifests": {**source_manifests, "prediction": prediction_declared},
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    payloads = {
        "summary.json": summary,
        "paired_comparisons.json": comparisons,
        "truth_report.json": truth_report,
        "decision.json": decision,
        "provenance.json": provenance,
    }
    for filename, payload in payloads.items():
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for filename, values in (
        ("condition_metrics.jsonl", records),
        ("pathway_metrics.jsonl", pathway_records),
    ):
        with (output / filename).open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    evaluation_manifest = {
        "format_version": 1,
        "decision": decision,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in sorted(output.iterdir())
        },
        "source": {
            "config_sha256": file_sha256(config_path),
            "prediction_manifest_sha256": prediction_declared,
            **{f"{name}_manifest_sha256": value for name, value in source_manifests.items()},
        },
        "leakage": {
            "adamson_outcomes_used_for_fit_selection_or_gate": False,
            "architecture_or_threshold_changes_after_outcomes": False,
        },
        "provenance": {"git": _git_state(), "runtime_source_sha256": _runtime_source_hash()},
    }
    evaluation_manifest["manifest_sha256"] = sha256(
        json.dumps(evaluation_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    Path(outputs["evaluation_manifest_path"]).write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary, comparisons, truth_report, decision, provenance
