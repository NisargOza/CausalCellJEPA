# Frozen condition-level evaluation on untouched Nadig HepG2 and Jurkat screens.
# External controls and outcomes are sampled independently; no cell pairing is constructed.
import json
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
from causalcelljepa.readout import (
    _bh_adjust,
    _condition_summary,
    _load_pathway_matrix,
    _paired_transcriptomic_models,
    decode_normalized_latents,
    gene_effect_metrics,
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
        self.scale = np.asarray(normalization["latent_std"], dtype=np.float32) * normalization[
            "dimension_scale"
        ]
        assert self.mean.shape == self.scale.shape and np.all(self.scale > 0)
        with h5py.File(self.cache_path, "r") as cache:
            roles = cache["role"].asstr()[:]
            contexts = cache["context"].asstr()[:]
            targets = cache["target"].asstr()[:]
            batches = cache["source_batch"].asstr()[:]
            cell_ids = cache["cell_id"].asstr()[:]
            context_indices = np.flatnonzero(contexts == context)
            self.latent = (
                cache["latent"][context_indices] - self.mean
            ) / self.scale
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
        (squares - np.square(sums) / counts[:, None])
        / np.maximum(counts[:, None] - 1, 1),
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
            source["bytes"], manifest["contexts"][context]["source_sha256"]
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
        matrix, _ = _load_pathway_matrix(
            transcriptomics["inputs"]["go_gmt_path"], overlap_genes
        )
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
                np.abs(effect), np.sqrt(standard_error_squared), out=np.zeros_like(effect),
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
    assert file_sha256(inputs["preregistered_config_path"]) == inputs[
        "preregistered_config_sha256"
    ]
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
        assert cache.attrs["teacher_sha256"] == preregistered["latent_cache"][
            "teacher_sha256"
        ]
        external_cache_provenance = json.loads(cache.attrs["provenance_json"])
        assert external_cache_provenance["git"]["dirty"] is False

    assert file_sha256(inputs["base_evaluation_config_path"]) == inputs[
        "base_evaluation_config_sha256"
    ]
    base_config = yaml.safe_load(Path(inputs["base_evaluation_config_path"]).read_text())
    assert base_config["sampling"]["population_size"] == preregistered["protocol"][
        "population_size"
    ]
    assert base_config["sampling"]["repeats"] == preregistered["protocol"]["repeats"]
    assert set(config["models"]) == set(preregistered["protocol"]["models"])
    assert config["latent_metrics"] == preregistered["protocol"]["metrics"]["latent"]

    models, linear_effect, mean_effect, baseline_report, model_provenance = (
        _load_external_models(config, base_config, device)
    )
    dynamics_manifest = json.loads(Path(base_config["inputs"]["dynamics_manifest_path"]).read_text())
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
                    name: model(control, action, action_known)
                    for name, model in models.items()
                }
                confidence = models["anchored_control_ood_gated"].residual_gate_confidence(
                    control
                )
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
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
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
    assert file_sha256(inputs["preregistered_config_path"]) == inputs[
        "preregistered_config_sha256"
    ]
    preregistered = yaml.safe_load(Path(inputs["preregistered_config_path"]).read_text())
    assert config["transcriptomic_metrics"] == preregistered["protocol"]["metrics"][
        "transcriptomic"
    ]
    manifest, manifest_sha256 = _self_hashed_manifest(
        inputs["external_manifest_path"], inputs["external_manifest_sha256"]
    )
    assert file_sha256(inputs["base_transcriptomics_config_path"]) == inputs[
        "base_transcriptomics_config_sha256"
    ]
    transcriptomics = yaml.safe_load(Path(inputs["base_transcriptomics_config_path"]).read_text())
    replogle, _ = _self_hashed_manifest(
        transcriptomics["inputs"]["replogle_manifest_path"],
        preregistered["inputs"]["replogle_manifest_sha256"],
    )
    for kind in ("readout_checkpoint", "go_gmt"):
        assert file_sha256(transcriptomics["inputs"][f"{kind}_path"]) == transcriptomics[
            "inputs"
        ][f"{kind}_sha256"]
    readout = torch.load(
        transcriptomics["inputs"]["readout_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    readout["weights"] = readout["weights"].to(device)
    readout["bias"] = readout["bias"].to(device)
    base = yaml.safe_load(Path(inputs["base_evaluation_config_path"]).read_text())
    models, linear_effect, mean_effect, baseline_report, model_provenance = (
        _load_external_models(config, base, device)
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
                        decode_normalized_latents(predicted.mean(1), readout)
                        - control_expression
                    ).cpu().numpy()[:, hvg_indices[context]]
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
        normalized_observed = observed / np.linalg.norm(
            observed, axis=1, keepdims=True
        ).clip(1e-12)
        for name in config["models"]:
            predicted = np.stack(
                [
                    signatures[context, name, target][1]
                    / signatures[context, name, target][0]
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
                        "target_in_hvg": hvg_index.get(target)
                        in overlap_positions[context],
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
        "pathway_metrics": _condition_summary(
            pathway_records, resamples, preregistered["seed"]
        ),
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
