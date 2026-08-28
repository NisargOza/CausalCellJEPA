# Frozen condition-level evaluation on untouched Nadig HepG2 and Jurkat screens.
# External controls and outcomes are sampled independently; no cell pairing is constructed.
import json
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
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
        with h5py.File(self.cache_path, "r") as cache:
            roles = cache["role"].asstr()[:]
            contexts = cache["context"].asstr()[:]
            targets = cache["target"].asstr()[:]
            batches = cache["source_batch"].asstr()[:]
            cell_ids = cache["cell_id"].asstr()[:]
        selected_context = contexts == context
        control_indices = np.flatnonzero(selected_context & (roles == "external_control"))
        outcome_indices = np.flatnonzero(selected_context & (roles == "external_test"))
        assert len(control_indices) >= population_size and len(outcome_indices)
        assert set(batches[np.concatenate((control_indices, outcome_indices))]) == {"unavailable"}
        assert len(np.unique(cell_ids[np.concatenate((control_indices, outcome_indices))])) == (
            len(control_indices) + len(outcome_indices)
        )
        self.controls = control_indices
        self.condition_targets = sorted(set(targets[outcome_indices]))
        if expected_targets is not None:
            assert self.condition_targets == sorted(expected_targets)
        self.outcomes = {
            target: outcome_indices[targets[outcome_indices] == target]
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

        manifest = json.loads(Path(dynamics_manifest_path).read_text())
        normalization = manifest["normalization"]
        self.mean = np.asarray(normalization["latent_mean"], dtype=np.float32)
        self.scale = np.asarray(normalization["latent_std"], dtype=np.float32) * normalization[
            "dimension_scale"
        ]
        assert self.mean.shape == self.scale.shape and np.all(self.scale > 0)
        self._cache = None

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
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r")
        controls, outcomes, target = self.sample_indices(index)
        populations = []
        for indices in (controls, outcomes):
            order = np.argsort(indices)
            latent = self._cache["latent"][indices[order]][np.argsort(order)]
            populations.append(torch.from_numpy((latent - self.mean) / self.scale))
        action, known = self.action[target]
        return {
            "control": populations[0],
            "perturbed": populations[1],
            "action": action,
            "action_known": known,
            "target": target,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


def _load_external_models(config, base_config):
    inputs = config["inputs"]
    primary_path = Path(inputs["primary_checkpoint_path"])
    assert (primary_path.stat().st_size, file_sha256(primary_path)) == (
        inputs["primary_checkpoint_bytes"],
        inputs["primary_checkpoint_sha256"],
    )
    primary_checkpoint = torch.load(primary_path, map_location="cpu", weights_only=False)
    primary = build_dynamics_model(primary_checkpoint["configuration"]).eval()
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

    anchored = build_dynamics_model(anchored_checkpoint["configuration"]).eval()
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


@torch.inference_mode()
def run_nadig_external_latent_evaluation(
    config_path="configs/nadig_external_evaluation.yaml",
    repeats=None,
    maximum_conditions=None,
    output_directory=None,
):
    """Evaluate all frozen models without fitting or selecting on external outcomes."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
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
        _load_external_models(config, base_config)
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
                control, observed = batch["control"], batch["perturbed"]
                learned = {
                    name: model(control, batch["action"], batch["action_known"])
                    for name, model in models.items()
                }
                confidence = models["anchored_control_ood_gated"].residual_gate_confidence(
                    control
                )
                gate_confidences[context].extend(confidence.tolist())
                predictions = {
                    **learned,
                    "linear_esm": control
                    + torch.from_numpy(
                        np.stack([linear_effect[target] for target in batch["target"]])
                    )[:, None],
                    "mean_effect": control + torch.from_numpy(mean_effect)[None, None],
                    "no_change": control,
                }
                assert set(predictions) == set(config["models"])
                assert all(value.dtype == observed.dtype for value in predictions.values())
                true_effect = observed.mean(1) - control.mean(1)
                for index, target in enumerate(batch["target"]):
                    key = (context, "true", target)
                    signatures[key][0] += 1
                    value = true_effect[index].numpy()
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
                            value = float(values[index])
                            record[metric] = value if np.isfinite(value) else None
                        records.append(record)
                        key = (context, name, target)
                        signatures[key][0] += 1
                        value = predicted_effect[index].numpy()
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
