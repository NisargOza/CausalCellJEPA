"""Validation-only selection for explicit post-primary architecture revisions."""

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from causalcelljepa.dynamics import (
    FrozenLowRankEffectAnchor,
    LatentPopulationDataset,
    anchored_dynamics_configs,
    build_dynamics_model,
)
from causalcelljepa.evaluation import population_metrics
from causalcelljepa.readout import (
    decode_normalized_latents,
    expression_truth,
    gene_effect_metrics,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash


def _device_readout(checkpoint, device):
    return {
        **checkpoint,
        "weights": checkpoint["weights"].to(device),
        "bias": checkpoint["bias"].to(device),
    }


def _mean_metrics(records):
    grouped = defaultdict(list)
    excluded = {"candidate", "target", "repeat"}
    for record in records:
        for metric, value in record.items():
            if metric not in excluded and value is not None:
                grouped[record["candidate"], metric].append(value)
    return {
        candidate: {
            metric: float(np.mean(values))
            for (name, metric), values in sorted(grouped.items())
            if name == candidate
        }
        for candidate in sorted({record["candidate"] for record in records})
    }


@torch.inference_mode()
def run_anchored_validation(
    path="configs/anchored_dynamics.yaml",
    checkpoint_paths=None,
    maximum_conditions=None,
    repeats=None,
    output_directory=None,
    device=None,
    write_decision=True,
):
    """Select one frozen architecture using K562 validation outcomes and nothing else."""
    device = device or torch.device("cpu")
    path = Path(path)
    configs, specification = anchored_dynamics_configs(path)
    selection = specification["selection"]
    assert (
        selection["context"],
        selection["outcome_role"],
        selection["control_role"],
    ) == ("K562", "perturbation_ood_validation", "control_train")
    transcriptomics_path = Path(selection["transcriptomics_config_path"])
    assert file_sha256(transcriptomics_path) == selection["transcriptomics_config_sha256"]
    transcriptomics = yaml.safe_load(transcriptomics_path.read_text())
    inputs = transcriptomics["inputs"]
    for kind in (
        "latent_cache",
        "expression_cache",
        "action_cache",
        "checkpoint",
        "readout_checkpoint",
    ):
        assert file_sha256(inputs[f"{kind}_path"]) == inputs[f"{kind}_sha256"]
    assert all(
        config["inputs"]["latent_cache_sha256"] == inputs["latent_cache_sha256"]
        and config["inputs"]["action_cache_sha256"] == inputs["action_cache_sha256"]
        for config in configs.values()
    )

    checkpoints = {}
    for name, config in configs.items():
        checkpoint_path = Path(
            checkpoint_paths[name]
            if checkpoint_paths is not None
            else Path(config["training"]["output_directory"]) / "best.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["configuration"]["revision"]["candidate"] == name
        model = build_dynamics_model(checkpoint["configuration"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        checkpoints[name] = {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "model": model,
            "state": checkpoint["state"],
        }

    primary_checkpoint = torch.load(
        inputs["checkpoint_path"], map_location="cpu", weights_only=False
    )
    primary_model = build_dynamics_model(primary_checkpoint["configuration"]).to(device).eval()
    primary_model.load_state_dict(primary_checkpoint["model"])
    anchor_checkpoint = torch.load(
        specification["effect_anchor"]["output_path"], map_location="cpu", weights_only=True
    )
    anchor_model = FrozenLowRankEffectAnchor(anchor_checkpoint).to(device).eval()
    readout = _device_readout(
        torch.load(inputs["readout_checkpoint_path"], map_location="cpu", weights_only=False),
        device,
    )

    regime = {
        "validation": {
            "context": "K562",
            "outcome_role": "perturbation_ood_validation",
            "control_role": "control_train",
        }
    }
    truth, truth_report = expression_truth(
        transcriptomics, regime, maximum_conditions=maximum_conditions
    )
    dataset = LatentPopulationDataset(
        inputs["latent_cache_path"],
        inputs["action_cache_path"],
        configs["anchor_only"]["inputs"]["dynamics_manifest_path"],
        "architecture_validation",
        configs["anchor_only"]["data"]["population_size"],
        specification["seed"],
        "perturbation_ood_validation",
        "control_train",
        "K562",
    )
    indices = range(min(len(dataset), maximum_conditions or len(dataset)))
    dynamics_manifest = json.loads(
        Path(configs["anchor_only"]["inputs"]["dynamics_manifest_path"]).read_text()
    )
    median_distance = dynamics_manifest["normalization"]["median_training_latent_distance"]
    metric_config = {
        "reference_sinkhorn_blur_ratio": selection["reference_sinkhorn_blur_ratio"],
        "mmd_bandwidth_ratio": selection["mmd_bandwidth_ratio"],
    }
    records = []
    for repeat in range(repeats or selection["repeats"]):
        dataset.set_epoch(repeat)
        loader = DataLoader(
            dataset,
            batch_size=selection["batch_size"],
            sampler=list(indices),
            num_workers=selection["num_workers"],
        )
        for batch in loader:
            control = batch["control"].to(device)
            observed = batch["perturbed"].to(device)
            action = batch["action"].to(device)
            known = batch["action_known"].to(device)
            anchor = anchor_model(action, known)
            predictions = {
                name: value["model"](control, action, known)
                for name, value in checkpoints.items()
            }
            predictions.update(
                {
                    "original_primary": primary_model(control, action, known),
                    "linear_anchor": control + anchor.unsqueeze(1),
                }
            )
            control_expression = decode_normalized_latents(control.mean(1), readout)
            for name, predicted in predictions.items():
                latent_metrics = population_metrics(
                    predicted, observed, control, median_distance, metric_config
                )
                decoded_effect = (
                    decode_normalized_latents(predicted.mean(1), readout) - control_expression
                ).cpu().numpy()
                for index, target in enumerate(batch["target"]):
                    observed_effect = truth["validation", target]["effect"]
                    gene_metrics = gene_effect_metrics(
                        decoded_effect[index],
                        observed_effect,
                        None,
                        truth["validation", target]["deg"],
                        (),
                    )
                    records.append(
                        {
                            "candidate": name,
                            "target": target,
                            "repeat": repeat,
                            "decoded_all_effect_pearson": gene_metrics[
                                "all_effect_pearson"
                            ],
                            "decoded_all_effect_spearman": gene_metrics[
                                "all_effect_spearman"
                            ],
                            "decoded_all_magnitude_absolute_error": gene_metrics[
                                "all_magnitude_absolute_error"
                            ],
                            **{
                                f"latent_{metric}": float(value[index].cpu())
                                for metric, value in latent_metrics.items()
                            },
                        }
                    )
    summaries = _mean_metrics(records)
    reference = summaries[selection["guardrail_reference"]]
    eligibility = {"anchor_only": {"eligible": True, "reasons": []}}
    for name in ("anchor_residual_025", "anchor_residual_050"):
        reasons = []
        if not np.isfinite(list(summaries[name].values())).all():
            reasons.append("non_finite_metric")
        if summaries[name]["latent_sinkhorn"] > (
            selection["maximum_sinkhorn_ratio"] * reference["latent_sinkhorn"]
        ):
            reasons.append("sinkhorn_guardrail")
        if summaries[name]["decoded_all_magnitude_absolute_error"] > (
            selection["maximum_magnitude_absolute_error_ratio"]
            * reference["decoded_all_magnitude_absolute_error"]
        ):
            reasons.append("magnitude_guardrail")
        eligibility[name] = {"eligible": not reasons, "reasons": reasons}

    eligible = [name for name, value in eligibility.items() if value["eligible"]]
    best_effect = max(summaries[name][selection["primary_metric"]] for name in eligible)
    practical_ties = [
        name
        for name in eligible
        if summaries[name][selection["primary_metric"]]
        >= best_effect - selection["practical_tie_margin"]
    ]
    ratios = {
        name: specification["experiments"][name]["mean_residual_max_ratio"]
        for name in configs
    }
    selected = min(
        practical_ties,
        key=lambda name: (summaries[name]["latent_sinkhorn"], ratios[name]),
    )
    selected_checkpoint = checkpoints[selected]
    decision = {
        "format_version": 1,
        "revision": specification["revision"],
        "selection_rule": selection,
        "candidate_summaries": summaries,
        "eligibility": eligibility,
        "practical_ties": practical_ties,
        "selected": {
            key: selected_checkpoint[key] for key in ("path", "bytes", "sha256")
        }
        | {
            "candidate": selected,
            "mean_residual_max_ratio": ratios[selected],
            "best_validation_epoch": selected_checkpoint["state"][
                "best_validation_epoch"
            ],
            "best_validation_loss": selected_checkpoint["state"][
                "best_validation_loss"
            ],
        },
        "truth_report": truth_report,
        "leakage": {
            "context": "K562",
            "outcome_role": "perturbation_ood_validation",
            "sealed_test_outcomes_used": False,
            "rpe1_outcomes_used": False,
        },
        "provenance": {
            "config_sha256": file_sha256(path),
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    }
    output = Path(output_directory or selection["output_directory"])
    assert not output.exists()
    output.mkdir(parents=True)
    with (output / "condition_metrics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if write_decision:
        decision["artifacts"] = {
            name: {
                "path": str(output / name),
                "bytes": (output / name).stat().st_size,
                "sha256": file_sha256(output / name),
            }
            for name in ("condition_metrics.jsonl", "decision.json")
        }
        decision["manifest_sha256"] = sha256(
            json.dumps(decision, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        Path(selection["decision_manifest_path"]).write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n"
        )
    return decision
