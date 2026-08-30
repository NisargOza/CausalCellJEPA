# Replay the frozen masked ESM+GO teacher and export its validation-selected ridge student.
# Perturbation outcomes are outside this script's input boundary.
import json
from hashlib import sha256
from pathlib import Path

import torch
import yaml
from torch.nn import functional

from causalcelljepa.action_student import (
    MaskedTeacherFusion,
    apply_ridge_student,
    fit_ridge_student,
    modality_statistics,
    representation_stable_rank,
    salt_public_split,
    standardized_modalities,
    teacher_neighbor_overlap,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import (
    _git_state,
    _runtime_environment,
    _runtime_source_hash,
    seed_everything,
)

CONFIG_PATH = Path("configs/salt_ridge_action.yaml")


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def split_report(targets, split):
    return {
        name: {
            "targets": len(indices),
            "target_sha256": sha256(
                "\n".join(sorted(targets[index] for index in indices)).encode()
            ).hexdigest(),
        }
        for name, indices in split.items()
    }


def ridge_metrics(prediction, target):
    return {
        "loss": float(functional.smooth_l1_loss(prediction, target)),
        "target_cosine": float(functional.cosine_similarity(prediction, target).mean()),
        "stable_rank": representation_stable_rank(prediction),
    }


def main():
    assert _git_state()["dirty"] is False, "ridge export requires a clean committed protocol"
    config = yaml.safe_load(CONFIG_PATH.read_text())
    source = config["source_action"]
    assert (Path(source["path"]).stat().st_size, file_sha256(source["path"])) == (
        source["bytes"],
        source["sha256"],
    )
    source_manifest = json.loads(Path(source["manifest_path"]).read_text())
    source_declared = source_manifest.pop("manifest_sha256")
    assert source_declared == source["manifest_sha256"] == self_hash(source_manifest)

    source_run = config["source_run"]
    source_run_manifest = json.loads(Path(source_run["manifest_path"]).read_text())
    source_run_declared = source_run_manifest.pop("manifest_sha256")
    assert source_run_declared == source_run["manifest_sha256"] == self_hash(source_run_manifest)
    assert source_run["public_test_status"].endswith("sealed_reporting_only")
    assert source_run_manifest["source"]["outcomes_read"] is False
    assert source_run_manifest["student"]["ridge_baseline"]["selected"] == source_run[
        "selected_ridge"
    ]
    assert source_run_manifest["student"]["ridge_baseline"]["validation_loss"] == source_run[
        "selected_ridge_validation_loss"
    ]
    assert source_run_manifest["student"]["best_validation"]["reconstruction"] == source_run[
        "nonlinear_validation_reconstruction"
    ]

    teacher_path = Path(source_run["teacher_checkpoint_path"])
    assert (teacher_path.stat().st_size, file_sha256(teacher_path)) == (
        source_run["teacher_checkpoint_bytes"],
        source_run["teacher_checkpoint_sha256"],
    )
    assert source_run_manifest["teacher"]["checkpoint"]["sha256"] == source_run[
        "teacher_checkpoint_sha256"
    ]

    action = torch.load(source["path"], map_location="cpu", weights_only=True)
    dims = config["teacher"]["modality_dims"]
    availability = action["embedding"][:, sum(dims) :].bool()
    split = salt_public_split(
        action["targets"], availability, config["seed"], config["public_split"]["fractions"]
    )
    current_split_report = split_report(action["targets"], split)
    assert current_split_report == source_run_manifest["public_split"]

    means, scales = modality_statistics(action["embedding"], dims, split["train"])
    blocks = standardized_modalities(action["embedding"], dims, means, scales)
    teacher_config = config["teacher"]
    teacher = MaskedTeacherFusion(
        dims, teacher_config["joint_dim"], teacher_config["projector_hidden_dim"]
    )
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    with torch.inference_mode():
        joint = teacher(blocks, availability)
    target_mean = joint[split["train"]].mean(0)
    target_scale = joint[split["train"]].std(0, correction=False).clamp_min(1e-6)
    targets = (joint - target_mean) / target_scale
    target_mean_error = float(targets[split["train"]].mean(0).abs().max())
    target_std_error = float(
        (targets[split["train"]].std(0, correction=False) - 1).abs().max()
    )

    seed_everything(config["seed"] + 2)
    inputs = blocks[0]
    candidate_reports = []
    candidate_weights = {}
    for ridge in config["ridge_student"]["candidates"]:
        weights = fit_ridge_student(inputs[split["train"]], targets[split["train"]], ridge)
        validation_prediction = apply_ridge_student(inputs[split["validation"]], weights)
        candidate_reports.append(
            {
                "ridge": ridge,
                "validation_loss": float(
                    functional.smooth_l1_loss(
                        validation_prediction, targets[split["validation"]]
                    )
                ),
            }
        )
        candidate_weights[ridge] = weights
    selected = min(candidate_reports, key=lambda row: row["validation_loss"])
    weights = candidate_weights[selected["ridge"]]
    predictions = apply_ridge_student(inputs, weights)

    validation_indices = split["validation"]
    test_indices = split["test"]
    train_indices = split["train"]
    validation = ridge_metrics(predictions[validation_indices], targets[validation_indices])
    test = ridge_metrics(predictions[test_indices], targets[test_indices])

    residual = torch.cat(
        (functional.normalize(inputs, dim=1), functional.normalize(predictions, dim=1)), 1
    )
    validation_esm_overlap = teacher_neighbor_overlap(
        inputs[validation_indices],
        inputs[train_indices],
        blocks[1][validation_indices],
        blocks[1][train_indices],
    )
    validation_residual_overlap = teacher_neighbor_overlap(
        residual[validation_indices],
        residual[train_indices],
        blocks[1][validation_indices],
        blocks[1][train_indices],
    )
    test_esm_overlap = teacher_neighbor_overlap(
        inputs[test_indices], inputs[train_indices], blocks[1][test_indices], blocks[1][train_indices]
    )
    test_residual_overlap = teacher_neighbor_overlap(
        residual[test_indices],
        residual[train_indices],
        blocks[1][test_indices],
        blocks[1][train_indices],
    )

    replay_loss_error = abs(selected["validation_loss"] - source_run["selected_ridge_validation_loss"])
    validation_loss_improvement = (
        source_run["nonlinear_validation_reconstruction"] - selected["validation_loss"]
    )
    gates = config["public_validation_gates"]
    gate_report = {
        "finite_tensors": bool(
            all(torch.isfinite(value).all() for value in teacher.state_dict().values())
            and torch.isfinite(weights).all()
            and torch.isfinite(predictions).all()
        ),
        "target_max_absolute_mean": target_mean_error,
        "target_max_absolute_std_error": target_std_error,
        "selected_ridge": selected["ridge"],
        "selected_ridge_replay_absolute_error": replay_loss_error,
        "validation_loss_improvement_over_nonlinear_student": validation_loss_improvement,
        "validation_target_cosine": validation["target_cosine"],
        "validation_stable_rank": validation["stable_rank"],
        "validation_esm_go_neighbor_overlap_at_10": validation_esm_overlap,
        "validation_residual_go_neighbor_overlap_at_10": validation_residual_overlap,
        "validation_residual_go_neighbor_overlap_gain": (
            validation_residual_overlap - validation_esm_overlap
        ),
    }
    gate_report["passed"] = bool(
        gate_report["finite_tensors"]
        and target_mean_error <= gates["standardized_target_max_absolute_mean"]
        and target_std_error <= gates["standardized_target_max_absolute_std_error"]
        and selected["ridge"] == source_run["selected_ridge"]
        and replay_loss_error <= gates["selected_ridge_replay_absolute_tolerance"]
        and validation_loss_improvement
        >= gates["minimum_loss_improvement_over_nonlinear_student"]
        and validation["target_cosine"] >= gates["minimum_target_cosine"]
        and validation["stable_rank"] >= gates["minimum_stable_rank"]
        and validation_residual_overlap - validation_esm_overlap
        >= gates["residual_minimum_go_neighbor_overlap_gain_over_esm"]
    )

    provenance = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_action_sha256": source["sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_run_manifest_sha256": source_run["manifest_sha256"],
        "teacher_checkpoint_sha256": source_run["teacher_checkpoint_sha256"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    checkpoint_path = Path(config["export"]["checkpoint_path"])
    save_artifact(
        checkpoint_path,
        {
            "weights": weights,
            "selected_ridge": selected["ridge"],
            "modality_means": means,
            "modality_scales": scales,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "validation": validation,
            "provenance": provenance,
        },
    )

    esm_known = availability[:, 0]
    exported_predictions = torch.zeros_like(predictions)
    exported_predictions[esm_known] = predictions[esm_known]
    exported_availability = torch.stack((esm_known, esm_known), 1)
    action_output = {
        "targets": action["targets"],
        "embedding": torch.cat(
            (
                action["embedding"][:, : dims[0]],
                exported_predictions,
                exported_availability.float(),
            ),
            1,
        ),
        "known": esm_known,
        "modality_dims": config["export"]["modality_dims"],
        "modalities": ["esm2_t6_8M_UR50D", "salt_ridge_distilled_joint"],
        "modality_availability": True,
    }
    action_path = Path(config["export"]["action_cache_path"])
    save_artifact(action_path, action_output)

    manifest = {
        "format_version": 1,
        "architecture": config["revision"]["name"],
        "artifact": {
            "path": str(action_path),
            "bytes": action_path.stat().st_size,
            "sha256": file_sha256(action_path),
            "modality_dims": action_output["modality_dims"],
            "input_dim": action_output["embedding"].shape[1],
            "modality_availability": action_output["modality_availability"],
            "known_targets": int(esm_known.sum()),
            "eligible_for_downstream_anchor": gate_report["passed"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
        },
        "public_split": current_split_report,
        "selection": {
            "candidates": candidate_reports,
            "selected_ridge": selected["ridge"],
            "selected_validation_loss": selected["validation_loss"],
            "nonlinear_validation_reconstruction": source_run[
                "nonlinear_validation_reconstruction"
            ],
            "validation_loss_improvement": validation_loss_improvement,
            "refit_after_selection": False,
        },
        "public_validation": validation,
        "public_validation_gates": gate_report,
        "public_test_reporting_only": {
            **test,
            "esm_go_neighbor_overlap_at_10": test_esm_overlap,
            "residual_go_neighbor_overlap_at_10": test_residual_overlap,
            "residual_go_neighbor_overlap_gain": test_residual_overlap - test_esm_overlap,
            "selection_influence": False,
        },
        "source": {
            "action_sha256": source["sha256"],
            "source_run_manifest_sha256": source_run["manifest_sha256"],
            "public_test_status": source_run["public_test_status"],
            "outcomes_read": False,
        },
        "provenance": provenance,
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    manifest_path = Path(config["export"]["manifest_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "public_validation_gates": gate_report,
                "public_validation": validation,
                "public_test_reporting_only": manifest["public_test_reporting_only"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
