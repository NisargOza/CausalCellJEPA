# Train the outcome-free masked ESM+GO teacher, then distill an ESM-only student.
# Export raw frozen ESM beside the student state only after frozen public gates.
import json
from hashlib import sha256
from pathlib import Path

import torch
import yaml
from torch.nn import functional

from causalcelljepa.action_student import (
    MaskedTeacherFusion,
    SaltActionStudent,
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
    epoch_order,
    learning_rate_schedule,
    seed_everything,
)

CONFIG_PATH = Path("configs/salt_action_student.yaml")


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def teacher_validation(model, blocks, indices, variance_weight):
    model.eval()
    reconstructions, variances = [], []
    with torch.inference_mode():
        selected = [block[indices] for block in blocks]
        for hidden in range(2):
            visible = torch.ones(len(indices), 2, dtype=torch.bool)
            visible[:, hidden] = False
            joint = model(selected, visible)
            reconstructions.append(
                functional.smooth_l1_loss(model.decoders[hidden](joint), selected[hidden])
            )
            variances.append(functional.relu(1 - joint.std(0, correction=False)).mean())
    reconstruction = float(torch.stack(reconstructions).mean())
    variance = float(torch.stack(variances).mean())
    return {
        "loss": reconstruction + variance_weight * variance,
        "reconstruction": reconstruction,
        "variance": variance,
    }


def student_validation(model, inputs, targets, indices, variance_weight):
    model.eval()
    with torch.inference_mode():
        prediction, state = model(inputs[indices])
        reconstruction = functional.smooth_l1_loss(prediction, targets[indices])
        variance = functional.relu(1 - prediction.std(0, correction=False)).mean()
        cosine = functional.cosine_similarity(prediction, targets[indices]).mean()
    return {
        "loss": float(reconstruction + variance_weight * variance),
        "reconstruction": float(reconstruction),
        "variance": float(variance),
        "target_cosine": float(cosine),
        "state_stable_rank": representation_stable_rank(state),
    }


def main():
    assert _git_state()["dirty"] is False, "SALT action fitting requires a clean commit"
    config = yaml.safe_load(CONFIG_PATH.read_text())
    source = config["source_action"]
    assert (Path(source["path"]).stat().st_size, file_sha256(source["path"])) == (
        source["bytes"],
        source["sha256"],
    )
    source_manifest = json.loads(Path(source["manifest_path"]).read_text())
    declared = source_manifest.pop("manifest_sha256")
    assert declared == source["manifest_sha256"] == self_hash(source_manifest)
    action = torch.load(source["path"], map_location="cpu", weights_only=True)
    dims = config["teacher"]["modality_dims"]
    availability = action["embedding"][:, sum(dims) :].bool()
    split = salt_public_split(
        action["targets"], availability, config["seed"], config["public_split"]["fractions"]
    )
    means, scales = modality_statistics(action["embedding"], dims, split["train"])
    blocks = standardized_modalities(action["embedding"], dims, means, scales)
    provenance = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_action_sha256": source["sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
    }
    output = Path(config["export"]["teacher_checkpoint_path"]).parent
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])
    teacher_config = config["teacher"]
    teacher = MaskedTeacherFusion(
        dims, teacher_config["joint_dim"], teacher_config["projector_hidden_dim"]
    )
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=teacher_config["learning_rate"],
        weight_decay=teacher_config["weight_decay"],
    )
    best_loss, stale, teacher_history = float("inf"), 0, []
    teacher_path = Path(config["export"]["teacher_checkpoint_path"])
    for epoch in range(teacher_config["epochs"]):
        teacher.train()
        order = epoch_order(split["train"], config["seed"], epoch)
        totals = {"loss": 0.0, "reconstruction": 0.0, "variance": 0.0}
        samples = 0
        for start in range(0, len(order), teacher_config["batch_size"]):
            indices = torch.tensor(order[start : start + teacher_config["batch_size"]])
            selected = [block[indices] for block in blocks]
            hidden = (indices + epoch).remainder(2)
            visible = torch.ones(len(indices), 2, dtype=torch.bool)
            visible[torch.arange(len(indices)), hidden] = False
            joint = teacher(selected, visible)
            losses = [
                functional.smooth_l1_loss(
                    teacher.decoders[modality](joint[hidden == modality]),
                    selected[modality][hidden == modality],
                )
                for modality in range(2)
                if bool((hidden == modality).any())
            ]
            reconstruction = torch.stack(losses).mean()
            variance = functional.relu(1 - joint.std(0, correction=False)).mean()
            loss = reconstruction + teacher_config["variance_weight"] * variance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("reconstruction", reconstruction),
                ("variance", variance),
            ):
                totals[name] += float(value.detach()) * len(indices)
            samples += len(indices)
        validation = teacher_validation(
            teacher, blocks, split["validation"], teacher_config["variance_weight"]
        )
        row = {
            "epoch": epoch,
            "train": {name: value / samples for name, value in totals.items()},
            "validation": validation,
        }
        teacher_history.append(row)
        with (output / "teacher_training.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        improved = validation["loss"] < best_loss
        if improved:
            best_loss, stale = validation["loss"], 0
            save_artifact(
                teacher_path,
                {
                    "model": teacher.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "provenance": provenance,
                },
            )
        else:
            stale += 1
        if stale >= teacher_config["early_stopping_patience"]:
            break
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    with torch.inference_mode():
        joint = teacher(blocks, availability)
    target_mean = joint[split["train"]].mean(0)
    target_scale = joint[split["train"]].std(0, correction=False).clamp_min(1e-6)
    targets = (joint - target_mean) / target_scale
    target_mean_error = float(targets[split["train"]].mean(0).abs().max())
    target_std_error = float((targets[split["train"]].std(0, correction=False) - 1).abs().max())

    student_config = config["student"]
    inputs = blocks[0]
    seed_everything(config["seed"] + 1)
    student = SaltActionStudent(
        student_config["input_dim"], student_config["hidden_dim"], student_config["output_dim"]
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=student_config["learning_rate"],
        weight_decay=student_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_schedule(
            student_config["epochs"],
            student_config["warmup_fraction"],
            student_config["minimum_learning_rate_fraction"],
        ),
    )
    best_loss, stale, student_history = float("inf"), 0, []
    student_path = Path(config["export"]["student_checkpoint_path"])
    for epoch in range(student_config["epochs"]):
        student.train()
        order = epoch_order(split["train"], config["seed"] + 1, epoch)
        total, samples = 0.0, 0
        for start in range(0, len(order), student_config["batch_size"]):
            indices = torch.tensor(order[start : start + student_config["batch_size"]])
            prediction, _ = student(inputs[indices])
            reconstruction = functional.smooth_l1_loss(prediction, targets[indices])
            variance = functional.relu(1 - prediction.std(0, correction=False)).mean()
            loss = reconstruction + student_config["variance_weight"] * variance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            samples += len(indices)
        scheduler.step()
        validation = student_validation(
            student, inputs, targets, split["validation"], student_config["variance_weight"]
        )
        row = {"epoch": epoch, "train_loss": total / samples, "validation": validation}
        student_history.append(row)
        with (output / "student_training.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        improved = validation["loss"] < best_loss
        if improved:
            best_loss, stale = validation["loss"], 0
            save_artifact(
                student_path,
                {
                    "model": student.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "provenance": provenance,
                },
            )
        else:
            stale += 1
        if stale >= student_config["early_stopping_patience"]:
            break
    student_checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    student.load_state_dict(student_checkpoint["model"])
    test = student_validation(
        student, inputs, targets, split["test"], student_config["variance_weight"]
    )
    student.eval()
    with torch.inference_mode():
        _, states = student(inputs)

    ridge_reports = []
    train_x = torch.cat((inputs[split["train"]], torch.ones(len(split["train"]), 1)), 1)
    validation_x = torch.cat(
        (inputs[split["validation"]], torch.ones(len(split["validation"]), 1)), 1
    )
    test_x = torch.cat((inputs[split["test"]], torch.ones(len(split["test"]), 1)), 1)
    identity = torch.eye(train_x.shape[1])
    identity[-1, -1] = 0
    for ridge in student_config["ridge_candidates"]:
        weights = torch.linalg.solve(
            train_x.T @ train_x + ridge * identity, train_x.T @ targets[split["train"]]
        )
        validation_prediction = validation_x @ weights
        ridge_reports.append(
            {
                "ridge": ridge,
                "weights": weights,
                "validation_loss": float(
                    functional.smooth_l1_loss(validation_prediction, targets[split["validation"]])
                ),
            }
        )
    ridge_selected = min(ridge_reports, key=lambda row: row["validation_loss"])
    ridge_test_prediction = test_x @ ridge_selected["weights"]
    ridge_report = {
        "selected": ridge_selected["ridge"],
        "validation_loss": ridge_selected["validation_loss"],
        "test_loss": float(
            functional.smooth_l1_loss(ridge_test_prediction, targets[split["test"]])
        ),
        "test_target_cosine": float(
            functional.cosine_similarity(ridge_test_prediction, targets[split["test"]]).mean()
        ),
    }

    train, test_indices = split["train"], split["test"]
    residual = torch.cat(
        (functional.normalize(inputs, dim=1), functional.normalize(states, dim=1)), 1
    )
    esm_overlap = teacher_neighbor_overlap(
        inputs[test_indices], inputs[train], blocks[1][test_indices], blocks[1][train]
    )
    residual_overlap = teacher_neighbor_overlap(
        residual[test_indices], residual[train], blocks[1][test_indices], blocks[1][train]
    )
    gates = config["public_gates"]
    gate_report = {
        "checkpoint_tensors_finite": all(
            torch.isfinite(value).all()
            for value in (*teacher.state_dict().values(), *student.state_dict().values())
        ),
        "target_max_absolute_mean": target_mean_error,
        "target_max_absolute_std_error": target_std_error,
        "student_test_target_cosine": test["target_cosine"],
        "student_test_stable_rank": test["state_stable_rank"],
        "esm_go_neighbor_overlap_at_10": esm_overlap,
        "residual_go_neighbor_overlap_at_10": residual_overlap,
        "residual_go_neighbor_overlap_gain": residual_overlap - esm_overlap,
    }
    gate_report["passed"] = bool(
        gate_report["checkpoint_tensors_finite"]
        and target_mean_error <= gates["standardized_target_max_absolute_mean"]
        and target_std_error <= gates["standardized_target_max_absolute_std_error"]
        and test["target_cosine"] >= gates["student_test_minimum_target_cosine"]
        and test["state_stable_rank"] >= gates["student_test_minimum_stable_rank"]
        and residual_overlap - esm_overlap
        >= gates["residual_minimum_go_neighbor_overlap_gain_over_esm"]
    )

    esm_known = availability[:, 0]
    exported_states = torch.zeros_like(states)
    exported_states[esm_known] = states[esm_known]
    exported_availability = torch.stack((esm_known, esm_known), 1)
    action_output = {
        "targets": action["targets"],
        "embedding": torch.cat(
            (
                action["embedding"][:, : dims[0]],
                exported_states,
                exported_availability.float(),
            ),
            1,
        ),
        "known": esm_known,
        "modality_dims": config["export"]["modality_dims"],
        "modalities": ["esm2_t6_8M_UR50D", "salt_distilled_joint"],
        "modality_availability": True,
    }
    action_path = Path(config["export"]["action_cache_path"])
    save_artifact(action_path, action_output)
    split_report = {
        name: {
            "targets": len(indices),
            "target_sha256": sha256(
                "\n".join(sorted(action["targets"][index] for index in indices)).encode()
            ).hexdigest(),
        }
        for name, indices in split.items()
    }
    manifest = {
        "format_version": 1,
        "architecture": config["revision"]["name"],
        "artifact": {
            "path": str(action_path),
            "bytes": action_path.stat().st_size,
            "sha256": file_sha256(action_path),
            "modality_dims": action_output["modality_dims"],
            "known_targets": int(esm_known.sum()),
        },
        "public_split": split_report,
        "teacher": {
            "best_epoch": teacher_checkpoint["epoch"],
            "best_validation": teacher_checkpoint["validation"],
            "epochs_run": len(teacher_history),
            "checkpoint": {
                "path": str(teacher_path),
                "bytes": teacher_path.stat().st_size,
                "sha256": file_sha256(teacher_path),
            },
        },
        "student": {
            "best_epoch": student_checkpoint["epoch"],
            "best_validation": student_checkpoint["validation"],
            "epochs_run": len(student_history),
            "test": test,
            "ridge_baseline": ridge_report,
            "checkpoint": {
                "path": str(student_path),
                "bytes": student_path.stat().st_size,
                "sha256": file_sha256(student_path),
            },
        },
        "public_gates": gate_report,
        "source": {
            "config_sha256": file_sha256(CONFIG_PATH),
            "action_sha256": source["sha256"],
            "outcomes_read": False,
        },
        "provenance": provenance,
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    Path(config["export"]["manifest_path"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"public_gates": gate_report, "student_test": test, "ridge": ridge_report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
