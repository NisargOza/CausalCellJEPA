# Export raw ESM and GO beside the frozen masked-teacher joint representation.
# The previously viewed public test split is not evaluated by this revision.
import json
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from causalcelljepa.action_student import (
    MaskedTeacherFusion,
    modality_statistics,
    representation_stable_rank,
    salt_public_split,
    standardized_modalities,
)
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import (
    _git_state,
    _runtime_environment,
    _runtime_source_hash,
)

CONFIG_PATH = Path("configs/salt_hybrid_action.yaml")


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_self_hashed(path, expected):
    payload = json.loads(Path(path).read_text())
    declared = payload.pop("manifest_sha256")
    assert declared == expected == self_hash(payload)
    return payload


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


def main():
    assert _git_state()["dirty"] is False, "hybrid export requires a clean committed protocol"
    config = yaml.safe_load(CONFIG_PATH.read_text())
    source = config["source_action"]
    assert (Path(source["path"]).stat().st_size, file_sha256(source["path"])) == (
        source["bytes"],
        source["sha256"],
    )
    source_manifest = load_self_hashed(source["manifest_path"], source["manifest_sha256"])
    assert source_manifest["artifact"]["sha256"] == source["sha256"]
    source_teacher = config["source_teacher"]
    teacher_run = load_self_hashed(
        source_teacher["run_manifest_path"], source_teacher["run_manifest_sha256"]
    )
    assert teacher_run["source"]["outcomes_read"] is False
    assert teacher_run["teacher"]["best_validation"]["reconstruction"] == source_teacher[
        "best_validation_reconstruction"
    ]
    assert teacher_run["teacher"]["checkpoint"]["sha256"] == source_teacher[
        "checkpoint_sha256"
    ]
    teacher_path = Path(source_teacher["checkpoint_path"])
    assert (teacher_path.stat().st_size, file_sha256(teacher_path)) == (
        source_teacher["checkpoint_bytes"],
        source_teacher["checkpoint_sha256"],
    )

    action = torch.load(source["path"], map_location="cpu", weights_only=True)
    dims = config["teacher"]["modality_dims"]
    feature_width = sum(dims)
    availability = action["embedding"][:, feature_width:].bool()
    split = salt_public_split(
        action["targets"], availability, config["seed"], config["public_split"]["fractions"]
    )
    current_split = split_report(action["targets"], split)
    assert current_split == teacher_run["public_split"]

    means, scales = modality_statistics(action["embedding"], dims, split["train"])
    blocks = standardized_modalities(action["embedding"], dims, means, scales)
    teacher_config = config["teacher"]
    teacher = MaskedTeacherFusion(
        dims, teacher_config["joint_dim"], teacher_config["projector_hidden_dim"]
    )
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    with torch.inference_mode():
        joint = teacher(blocks, availability)
    target_mean = joint[split["train"]].mean(0)
    target_scale = joint[split["train"]].std(0, correction=False).clamp_min(1e-6)
    standardized_joint = (joint - target_mean) / target_scale
    train_mean_error = float(standardized_joint[split["train"]].mean(0).abs().max())
    train_std_error = float(
        (standardized_joint[split["train"]].std(0, correction=False) - 1).abs().max()
    )
    validation_stable_rank = representation_stable_rank(
        standardized_joint[split["validation"]]
    )

    any_available = availability.any(1)
    exported_joint = torch.zeros_like(standardized_joint)
    exported_joint[any_available] = standardized_joint[any_available]
    exported_availability = torch.cat((availability, any_available.unsqueeze(1)), 1)
    action_output = {
        "targets": action["targets"],
        "embedding": torch.cat(
            (
                action["embedding"][:, :feature_width],
                exported_joint,
                exported_availability.float(),
            ),
            1,
        ),
        "known": any_available,
        "modality_dims": config["export"]["modality_dims"],
        "modalities": [
            "esm2_t6_8M_UR50D",
            "go_biological_process_svd",
            "salt_masked_joint_teacher",
        ],
        "modality_availability": True,
    }
    raw_prefix_exact = torch.equal(
        action_output["embedding"][:, :feature_width], action["embedding"][:, :feature_width]
    ) and torch.equal(exported_availability[:, :2], availability)
    gates = config["public_validation_gates"]
    gate_report = {
        "finite_tensors": bool(
            all(torch.isfinite(value).all() for value in teacher.state_dict().values())
            and torch.isfinite(action_output["embedding"]).all()
        ),
        "raw_teacher_prefix_exact": raw_prefix_exact,
        "target_max_absolute_mean": train_mean_error,
        "target_max_absolute_std_error": train_std_error,
        "validation_joint_stable_rank": validation_stable_rank,
        "source_teacher_reconstruction_absolute_error": abs(
            checkpoint["validation"]["reconstruction"]
            - source_teacher["best_validation_reconstruction"]
        ),
    }
    gate_report["passed"] = bool(
        gate_report["finite_tensors"]
        and raw_prefix_exact
        and train_mean_error <= gates["standardized_target_max_absolute_mean"]
        and train_std_error <= gates["standardized_target_max_absolute_std_error"]
        and validation_stable_rank >= gates["minimum_joint_stable_rank"]
        and gate_report["source_teacher_reconstruction_absolute_error"]
        <= gates["source_teacher_reconstruction_absolute_tolerance"]
    )

    provenance = {
        "config_sha256": file_sha256(CONFIG_PATH),
        "source_action_sha256": source["sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_teacher_run_manifest_sha256": source_teacher["run_manifest_sha256"],
        "teacher_checkpoint_sha256": source_teacher["checkpoint_sha256"],
        "runtime_source_sha256": _runtime_source_hash(),
        "runtime_environment": _runtime_environment(),
        "git": _git_state(),
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
            "input_dim": action_output["embedding"].shape[1],
            "modality_dims": action_output["modality_dims"],
            "modality_availability": action_output["modality_availability"],
            "known_targets": int(any_available.sum()),
            "eligible_for_downstream_anchor": gate_report["passed"],
        },
        "public_split": current_split,
        "public_validation_gates": gate_report,
        "teacher": {
            "checkpoint": {
                "path": str(teacher_path),
                "bytes": teacher_path.stat().st_size,
                "sha256": file_sha256(teacher_path),
            },
            "source_best_validation": checkpoint["validation"],
        },
        "source": {
            "action_sha256": source["sha256"],
            "action_manifest_sha256": source["manifest_sha256"],
            "teacher_run_manifest_sha256": source_teacher["run_manifest_sha256"],
            "public_test_read": False,
            "outcomes_read": False,
        },
        "provenance": provenance,
    }
    manifest["manifest_sha256"] = self_hash(manifest)
    manifest_path = Path(config["export"]["manifest_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": manifest["artifact"], "gates": gate_report}, indent=2))


if __name__ == "__main__":
    main()
