# Verify exact CPU checkpoint/resume behavior for the frozen hybrid dynamics candidate.
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import torch

from causalcelljepa.dynamics import anchored_dynamics_configs, train_dynamics
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state

PATH = Path("configs/salt_hybrid_dynamics.yaml")


def self_hash(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    assert _git_state()["dirty"] is False, "hybrid CPU smoke requires a clean protocol"
    configs, specification = anchored_dynamics_configs(PATH)
    assert list(configs) == ["salt_hybrid_static"]
    config = configs["salt_hybrid_static"]
    smoke = specification["cpu_smoke"]
    root = Path(smoke["output_directory"])
    output, uninterrupted_output = root / "resumed", root / "uninterrupted"
    assert not output.exists() and not uninterrupted_output.exists()

    trial = deepcopy(config)
    trial["training"].update(
        {
            "batch_size": smoke["batch_size"],
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "checkpoint_every_steps": smoke["checkpoint_step"],
            "output_directory": str(output),
        }
    )
    _, first, first_report = train_dynamics(
        trial,
        torch.device("cpu"),
        max_steps=smoke["checkpoint_step"],
        config_path=PATH,
    )
    trial["training"]["resume_from"] = first_report["checkpoint"]
    resumed_model, resumed, resumed_report = train_dynamics(
        trial, torch.device("cpu"), max_steps=smoke["steps"], config_path=PATH
    )

    uninterrupted = deepcopy(trial)
    uninterrupted["training"]["resume_from"] = None
    uninterrupted["training"]["output_directory"] = str(uninterrupted_output)
    uninterrupted_model, uninterrupted_state, uninterrupted_report = train_dynamics(
        uninterrupted, torch.device("cpu"), max_steps=smoke["steps"], config_path=PATH
    )
    exact_model = all(
        torch.equal(resumed_model.state_dict()[key], value)
        for key, value in uninterrupted_model.state_dict().items()
    )
    resumed_events = [
        json.loads(line)
        for line in (output / "training.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "train_step"
    ]
    uninterrupted_events = [
        json.loads(line)
        for line in (uninterrupted_output / "training.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "train_step"
    ]
    exact_log = resumed_events == uninterrupted_events
    assert exact_model and exact_log
    report = {
        "format_version": 1,
        "architecture": specification["revision"]["name"],
        "config_sha256": file_sha256(PATH),
        "first_global_step": first["global_step"],
        "resumed_global_step": resumed["global_step"],
        "uninterrupted_global_step": uninterrupted_state["global_step"],
        "exact_model_resume": exact_model,
        "exact_training_log_resume": exact_log,
        "resumed_train_conditions": resumed_report["train_conditions"],
        "resumed_validation_conditions": resumed_report["validation_conditions"],
        "uninterrupted_train_conditions": uninterrupted_report["train_conditions"],
        "leakage": {
            "context": "K562",
            "sealed_test_outcomes_used": False,
            "rpe1_outcomes_used": False,
        },
    }
    report["manifest_sha256"] = self_hash(report)
    manifest_path = Path(smoke["manifest_path"])
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
