# Train each locked Sinkhorn blur candidate identically on K562 training outcomes.
# Select only by fixed-reference Sinkhorn on all perturbation-OOD validation targets.
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from causalcelljepa.dynamics import LatentPopulationDataset, train_dynamics, validate_dynamics
from causalcelljepa.resources import file_sha256

config = yaml.safe_load(Path("configs/dynamics.yaml").read_text())
assert torch.cuda.is_available(), "Sinkhorn blur selection requires CUDA; refusing CPU fallback"
selection = config["blur_selection"]
assert selection["validation_metric"] == "sinkhorn"
root = Path(selection["output_directory"])
root.mkdir(parents=True, exist_ok=True)
results = []
for candidate in config["loss"]["sinkhorn_blur_candidates"]:
    candidate_config = deepcopy(config)
    candidate_config["loss"]["sinkhorn_blur_ratio"] = candidate
    output = root / f"blur_{candidate:.2f}".replace(".", "p")
    candidate_config["training"]["output_directory"] = str(output)
    latest, result_path = output / "latest.pt", output / "candidate_report.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["candidate_blur_ratio"] == candidate
        assert result["reference_blur_ratio"] == selection["reference_blur_ratio"]
        results.append(result)
        continue
    if latest.exists():
        candidate_config["training"]["resume_from"] = str(latest)
    model, state, training = train_dynamics(candidate_config, torch.device("cuda"))
    evaluation_config = deepcopy(candidate_config)
    evaluation_config["loss"]["sinkhorn_blur_ratio"] = selection["reference_blur_ratio"]
    validation = LatentPopulationDataset(
        config["inputs"]["latent_cache_path"],
        config["inputs"]["action_cache_path"],
        config["inputs"]["dynamics_manifest_path"],
        "validation",
        config["data"]["population_size"],
        config["seed"],
    )
    metrics = validate_dynamics(model, validation, evaluation_config, torch.device("cuda"))
    best = Path(training["best_checkpoint"])
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    result = {
        "candidate_blur_ratio": candidate,
        "reference_blur_ratio": selection["reference_blur_ratio"],
        "selection_metric": selection["validation_metric"],
        "selection_value": metrics[selection["validation_metric"]],
        "validation_conditions": len(validation),
        "validation": metrics,
        "training_state": state,
        "training_report": training,
        "best_checkpoint_sha256": file_sha256(best),
        "checkpoint_provenance": checkpoint["provenance"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results.append(result)
    del model, checkpoint, validation
    torch.cuda.empty_cache()
winner = min(results, key=lambda item: (item["selection_value"], item["candidate_blur_ratio"]))
report = {
    "selection_rule": "minimum fixed-reference Sinkhorn on K562 perturbation-OOD validation",
    "selected_blur_ratio": winner["candidate_blur_ratio"],
    "selected_checkpoint": winner["training_report"]["best_checkpoint"],
    "selected_checkpoint_sha256": winner["best_checkpoint_sha256"],
    "candidates": results,
}
(root / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
