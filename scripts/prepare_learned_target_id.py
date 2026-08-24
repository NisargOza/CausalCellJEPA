# Build a categorical action cache using only dynamics-training target identities.
import json
from hashlib import sha256
from pathlib import Path

import torch
import yaml

from causalcelljepa.actions import learned_target_id_payload
from causalcelljepa.resources import file_sha256
from causalcelljepa.training import _git_state, _runtime_environment, _runtime_source_hash

config_path = Path("configs/learned_target_id.yaml")
config = yaml.safe_load(config_path.read_text())
assert file_sha256(config["base_config_path"]) == config["base_config_sha256"]
assert file_sha256(config["source_action_cache_path"]) == config[
    "source_action_cache_sha256"
]
specification = json.loads(Path(config["specification_manifest_path"]).read_text())
declared = specification.pop("manifest_sha256")
computed = sha256(
    json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert declared == config["specification_manifest_sha256"] == computed
dynamics = yaml.safe_load(Path(config["base_config_path"]).read_text())
dynamics_manifest = json.loads(Path(dynamics["inputs"]["dynamics_manifest_path"]).read_text())
assert dynamics_manifest["manifest_sha256"] == specification["source"][
    "dynamics_manifest_sha256"
]
train_targets = sorted(dynamics_manifest["conditions"]["train"]["cells_per_target"])
validation_targets = sorted(
    dynamics_manifest["conditions"]["validation"]["cells_per_target"]
)
source = torch.load(config["source_action_cache_path"], map_location="cpu", weights_only=True)
targets = list(source["targets"])


def target_hash(values):
    return sha256("\n".join(values).encode()).hexdigest()


policy = specification["policy"]
assert len(train_targets) == policy["known_target_count"]
assert target_hash(train_targets) == policy["known_targets_sha256"]
assert len(validation_targets) == policy["validation_target_count"]
assert target_hash(validation_targets) == policy["validation_targets_sha256"]
assert not set(train_targets) & set(validation_targets)
assert len(targets) == specification["source"]["target_universe_count"]
assert target_hash(targets) == specification["source"]["target_universe_sha256"]
assert set(train_targets) | set(validation_targets) <= set(targets)
known, embedding = learned_target_id_payload(targets, train_targets)
assert embedding.shape[1] == policy["vocabulary_dimension"]
assert int(known.sum()) == len(train_targets)
assert torch.equal(embedding[known].sum(1), torch.ones(len(train_targets)))
assert torch.count_nonzero(embedding[~known]) == 0

output = Path(config["action_cache_path"])
assert not output.exists()
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
torch.save(
    {
        "format_version": 1,
        "targets": targets,
        "known": known,
        "embedding": embedding,
        "input_dim": policy["vocabulary_dimension"],
        "projection_dim": specification["protocol"]["model_action_dimension"],
        "mechanism": "learned_target_id",
        "unknown_policy": policy["unknown_action_encoding"],
        "provenance": {
            "specification_manifest_sha256": declared,
            "config_sha256": file_sha256(config_path),
            "source_action_cache_sha256": config["source_action_cache_sha256"],
            "runtime_source_sha256": _runtime_source_hash(),
            "runtime_environment": _runtime_environment(),
            "git": _git_state(),
        },
    },
    temporary,
)
temporary.replace(output)
print(
    json.dumps(
        {
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "targets": len(targets),
            "known_training_targets": int(known.sum()),
            "unknown_targets": int((~known).sum()),
            "input_dim": embedding.shape[1],
        },
        indent=2,
    )
)
