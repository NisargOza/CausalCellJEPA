# Validate transferred v4 runs and freeze the K562-validation-only selection.
# Reuse the v3 freezer so checkpoint and leakage checks remain identical.
from pathlib import Path

from freeze_multiteacher_training import main

REMOTE_SHA256 = {
    "availability_static": {
        "best.pt": "9b179337ebdab233203166345ed4e0381ef2e9cb74123239b7be4652df1b2761",
        "latest.pt": "abae27373c4cf8c79f231e6c2e6f50764cc04ce5ec4d6b1fd5c20e0c6393fea2",
        "training.jsonl": "40a1c347a0bc85cc6f4bc4779af36759e6bb3113371aa7c657e031edd263cb3c",
    },
    "context_query": {
        "best.pt": "bcb0af3af4ab9ce571ba0bd8303e4362d4f82175b39dc92edf501d4bdbe72741",
        "latest.pt": "697b19ca476c561cfa72b80d0f1ebaf5e42e23969b4b2d2c9cdb225ae1ec9f89",
        "training.jsonl": "cc14f95ec9e85c2bc61ebcd943ba42f5f3613c99581acfa464ac8a0b83e4fe79",
    },
}

if __name__ == "__main__":
    main(
        config_path=Path("configs/contextual_multiteacher_dynamics.yaml"),
        artifact_root=Path("artifacts/contextual_multiteacher_dynamics"),
        console_path=Path("artifacts/contextual_multiteacher_training_console.log"),
        training_manifest_path=Path("manifests/contextual_multiteacher_dynamics_training_v1.json"),
        selection_manifest_path=Path("manifests/contextual_multiteacher_dynamics_selection_v1.json"),
        remote_sha256=REMOTE_SHA256,
    )
