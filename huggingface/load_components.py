"""Verified loaders for the CausalCellJEPA Hugging Face component bundle."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release(root) -> tuple[Path, dict]:
    root = Path(root)
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text())
    return root, manifest


def verify_component(root, name: str) -> dict:
    root, release = _release(root)
    component = release["components"][name]
    for section in ("weights", "metadata"):
        record = component[section]
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"integrity check failed for {path}")
    return component


def load_tensor_component(root, name: str) -> tuple[dict, dict]:
    """Load a flat tensor payload plus its JSON-safe non-tensor metadata."""
    root = Path(root)
    component = verify_component(root, name)
    tensors = load_file(root / component["weights"]["path"], device="cpu")
    metadata = json.loads((root / component["metadata"]["path"]).read_text())
    return tensors, metadata


def load_primary_dynamics(root):
    """Instantiate and load the proposal-locked primary population-dynamics model."""
    from causalcelljepa.dynamics import build_dynamics_model

    state, metadata = load_tensor_component(root, "stage2_primary")
    model = build_dynamics_model(metadata["configuration"])
    model.load_state_dict(state, strict=True)
    return model, metadata


def load_multiteacher_dynamics(root):
    """Instantiate and load the validation-selected exploratory multiteacher model."""
    from causalcelljepa.dynamics import build_dynamics_model

    root = Path(root).resolve()
    state, metadata = load_tensor_component(root, "stage2_multiteacher_v4")
    config = copy.deepcopy(metadata["configuration"])
    anchor = config["effect_anchor"]
    anchor["output_path"] = str(root / "weights/contextual_multiteacher_effect_anchor_v1.pt")
    anchor["manifest_path"] = str(
        root / "provenance/manifests/contextual_multiteacher_effect_anchor_v1.json"
    )
    model = build_dynamics_model(config)
    model.load_state_dict(state, strict=True)
    return model, metadata


def load_stage1_teacher_state(root) -> tuple[dict, dict]:
    """Return the frozen EMA teacher state dict and exact encoder metadata."""
    return load_tensor_component(root, "stage1_teacher")
