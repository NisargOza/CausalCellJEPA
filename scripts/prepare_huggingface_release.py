# Build a compact, provenance-verified Hugging Face model repository from frozen artifacts.
# Raw data, optimizer states, smoke checkpoints, and third-party ESM weights are excluded.

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".hf_upload/CausalCellJEPA"

COMPONENTS = {
    "stage1_teacher": {
        "source": "artifacts/stage1_thunder_f1e5dce/stage1/canonical_ema_teacher.pt",
        "sha256": "03924557312b4829716fe643fd6d6f9f0704acec470773094326ac1fb47c074f",
        "tensor_key": "teacher",
        "metadata_keys": [
            "format_version",
            "frozen",
            "cell_dim",
            "encoder_configuration",
            "training_state",
            "provenance",
        ],
        "role": "frozen EMA cell-state teacher",
        "status": "proposal-locked primary component",
    },
    "stage2_primary": {
        "source": "artifacts/dynamics_selected_0810656.pt",
        "sha256": "d0d4eed4dfbc6e63843c0fb3294c79c115832b75eaf6712617d31f6c1ce160b4",
        "tensor_key": "model",
        "metadata_keys": ["format_version", "state", "configuration", "provenance"],
        "role": "unpaired population-dynamics model",
        "status": "proposal-locked primary component",
    },
    "transcriptomic_readout": {
        "source": "artifacts/readout_linear_v1.pt",
        "sha256": "7a4fd5d723699ee254d612f3e7e073fb204fdcdac69a994392d65ca4e8af4d2b",
        "tensor_key": None,
        "metadata_keys": [
            "format_version",
            "output_clamp_min",
            "report",
            "provenance",
        ],
        "role": "latent-to-3,000-HVG linear decoder",
        "status": "proposal-locked reporting component",
    },
    "stage2_multiteacher_v4": {
        "source": "artifacts/contextual_multiteacher_dynamics/availability_static/best.pt",
        "sha256": "9b179337ebdab233203166345ed4e0381ef2e9cb74123239b7be4652df1b2761",
        "tensor_key": "model",
        "metadata_keys": ["format_version", "state", "configuration", "provenance"],
        "role": "validation-selected ESM-2 + GO multiteacher population model",
        "status": "post-primary exploratory extension",
    },
    "multiteacher_effect_anchor": {
        "source": "artifacts/contextual_multiteacher_effect_anchor_v1.pt",
        "sha256": "295ac6efd8c1bd58caeabf33604a571ed7e0391eacd0c745cd9b9ca5e690541d",
        "tensor_key": None,
        "metadata_keys": ["format_version", "architecture", "report"],
        "role": "multiteacher low-rank latent-effect anchor",
        "status": "post-primary exploratory extension",
        "retain_original": True,
    },
    "external_response_predictor": {
        "source": "artifacts/external_response_student_v1.pt",
        "sha256": "b458f31547b1fc056f1e9b00757b45f87217e955a61e2f75695748f2514c1e14",
        "tensor_key": None,
        "metadata_keys": [
            "format_version",
            "architecture",
            "feature_slice",
            "kernel",
            "output_scale",
            "report",
            "provenance",
        ],
        "role": "multi-context external-response gene-effect predictor",
        "status": "exploratory post-test component used in final external confirmation",
    },
    "string_go_predictor": {
        "source": "artifacts/string_kernel_gene_student_v1.pt",
        "sha256": "87d6bad11f947d98ee57ff30124f93de8fe983a0c0bd02ae737fbf77940b43cd",
        "tensor_key": None,
        "metadata_keys": [
            "format_version",
            "architecture",
            "feature_block",
            "feature_slice",
            "kernel",
            "report",
            "provenance",
        ],
        "role": "STRING + GO kernel gene-effect predictor",
        "status": "exploratory post-test component used in final external confirmation",
    },
    "control_ood_gate": {
        "source": "artifacts/control_ood_residual_gate_v1.pt",
        "sha256": "3c238a040ab881aedff6815bd093808d01e5bc2c731db5ebf47828763a1329f9",
        "tensor_key": None,
        "metadata_keys": ["format_version", "architecture", "threshold", "temperature"],
        "role": "control-population OOD confidence gate",
        "status": "control-only frozen component used in final external confirmation",
    },
    "replogle_actions_multiteacher": {
        "source": "artifacts/replogle_actions_esm2_go_context_v1.pt",
        "sha256": "d6987ef03b3db4bf6d26990e0ebe3a47f564e196f5d7b4c935c4680e04778983",
        "tensor_key": None,
        "metadata_keys": ["targets", "modality_dims", "modalities", "modality_availability"],
        "role": "precomputed ESM-2 + GO action features for 997 Replogle targets",
        "status": "derived feature cache; no third-party model weights included",
    },
    "replogle_actions_final": {
        "source": "artifacts/replogle_actions_esm2_go_string_v1.pt",
        "sha256": "bf30fe2bcb1cd3c79612cc6ce570f82002b96de92b8b8dd24ed3ee142b9ddee0",
        "tensor_key": None,
        "metadata_keys": ["targets", "modality_dims", "modalities", "modality_availability"],
        "role": "precomputed ESM-2 + GO + STRING features for 997 Replogle targets",
        "status": "derived feature cache; no third-party model weights included",
    },
}

RELEASE_FILES = [
    "configs/stage1.yaml",
    "configs/dynamics.yaml",
    "configs/readout.yaml",
    "configs/contextual_multiteacher_dynamics.yaml",
    "configs/adamson_external_confirmation.yaml",
    "configs/adamson_external_evaluation.yaml",
    "manifests/replogle_v1.json",
    "manifests/dynamics_selection_v1.json",
    "manifests/readout_v1.json",
    "manifests/contextual_multiteacher_dynamics_selection_v1.json",
    "manifests/contextual_multiteacher_effect_anchor_v1.json",
    "manifests/contextual_multiteacher_action_v1.json",
    "manifests/string_action_v1.json",
    "manifests/external_response_selection_v1.json",
    "manifests/string_kernel_selection_v1.json",
    "manifests/control_ood_residual_gate_v1.json",
    "manifests/evaluation_pseudo_v1.json",
    "manifests/adamson_external_prediction_v1.json",
    "manifests/adamson_external_confirmation_v1.json",
    "docs/adamson_external_confirmation_v1.md",
    "docs/publication_positioning_and_figure_design.md",
]

ASSETS = [
    "figures/empirical/empirical_1_replogle_population_umap.png",
    "figures/empirical/empirical_2_replogle_target_paired_scatter.png",
    "figures/empirical/empirical_3_adamson_systema_paired_scatter.png",
    "figures/empirical/empirical_4_adamson_gene_effect_heatmap.png",
    "figures/empirical/empirical_5_adamson_gene_effect_density.png",
    "figures/empirical/empirical_figure_manifest.json",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        raise TypeError("Tensor must be stored in safetensors, not JSON metadata")
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def load_component(path: Path, tensor_key: str | None) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    tensors = (
        payload[tensor_key]
        if tensor_key
        else {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    )
    assert tensors and all(isinstance(value, torch.Tensor) for value in tensors.values())
    return payload


def export_component(name: str, specification: dict, output: Path) -> dict:
    source = ROOT / specification["source"]
    assert source.is_file()
    observed = file_sha256(source)
    assert observed == specification["sha256"], f"hash mismatch for {source}"
    payload = load_component(source, specification["tensor_key"])
    raw_tensors = (
        payload[specification["tensor_key"]]
        if specification["tensor_key"]
        else {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    )
    tensors = {key: value.detach().cpu().contiguous() for key, value in raw_tensors.items()}
    weights_path = output / "weights" / f"{name}.safetensors"
    save_file(tensors, weights_path, metadata={"format": "pt", "component": name})
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(tensors)
        for key, tensor in tensors.items():
            restored = handle.get_tensor(key)
            assert restored.dtype == tensor.dtype and restored.shape == tensor.shape
            assert torch.equal(restored, tensor)
    metadata = {
        key: json_safe(payload[key]) for key in specification["metadata_keys"] if key in payload
    }
    metadata_path = output / "metadata" / f"{name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    original = None
    if specification.get("retain_original"):
        original_path = output / "weights" / source.name
        shutil.copy2(source, original_path)
        original = {
            "path": str(original_path.relative_to(output)),
            "bytes": original_path.stat().st_size,
            "sha256": file_sha256(original_path),
        }
    return {
        "role": specification["role"],
        "status": specification["status"],
        "source": {
            "path": specification["source"],
            "bytes": source.stat().st_size,
            "sha256": observed,
        },
        "weights": {
            "path": str(weights_path.relative_to(output)),
            "bytes": weights_path.stat().st_size,
            "sha256": file_sha256(weights_path),
            "tensors": len(tensors),
        },
        "metadata": {
            "path": str(metadata_path.relative_to(output)),
            "bytes": metadata_path.stat().st_size,
            "sha256": file_sha256(metadata_path),
        },
        **({"original_small_dependency": original} if original else {}),
    }


def copy_file(relative: str, output: Path, destination_root: str) -> dict:
    source = ROOT / relative
    assert source.is_file()
    destination = output / destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination.relative_to(output)),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def build_release(output: Path = DEFAULT_OUTPUT, replace: bool = False) -> dict:
    if output.exists():
        assert replace, f"output already exists: {output}; pass --replace"
        shutil.rmtree(output)
    (output / "weights").mkdir(parents=True)
    (output / "metadata").mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        ).strip()
    )

    components = {
        name: export_component(name, specification, output)
        for name, specification in COMPONENTS.items()
    }
    provenance = {relative: copy_file(relative, output, "provenance") for relative in RELEASE_FILES}
    assets = {relative: copy_file(relative, output, "assets") for relative in ASSETS}
    for relative in (
        "huggingface/README.md",
        "huggingface/load_components.py",
        "huggingface/requirements.txt",
        "huggingface/.gitattributes",
        "huggingface/CITATION.cff",
    ):
        source = ROOT / relative
        destination = output / source.name
        shutil.copy2(source, destination)

    code = {}
    for source in sorted((ROOT / "src/causalcelljepa").glob("*.py")):
        destination = output / "code/causalcelljepa" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        code[str(source.relative_to(ROOT))] = {
            "path": str(destination.relative_to(output)),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        }

    model_config = {
        "architectures": ["CausalCellJEPA"],
        "model_type": "causalcelljepa",
        "library_name": "pytorch",
        "cell_latent_dimension": 256,
        "gene_vocabulary_size": 3000,
        "population_size": 32,
        "components": {name: item["weights"]["path"] for name, item in components.items()},
        "github_commit": commit,
    }
    (output / "config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True) + "\n")
    bundle_files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "release": "CausalCellJEPA Hugging Face component bundle",
        "github": {
            "repository": "https://github.com/NisargOza/CausalCellJEPA",
            "commit": commit,
            "tracked_worktree_dirty_during_packaging": dirty,
        },
        "license": {
            "huggingface_tag": "other",
            "note": "No model/software license was present in the source repository; this package does not invent one.",
        },
        "exclusions": [
            "raw or processed single-cell data",
            "optimizer, scheduler, and RNG checkpoint state",
            "smoke-test and superseded checkpoints",
            "third-party ESM-2 weights",
            "State baseline weights",
        ],
        "components": components,
        "provenance_files": provenance,
        "assets": assets,
        "code": code,
        "bundle_files": bundle_files,
    }
    manifest_path = output / "MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["package"] = {
        "files": sum(path.is_file() for path in output.rglob("*")),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "manifest_sha256": file_sha256(manifest_path),
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    result = build_release(arguments.output, arguments.replace)
    print(json.dumps(result["package"], sort_keys=True))
