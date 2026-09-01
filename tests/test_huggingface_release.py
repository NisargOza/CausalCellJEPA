# Validate the Hugging Face release inventory and its exclusion boundaries.

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_huggingface_release.py"
SPEC = importlib.util.spec_from_file_location("prepare_huggingface_release", SCRIPT)
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


def test_release_components_exist_and_match_frozen_hashes():
    assert "stage1_teacher" in RELEASE.COMPONENTS
    assert "stage2_primary" in RELEASE.COMPONENTS
    assert "control_ood_gate" in RELEASE.COMPONENTS
    for specification in RELEASE.COMPONENTS.values():
        source = RELEASE.ROOT / specification["source"]
        assert source.is_file()
        assert RELEASE.file_sha256(source) == specification["sha256"]


def test_release_inventory_excludes_data_and_third_party_weights():
    paths = [item["source"] for item in RELEASE.COMPONENTS.values()] + RELEASE.RELEASE_FILES
    assert not any(path.startswith("data/") for path in paths)
    assert not any("esm2_t6_8M" in path for path in paths)
    assert not any("state_baseline" in path for path in paths)


def test_json_safe_normalizes_numpy_and_paths():
    value = {"array": np.array([1, 2]), "scalar": np.float32(1.5), "path": Path("x")}
    assert RELEASE.json_safe(value) == {"array": [1, 2], "scalar": 1.5, "path": "x"}
