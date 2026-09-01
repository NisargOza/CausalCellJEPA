# Test condition-level aggregation, deterministic uncertainty, and artifact integrity.
# Plot rendering itself is covered by the real-data publication integration run.

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/generate_publication_figures.py"
SPEC = importlib.util.spec_from_file_location("publication_figures", SCRIPT)
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def test_condition_values_average_repeats_by_target():
    rows = [
        {"model": "candidate", "regime": "test", "target": "A", "score": 1.0},
        {"model": "candidate", "regime": "test", "target": "A", "score": 3.0},
        {"model": "candidate", "regime": "test", "target": "B", "score": -1.0},
        {"model": "other", "regime": "test", "target": "A", "score": 99.0},
    ]
    assert FIGURES.condition_values(rows, "candidate", "test", "score") == {"A": 2.0, "B": -1.0}


def test_bootstrap_and_paired_difference_are_deterministic():
    first = FIGURES.bootstrap_mean(np.array([1.0, 2.0, 3.0]), 1000, 17)
    second = FIGURES.bootstrap_mean(np.array([1.0, 2.0, 3.0]), 1000, 17)
    assert first == second
    assert first[0] == 2.0
    assert FIGURES.paired_difference({"A": 3.0, "B": 5.0}, {"A": 1.0, "B": 2.0}, 1000, 4)[0] == 2.5


def test_load_sources_requires_manifest_hash(tmp_path, monkeypatch):
    artifact = tmp_path / "metrics.jsonl"
    artifact.write_text(json.dumps({"target": "A"}) + "\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"condition_metrics": {"sha256": digest}}}))
    monkeypatch.setattr(FIGURES, "ROOT", tmp_path)
    config = {
        "sources": {
            "metrics": {
                "path": "metrics.jsonl",
                "manifest": "manifest.json",
                "sha256_keys": ["artifacts", "condition_metrics", "sha256"],
            }
        }
    }
    loaded, hashes = FIGURES.load_sources(config)
    assert loaded == {"metrics": [{"target": "A"}]}
    assert hashes == {"metrics.jsonl": digest}
