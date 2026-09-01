# Test outcome-blind sampling, target-level aggregation, and oriented bootstrap summaries.

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/generate_empirical_figures.py"
SPEC = importlib.util.spec_from_file_location("empirical_figures", SCRIPT)
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def test_blind_target_subset_is_deterministic_and_value_free():
    targets = ["D", "B", "A", "C", "E"]
    first = FIGURES.blind_target_subset(targets, 3, 17, "double_ood")
    second = FIGURES.blind_target_subset(list(reversed(targets)), 3, 17, "double_ood")
    assert first == second
    assert first == sorted(first)
    assert len(first) == 3


def test_condition_values_average_repeats_before_target_comparison():
    rows = [
        {"model": "candidate", "regime": "test", "target": "A", "score": 1.0},
        {"model": "candidate", "regime": "test", "target": "A", "score": 3.0},
        {"model": "candidate", "regime": "test", "target": "B", "score": -1.0},
        {"model": "reference", "regime": "test", "target": "A", "score": 99.0},
    ]
    assert FIGURES.condition_values(rows, "candidate", "test", "score") == {
        "A": 2.0,
        "B": -1.0,
    }


def test_bootstrap_advantage_is_deterministic_and_oriented():
    candidate = np.array([1.0, 2.0, 3.0])
    reference = np.array([0.0, 1.0, 2.0])
    higher = FIGURES.bootstrap_advantage(candidate, reference, True, 9, 1000)
    lower = FIGURES.bootstrap_advantage(reference, candidate, False, 9, 1000)
    assert higher == lower
    assert higher == (1.0, 1.0, 1.0)
