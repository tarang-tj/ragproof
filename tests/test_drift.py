"""Unit tests for ragproof.drift: centroid distance, PSI, and detect_drift."""

from __future__ import annotations

import numpy as np
import pytest

from ragproof.datasets import load_demo_dataset
from ragproof.drift import (
    centroid_cosine_distance,
    detect_drift,
    embed_queries,
    population_stability_index,
)
from ragproof.retrievers import DenseRetriever


def test_centroid_cosine_distance_identical_windows_is_zero():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert centroid_cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)


def test_centroid_cosine_distance_orthogonal_vectors_is_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert centroid_cosine_distance(a, b) == pytest.approx(1.0)


def test_centroid_cosine_distance_opposite_vectors_is_two():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    assert centroid_cosine_distance(a, b) == pytest.approx(2.0)


def test_centroid_cosine_distance_rejects_empty_input():
    with pytest.raises(ValueError):
        centroid_cosine_distance(np.array([]), np.array([[1.0]]))


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=500)
    psi = population_stability_index(sample, sample.copy())
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    expected = rng.normal(loc=0.0, size=1000)
    actual = rng.normal(loc=5.0, size=1000)  # large mean shift
    psi = population_stability_index(expected, actual)
    assert psi > 0.25  # standard "major shift" threshold


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        population_stability_index(np.array([]), np.array([1.0]))


def test_detect_drift_flags_large_shift_and_not_small_shift():
    corpus, _ = load_demo_dataset()
    dense = DenseRetriever(corpus, n_components=16)

    window_a = embed_queries(dense, ["python try except error handling", "list comprehension syntax"])
    window_b_same_topic = embed_queries(dense, ["python for loop over a list", "dictionary key lookup"])
    window_b_diff_topic = embed_queries(
        dense, ["rocket booster reusable landing", "mars rover solar panel power"]
    )

    result_same = detect_drift(window_a, window_b_same_topic)
    result_diff = detect_drift(window_a, window_b_diff_topic)

    assert set(result_same.keys()) == {
        "centroid_cosine_distance",
        "psi",
        "psi_threshold",
        "drift_detected",
    }
    # A genuine topic switch (python -> rockets) should shift the centroid
    # further than staying within the same topic (python -> python).
    assert result_diff["centroid_cosine_distance"] > result_same["centroid_cosine_distance"]
