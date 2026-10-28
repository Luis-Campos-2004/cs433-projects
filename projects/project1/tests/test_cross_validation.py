"""Tests for src/cross_validation.py. Run with `pytest -q` from the repo root."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_validation import best_threshold, f1_score, stratified_kfold


def test_folds_partition_rows_and_keep_class_balance():
    y = (np.random.default_rng(0).random(1000) < 0.1).astype(float)
    splits = stratified_kfold(y, k=5, seed=1)
    all_val = np.concatenate([val for _, val in splits])
    assert np.array_equal(np.sort(all_val), np.arange(len(y)))
    for train, val in splits:
        assert len(np.intersect1d(train, val)) == 0
        assert len(train) + len(val) == len(y)
        assert abs(y[val].mean() - y.mean()) < 0.01


def test_f1_score_small_example():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])  # TP = 2, FP = 1, FN = 1
    assert f1_score(y_true, y_pred) == 2 * 2 / (2 * 2 + 1 + 1)


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.2).astype(float)
    scores = np.round(y + rng.normal(size=500), 1)  # rounded: many ties
    threshold, f1 = best_threshold(y, scores)
    brute = max(f1_score(y, (scores >= t).astype(int)) for t in np.unique(scores))
    assert np.isclose(f1, brute)
    assert np.isclose(f1, f1_score(y, (scores >= threshold).astype(int)))
