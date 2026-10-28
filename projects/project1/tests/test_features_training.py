"""Tests for src/features.py and src/training.py. Run with `pytest -q`."""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from features import apply_feature_expansion, fit_feature_expansion
from training import add_bias, safe_gamma, train_reg_logistic


def small_data():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 3, size=200).astype(float)  # 3 levels
    cont = rng.normal(size=200)
    z = np.c_[(cat - cat.mean()) / cat.std(), cont]
    return z, ["cat", "cont"]


def test_feature_expansion_shapes_and_no_nan():
    z, kinds = small_data()
    params = fit_feature_expansion(z, kinds, onehot=True, squares=True)
    out = apply_feature_expansion(z, params)
    # 2 original columns + 2 indicators (3 levels - reference) + 1 square
    assert out.shape == (200, 5)
    assert not np.isnan(out).any()
    assert np.allclose(out[:, 2:].mean(axis=0), 0, atol=1e-8)


def test_unseen_level_gives_all_zero_indicators():
    z, kinds = small_data()
    params = fit_feature_expansion(z, kinds, onehot=True, squares=False)
    new = np.array([[123.0, 0.0]])  # level never seen in training
    raw_indicators = apply_feature_expansion(new, params)[:, 2:]
    expected = (0 - params["mean"]) / params["std"]
    assert np.allclose(raw_indicators, expected)


def test_safe_gamma_on_orthonormal_columns():
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(400, 5)))
    tx = q * np.sqrt(400)  # tx^T tx / N = identity, so lambda_max = 1
    assert np.isclose(safe_gamma(tx), 4.0, rtol=1e-3)


def test_reg_logistic_learns_the_sign_of_the_effect():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 1))
    y = (x[:, 0] + 0.3 * rng.normal(size=500) > 0).astype(float)
    w = train_reg_logistic(y, add_bias(x), lambda_=1e-4, max_iters=200)
    assert w.shape == (2,)
    assert w[1] > 0
