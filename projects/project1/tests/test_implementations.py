"""Sanity tests. Run with `pytest -q` from the repo root."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from implementations import (  # noqa: E402
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(50), rng.normal(size=(50, 2))]
    w_true = np.array([1.0, 2.0, -3.0])
    return tx @ w_true, tx, w_true


def test_least_squares_recovers_w(linear_data):
    y, tx, w_true = linear_data
    w, loss = least_squares(y, tx)
    assert np.allclose(w, w_true)
    assert loss == pytest.approx(0.0, abs=1e-10)


def test_ridge_shrinks_towards_zero(linear_data):
    y, tx, _ = linear_data
    w_small, _ = ridge_regression(y, tx, 1e-8)
    w_big, _ = ridge_regression(y, tx, 10.0)
    assert np.linalg.norm(w_big) < np.linalg.norm(w_small)


def test_gd_decreases_loss(linear_data):
    y, tx, _ = linear_data
    w0 = np.zeros(tx.shape[1])
    _, loss_few = mean_squared_error_gd(y, tx, w0, 5, 0.1)
    _, loss_many = mean_squared_error_gd(y, tx, w0, 200, 0.1)
    assert loss_many < loss_few


def test_sigmoid_is_stable():
    assert np.all(np.isfinite(sigmoid(np.array([-1e3, 0.0, 1e3]))))
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_shapes_and_return_contract(linear_data):
    y, tx, _ = linear_data
    y01 = (y > 0).astype(float)
    w0 = np.zeros(tx.shape[1])
    for w, loss in [
        logistic_regression(y01, tx, w0, 10, 0.1),
        reg_logistic_regression(y01, tx, 0.1, w0, 10, 0.1),
    ]:
        assert w.shape == w0.shape
        assert np.isscalar(loss) or np.ndim(loss) == 0
