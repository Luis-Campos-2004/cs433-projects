"""Models compared in experiments.py, built on the six functions of
implementations.py.

Every model is a pair of functions used the same way everywhere:

    w = train_xxx(y, tx, ...)       # y in {0, 1}, tx has a first column of 1s
    scores = xxx_scores(tx, w)      # higher score = more likely to be positive

The decision threshold on the scores is chosen separately (see
src/cross_validation.py): with ~9% positives, the usual 0.5 threshold
predicts far too few positives.
"""

import numpy as np
from implementations import (
    least_squares,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)

MAX_ITERS = 1000  # gradient descent steps for the logistic models


def add_bias(z):
    """Prepend a column of 1s so that w[0] is the intercept."""
    return np.c_[np.ones(len(z)), z]


def safe_gamma(tx, n_steps=50, seed=0):
    """Step size for logistic gradient descent: half the stability limit.

    The logistic loss has curvature at most lambda_max(tx^T tx / N) / 4, so
    gradient descent is stable for gamma < 8 / lambda_max. lambda_max is
    estimated with power iteration (repeated multiplication by tx^T tx / N).
    """
    v = np.random.default_rng(seed).normal(size=tx.shape[1])
    lambda_max = 1.0
    for _ in range(n_steps):
        v = tx.T @ (tx @ v) / len(tx)
        lambda_max = np.linalg.norm(v)
        v = v / lambda_max
    return 4.0 / lambda_max


def train_least_squares(y, tx):
    """Least squares on the {0, 1} labels (normal equations)."""
    w, _ = least_squares(y, tx)
    return w


def train_ridge(y, tx, lambda_):
    """Ridge regression on the {0, 1} labels (normal equations)."""
    w, _ = ridge_regression(y, tx, lambda_)
    return w


def train_logistic(y, tx, max_iters=MAX_ITERS):
    """Logistic regression by gradient descent from w = 0."""
    w0 = np.zeros(tx.shape[1])
    w, _ = logistic_regression(y, tx, w0, max_iters, safe_gamma(tx))
    return w


def train_reg_logistic(y, tx, lambda_, max_iters=MAX_ITERS):
    """L2-regularized logistic regression by gradient descent from w = 0."""
    w0 = np.zeros(tx.shape[1])
    w, _ = reg_logistic_regression(y, tx, lambda_, w0, max_iters, safe_gamma(tx))
    return w


def linear_scores(tx, w):
    """Scores of the regression models: the predicted value tx @ w."""
    return tx @ w


def logistic_scores(tx, w):
    """Scores of the logistic models: the predicted probability of y = 1."""
    return sigmoid(tx @ w)
