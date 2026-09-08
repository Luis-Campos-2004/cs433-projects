"""The six mandatory functions for CS-433 Project 1.

Signatures and return values follow the usual required format: each function
returns `(w, loss)`, where `loss` is computed with the FINAL `w`. Check the
exact names against this year's official handout before submitting — grading
is automated and does not forgive typos.

Usual Project 1 constraint: numpy only. No scikit-learn.
"""

import numpy as np

# ----------------------------------------------------------------- losses


def compute_mse(y, tx, w):
    """Mean squared error loss: 1/(2N) * sum(e^2)."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_gradient_mse(y, tx, w):
    """Gradient of the MSE."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Numerically stable sigmoid (no overflow for large |t|)."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def compute_logistic_loss(y, tx, w):
    """Mean negative log-likelihood. Expects y in {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# --------------------------------------------------- mandatory functions


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient_mse(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with batch size 1."""
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        grad = compute_gradient_mse(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Exact solution via the normal equations."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations."""
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. y in {0, 1}."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression. The returned loss does NOT include
    the penalty term — watch out, this is where everyone slips up."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
