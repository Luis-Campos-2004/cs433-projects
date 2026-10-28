"""One-hidden-layer MLP in numpy (ReLU, weighted cross-entropy, Adam).

Same interface as the other models: train_mlp(y, tx) -> params, mlp_scores(tx, params).
`tx` has a leading bias column, which is dropped here (the network has its own biases).
Early stopping uses the best F1 on a 10% holdout of the training rows.
"""

import numpy as np
from cross_validation import best_threshold
from implementations import sigmoid


def _forward(x, p):
    h = np.maximum(0.0, x @ p["W1"] + p["b1"])
    return h, h @ p["w2"] + p["b2"]


def _train_one(y, x, hidden, epochs, lr, wd, pos_weight, batch, patience, seed):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    perm = rng.permutation(n)
    n_hold = n // 10
    hold, fit = perm[:n_hold], perm[n_hold:]

    p = {
        "W1": (rng.normal(0, np.sqrt(2 / d), (d, hidden))).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": (rng.normal(0, np.sqrt(1 / hidden), hidden)).astype(np.float32),
        "b2": np.float32(0.0),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    best, best_f1, bad, t = {k: np.copy(v) for k, v in p.items()}, -1.0, 0, 0

    for epoch in range(epochs):
        order = rng.permutation(fit)
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            xb, yb = x[idx], y[idx]
            h, z = _forward(xb, p)
            prob = 1.0 / (1.0 + np.exp(-z))
            wgt = np.where(yb == 1, pos_weight, 1.0).astype(np.float32)
            dz = ((prob - yb) * wgt / len(idx)).astype(np.float32)
            dh = np.outer(dz, p["w2"]) * (h > 0)
            g = {
                "W1": xb.T @ dh + wd * p["W1"],
                "b1": dh.sum(0),
                "w2": h.T @ dz + wd * p["w2"],
                "b2": np.float32(dz.sum()),
            }
            t += 1
            for k in p:  # Adam
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                p[k] = p[k] - lr * (m[k] / (1 - 0.9**t)) / (
                    np.sqrt(v2[k] / (1 - 0.999**t)) + 1e-8
                )
        _, f1 = best_threshold(y[hold], _forward(x[hold], p)[1])
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best = {k: np.copy(v) for k, v in p.items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best


def train_mlp(y, tx, n_models=1, hidden=64, epochs=30, lr=1e-3, wd=1e-4,
              pos_weight=3.0, batch=512, patience=4):
    """Ensemble of n_models networks with different seeds."""
    x = tx[:, 1:].astype(np.float32)
    return [
        _train_one(y, x, hidden, epochs, lr, wd, pos_weight, batch, patience, seed)
        for seed in range(n_models)
    ]


def mlp_scores(tx, models):
    """Average of the networks' predicted probabilities."""
    x = tx[:, 1:].astype(np.float32)
    return np.mean([sigmoid(_forward(x, p)[1]) for p in models], axis=0)