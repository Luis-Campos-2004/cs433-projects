"""Evaluation for Project 1: stratified k-fold cross-validation, F1 and the
choice of the decision threshold.

Protocol, for each of the k folds:
    1. fit the preprocessing on the training rows only;
    2. train the model on the training rows;
    3. choose the threshold that maximizes F1 on the training rows;
    4. measure F1 and accuracy on the validation rows with that threshold.
Nothing is learned on the validation rows, so the scores are honest estimates
of the performance on new data. Labels are in {0, 1}.
"""

import csv
import time
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD = 0.5  # the "naive" threshold, kept for the ablation


def stratified_kfold(y, k=5, seed=42):
    """Split the row indices into k folds with the same share of positives.

    Returns a list of k (train_idx, val_idx) pairs.
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        rows = rng.permutation(np.where(y == label)[0])
        for f, part in enumerate(np.array_split(rows, k)):
            folds[f].append(part)
    folds = [np.concatenate(parts) for parts in folds]
    splits = []
    for f in range(k):
        train = np.concatenate([folds[g] for g in range(k) if g != f])
        splits.append((np.sort(train), np.sort(folds[f])))
    return splits


def f1_score(y_true, y_pred):
    """F1 = 2 TP / (2 TP + FP + FN) for labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def accuracy(y_true, y_pred):
    """Share of correct predictions."""
    return float(np.mean(y_true == y_pred))


def best_threshold(y_true, scores):
    """Threshold t maximizing F1 when `scores >= t` is predicted positive.

    Sort the rows by decreasing score. Predicting the top k rows as positive
    gives TP_k = number of positives among them and FP + FN = k - TP_k + P -
    TP_k (P = number of positives), so F1 = 2 TP_k / (k + P). One cumulative
    sum gives F1 for every k at once; cuts are only allowed between different
    scores. Works for probabilities and for unbounded linear scores.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tp = np.cumsum(y_true[order])
    k = np.arange(1, len(scores) + 1)
    f1 = 2 * tp / (k + np.sum(y_true))
    can_cut = np.r_[sorted_scores[:-1] > sorted_scores[1:], True]
    f1 = np.where(can_cut, f1, -1.0)
    i = int(np.argmax(f1))
    return float(sorted_scores[i]), float(f1[i])


def cross_validate(x, y, prep_fit, prep_apply, models, k=5, seed=42):
    """Evaluate several models with k-fold CV, sharing the preprocessing.

    prep_fit(x, y) -> (params, tx)  learns on the training rows, returns them
                                    preprocessed (with the bias column)
    prep_apply(x, params) -> tx     applies the learned params to other rows
    models: {name: (train_fn, score_fn)} with train_fn(y, tx) -> w and
            score_fn(tx, w) -> scores

    Returns {name: summary dict} (see `summarize`).
    """
    folds = {name: [] for name in models}
    for f, (train, val) in enumerate(stratified_kfold(y, k, seed)):
        params, tx_train = prep_fit(x[train], y[train])
        tx_val = prep_apply(x[val], params)
        for name, (train_fn, score_fn) in models.items():
            start = time.time()
            w = train_fn(y[train], tx_train)
            seconds = time.time() - start
            threshold, _ = best_threshold(y[train], score_fn(tx_train, w))
            val_scores = score_fn(tx_val, w)
            pred = (val_scores >= threshold).astype(int)
            pred_default = (val_scores >= DEFAULT_THRESHOLD).astype(int)
            folds[name].append(
                {
                    "f1": f1_score(y[val], pred),
                    "accuracy": accuracy(y[val], pred),
                    "f1_default": f1_score(y[val], pred_default),
                    "threshold": threshold,
                    "seconds": seconds,
                    "n_features": tx_train.shape[1] - 1,
                }
            )
            print(f"  fold {f + 1}/{k}  {name:32s} F1 = {folds[name][-1]['f1']:.4f}")
    return {name: summarize(results) for name, results in folds.items()}


def summarize(fold_results):
    """Mean and standard deviation over the folds."""
    out = {}
    for key in ("f1", "accuracy", "f1_default", "threshold", "seconds"):
        values = np.array([r[key] for r in fold_results])
        out[key + "_mean"] = float(values.mean())
        out[key + "_std"] = float(values.std())
    out["n_features"] = fold_results[0]["n_features"]
    return out


def log_results(path, rows):
    """Append rows (dicts with the same keys) to a CSV experiment log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
