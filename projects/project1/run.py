"""End-to-end pipeline: raw CSV files -> baseline model -> AIcrowd submission.

Baseline (week 1):
1. Load the raw data with the course helper `load_csv_data`.
2. Simple preprocessing, fitted on training rows only: drop empty or constant
   columns, fill missing values with the column mean, standardize.
3. Regularized logistic regression from implementations.py (labels in {0, 1}).
4. Choose the decision threshold that maximizes F1 on a held-out validation
   split. Only ~9% of the labels are positive, so 0.5 would be a poor threshold.
5. Retrain on all training rows and write submissions/submission.csv.

Usage, from projects/project1/:
    python run.py
"""

from pathlib import Path

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import sigmoid
from model import apply_preprocessing, fit_preprocessing, train

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "submissions"

SEED = 42
VAL_FRACTION = 0.2


def f1_score(y_true, y_pred):
    """F1 score for labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(y_true, probs):
    """Probability threshold in [0.05, 0.95] that maximizes F1."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(scores))
    return thresholds[i], scores[i]


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA)
    y = (y_train == 1).astype(float)  # {-1, 1} -> {0, 1}

    # 1) Hold-out validation split to choose the threshold.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_val = int(VAL_FRACTION * len(y))
    val, tr = idx[:n_val], idx[n_val:]

    params = fit_preprocessing(x_train[tr])
    w = train(y[tr], apply_preprocessing(x_train[tr], params))
    probs_val = sigmoid(apply_preprocessing(x_train[val], params) @ w)
    threshold, f1 = best_threshold(y[val], probs_val)
    acc = np.mean((probs_val >= threshold) == y[val])
    print(
        f"validation: F1 = {f1:.4f}, accuracy = {acc:.4f}, threshold = {threshold:.2f}"
    )

    # 2) Retrain on all training rows and predict the test set.
    params = fit_preprocessing(x_train)
    w = train(y, apply_preprocessing(x_train, params))
    probs_test = sigmoid(apply_preprocessing(x_test, params) @ w)
    preds = np.where(probs_test >= threshold, 1, -1)

    OUT.mkdir(exist_ok=True)
    create_csv_submission(test_ids, preds, OUT / "submission.csv")
    print(f"written: {OUT / 'submission.csv'} ({np.mean(preds == 1):.1%} positive)")


if __name__ == "__main__":
    main()
