"""Reproduces our final AIcrowd submission: raw CSV files -> submission CSV.

Usage, from projects/project1/:
    python run.py        # about 4 minutes, writes submissions/submission.csv

Every choice below was selected by 5-fold cross-validation in experiments.py
(see results/cv_results.md):
1. Load the raw data with the course helper `load_csv_data`.
2. BRFSS-aware preprocessing (src/preprocessing.py), fitted on the training rows.
3. One-hot encoding of the categorical columns (src/features.py).
4. L2-regularized logistic regression (implementations.py), gradient descent
   from w = 0 with the step size of src/training.py.
5. Decision threshold: the one that maximizes F1 on the training rows. Only
   ~9% of the labels are positive, so 0.5 would predict far too few positives.
6. Predict the test rows and write the submission with labels in {-1, 1}.

The pipeline is deterministic: it always writes exactly the same file.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cross_validation import best_threshold
from features import apply_feature_expansion, fit_feature_expansion
from helpers import create_csv_submission, load_csv_data
from preprocessing import fit_preprocessor, load_feature_names, transform
from training import add_bias, logistic_scores, train_reg_logistic

DATA = ROOT / "data"
OUT = ROOT / "submissions"

LAMBDA = 1e-3  # best of [1e-6, 1e-5, 1e-4, 1e-3] in cross-validation
ONEHOT = True  # +0.003 F1, better in 5/5 folds
SQUARES = False  # no gain in cross-validation


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA)
    y = (y_train == 1).astype(float)  # {-1, 1} -> {0, 1}
    header = load_feature_names(DATA / "x_train.csv")

    # Learn every step on the training rows.
    params, z = fit_preprocessor(x_train, header, dtype=np.float64)
    kinds = [c["kind"] for c in params["columns"]]
    expansion = fit_feature_expansion(z, kinds, onehot=ONEHOT, squares=SQUARES)
    tx = add_bias(apply_feature_expansion(z, expansion))
    w = train_reg_logistic(y, tx, LAMBDA)
    threshold, f1_train = best_threshold(y, logistic_scores(tx, w))
    print(
        f"features: {tx.shape[1] - 1}, threshold: {threshold:.4f}, "
        f"training F1: {f1_train:.4f}"
    )

    # Apply the same steps, unchanged, to the test rows.
    z_test = transform(x_test, header, params, dtype=np.float64)
    tx_test = add_bias(apply_feature_expansion(z_test, expansion))
    preds = np.where(logistic_scores(tx_test, w) >= threshold, 1, -1)

    OUT.mkdir(exist_ok=True)
    create_csv_submission(test_ids, preds, OUT / "submission.csv")
    print(f"written: {OUT / 'submission.csv'} ({np.mean(preds == 1):.1%} positive)")


if __name__ == "__main__":
    main()
