"""Submission for the logistic + MLP blend (one-hot features)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cross_validation import best_threshold
from features import apply_feature_expansion, fit_feature_expansion
from helpers import create_csv_submission, load_csv_data
from mlp import mlp_scores, train_mlp
from preprocessing import fit_preprocessor, load_feature_names, transform
from training import add_bias, logistic_scores, train_reg_logistic

DATA = ROOT / "data"
OUT = ROOT / "submissions"
LAMBDA = 1e-3


def blend(tx, w, nets):
    # same 50/50 average of probabilities as in run_blend.py (CV)
    return 0.5 * logistic_scores(tx, w) + 0.5 * mlp_scores(tx, nets)


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA)
    y = (y_train == 1).astype(float)
    header = load_feature_names(DATA / "x_train.csv")

    # Fit everything on all training rows
    params, z = fit_preprocessor(x_train, header, dtype=np.float64)
    kinds = [c["kind"] for c in params["columns"]]
    expansion = fit_feature_expansion(z, kinds, onehot=True, squares=False)
    tx = add_bias(apply_feature_expansion(z, expansion))

    w = train_reg_logistic(y, tx, LAMBDA)
    nets = train_mlp(y, tx)
    threshold, f1_train = best_threshold(y, blend(tx, w, nets))
    print(f"features: {tx.shape[1] - 1}, threshold: {threshold:.4f}, "
          f"training F1: {f1_train:.4f}")

    # Same steps, unchanged, on the test rows
    z_test = transform(x_test, header, params, dtype=np.float64)
    tx_test = add_bias(apply_feature_expansion(z_test, expansion))
    preds = np.where(blend(tx_test, w, nets) >= threshold, 1, -1)

    OUT.mkdir(exist_ok=True)
    create_csv_submission(test_ids, preds, OUT / "submission_blend.csv")
    print(f"written: {OUT / 'submission_blend.csv'} ({np.mean(preds == 1):.1%} positive)")


if __name__ == "__main__":
    main()