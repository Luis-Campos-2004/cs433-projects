"""End-to-end pipeline: data -> model -> submission file.

Golden rule: if this doesn't run on a clean clone of the repo, the submission
isn't reproducible. Notebooks are for exploring; run.py is the source of truth.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
OUT = ROOT / "submissions"

SEED = 42


def load_data():
    """TODO: load train/test from DATA."""
    raise NotImplementedError


def preprocess(x_train, x_test):
    """TODO: handle NaNs, standardize (using TRAIN statistics), expand features."""
    raise NotImplementedError


def train(y, tx):
    """TODO: pick model + hyperparameters via cross-validation."""
    raise NotImplementedError


def main():
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)

    y_train, x_train, x_test, test_ids = load_data()
    tx_train, tx_test = preprocess(x_train, x_test)
    w = train(y_train, tx_train)

    preds = np.where(tx_test @ w > 0, 1, -1)
    np.savetxt(
        OUT / "submission.csv",
        np.c_[test_ids, preds],
        fmt="%d",
        delimiter=",",
        header="Id,Prediction",
        comments="",
    )
    print(f"written: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
