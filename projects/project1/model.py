
import sys
from pathlib import Path

import numpy as np

# implementations.py lives in the src folder next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from implementations import reg_logistic_regression  # noqa: E402

# Hyperparameters (GAMMA must stay below ~0.39)
LAMBDA = 1e-4
GAMMA = 0.2
MAX_ITERS = 500


def fit_preprocessing(x):
    """Learn which columns to keep and their mean/std, from x only."""
    not_all_nan = ~np.all(np.isnan(x), axis=0)       # drop columns that are entirely NaN
    std = np.nanstd(x, axis=0)
    not_constant = np.nan_to_num(std) > 0            # drop columns with one single value
    keep = not_all_nan & not_constant
    mean = np.nanmean(x[:, keep], axis=0)            # mean of each kept column, ignoring NaN
    std = np.nanstd(x[:, keep], axis=0)              # std of each kept column, ignoring NaN
    return keep, mean, std


def apply_preprocessing(x, params):
    """Apply learned params to any x: keep columns, fill NaN, standardize, add bias."""
    keep, mean, std = params
    x = x[:, keep].copy()                            # only the kept columns
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = mean[nan_cols]           # replace NaN with the column mean
    x = (x - mean) / std                             # standardize
    return np.c_[np.ones(len(x)), x]                 # first column of 1s (bias)


def train(y, tx):
    """Train regularized logistic regression from w = 0, return w only."""
    initial_w = np.zeros(tx.shape[1])
    w, _ = reg_logistic_regression(y, tx, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    return w