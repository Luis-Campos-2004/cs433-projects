import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from experiments import RESULTS, brfss_prep, header_and_data, run
from cross_validation import log_results
from mlp import mlp_scores, train_mlp
from training import logistic_scores, train_reg_logistic


def train_blend(y, tx):
    return train_reg_logistic(y, tx, 1e-3), train_mlp(y, tx)


def blend_scores(tx, models):
    w, nets = models
    return 0.5 * logistic_scores(tx, w) + 0.5 * mlp_scores(tx, nets)


x, y, header = header_and_data()
rows = []
run("blend", x, y, "BRFSS + one-hot", brfss_prep(header, onehot=True),
    {"logistic + MLP blend": (train_blend, blend_scores)}, rows)
log_results(RESULTS / "cv_results.csv", rows)