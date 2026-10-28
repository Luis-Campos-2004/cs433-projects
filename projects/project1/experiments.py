"""Cross-validation experiments behind the report: model comparison + ablation.

Usage, from projects/project1/:
    python experiments.py

Takes about 30 minutes. Writes:
    results/cv_results.csv          one row per experiment (the experiment log)
    results/cv_results.md           the same results as a readable table
    figures/models/convergence.pdf  gradient descent loss for several step sizes

Every number in the report comes from these files. All experiments use the
same 5 stratified folds (seed 42), so their scores can be compared directly.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
from cross_validation import cross_validate, log_results, stratified_kfold
from features import apply_feature_expansion, fit_feature_expansion
from helpers import load_csv_data
from implementations import compute_logistic_gradient, compute_logistic_loss
from model import apply_preprocessing, fit_preprocessing
from preprocessing import (
    fit_preprocessor,
    load_feature_names,
    select_by_target_corr,
    transform,
    undersample_majority,
)
from training import (
    add_bias,
    linear_scores,
    logistic_scores,
    safe_gamma,
    train_least_squares,
    train_logistic,
    train_reg_logistic,
    train_ridge,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "models"

K = 5
SEED = 42
RIDGE_LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]
LOGISTIC_LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3]
TARGET_CORR_MIN = 0.01  # feature selection threshold (as in the EDA notebook)


# Preprocessing variants: prep_fit(x, y) -> (params, tx), prep_apply(x, params)


def simple_prep():
    """Week-1 baseline: mean imputation + standardization (model.py)."""

    def fit(x, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            params = fit_preprocessing(x)
        return params, apply_preprocessing(x, params)

    return fit, apply_preprocessing


def brfss_prep(header, onehot=False, squares=False, target_corr=False):
    """src/preprocessing.py, optionally followed by feature expansion or by
    the selection of the columns most correlated with y (fold-local)."""

    def fit(x, y):
        params, z = fit_preprocessor(x, header, dtype=np.float64)
        expansion, selected = None, None
        if onehot or squares:
            kinds = [c["kind"] for c in params["columns"]]
            expansion = fit_feature_expansion(z, kinds, onehot, squares)
            z = apply_feature_expansion(z, expansion)
        if target_corr:
            selected = select_by_target_corr(z, y, TARGET_CORR_MIN)
            z = z[:, selected]
        return (params, expansion, selected), add_bias(z)

    def apply(x, all_params):
        params, expansion, selected = all_params
        z = transform(x, header, params, dtype=np.float64)
        if expansion is not None:
            z = apply_feature_expansion(z, expansion)
        if selected is not None:
            z = z[:, selected]
        return add_bias(z)

    return fit, apply


def undersampled(train_fn, seed=SEED):
    """Train on all positives + as many random negatives (class balancing)."""

    def fit(y, tx):
        rows = undersample_majority(y, 1.0, np.random.default_rng(seed))
        return train_fn(y[rows], tx[rows])

    return fit


# Experiments


def run(stage, x, y, prep_name, prep, models, rows):
    """Cross-validate `models` with one preprocessing and log the results."""
    print(f"\n[{stage}] preprocessing: {prep_name}")
    summaries = cross_validate(x, y, *prep, models, k=K, seed=SEED)
    for name, s in summaries.items():
        rows.append(
            {
                "stage": stage,
                "preprocessing": prep_name,
                "model": name,
                "k": K,
                "seed": SEED,
                "n_features": s["n_features"],
                "f1_mean": round(s["f1_mean"], 4),
                "f1_std": round(s["f1_std"], 4),
                "accuracy_mean": round(s["accuracy_mean"], 4),
                "f1_default_threshold": round(s["f1_default_mean"], 4),
                "threshold_mean": round(s["threshold_mean"], 4),
                "train_seconds": round(s["seconds_mean"], 1),
            }
        )
    return summaries


def convergence_figure(x, y, header):
    """Logistic regression loss along gradient descent for several step sizes,
    on the training rows of the first fold."""
    train, _ = stratified_kfold(y, K, SEED)[0]
    _, z = fit_preprocessor(x[train], header, dtype=np.float64)
    tx, yt = add_bias(z), y[train]
    gamma_safe = safe_gamma(tx)
    plt.figure(figsize=(4.2, 2.8))
    for factor in (0.25, 1, 2, 4):
        w, losses = np.zeros(tx.shape[1]), []
        for _ in range(300):
            losses.append(compute_logistic_loss(yt, tx, w))
            w = w - factor * gamma_safe * compute_logistic_gradient(yt, tx, w)
        label = rf"$\gamma$ = {factor:g} $\gamma^*$ = {factor * gamma_safe:.2f}"
        plt.plot(losses, label=label)
    plt.ylim(0.22, 0.30)
    plt.xlabel("gradient descent iteration")
    plt.ylabel("training logistic loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / "convergence.pdf")
    plt.savefig(FIGURES / "convergence.png", dpi=200)
    print(f"\nwritten: {FIGURES / 'convergence.pdf'} (gamma* = {gamma_safe:.3f})")


def write_markdown(rows, path):
    """Readable version of the experiment log."""
    lines = [
        (
            "| stage | preprocessing | model | features | F1 (mean ± std) "
            "| accuracy | F1 @ 0.5 | threshold |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['preprocessing']} | {r['model']} "
            f"| {r['n_features']} | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
            f"| {r['accuracy_mean']:.4f} | {r['f1_default_threshold']:.4f} "
            f"| {r['threshold_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    start = time.time()
    x, _, y_pm1, _, _ = load_csv_data(DATA)
    y = (y_pm1 == 1).astype(float)  # {-1, 1} -> {0, 1}
    header = load_feature_names(DATA / "x_train.csv")
    print(f"data: {x.shape}, {y.mean():.1%} positives")
    rows = []

    # Stage 1: compare the models on the BRFSS preprocessing.
    models = {"least squares": (train_least_squares, linear_scores)}
    for lam in RIDGE_LAMBDAS:
        models[f"ridge, lambda={lam:g}"] = (
            lambda y, tx, lam=lam: train_ridge(y, tx, lam),
            linear_scores,
        )
    models["logistic"] = (train_logistic, logistic_scores)
    for lam in LOGISTIC_LAMBDAS:
        models[f"reg. logistic, lambda={lam:g}"] = (
            lambda y, tx, lam=lam: train_reg_logistic(y, tx, lam),
            logistic_scores,
        )
    stage1 = run("models", x, y, "BRFSS", brfss_prep(header), models, rows)

    best = max(
        LOGISTIC_LAMBDAS,
        key=lambda lam: stage1[f"reg. logistic, lambda={lam:g}"]["f1_mean"],
    )
    print(f"\nbest lambda for regularized logistic regression: {best:g}")

    def reg_logistic(y, tx):
        return train_reg_logistic(y, tx, best)

    model = {f"reg. logistic, lambda={best:g}": (reg_logistic, logistic_scores)}

    # Stage 2: ablation around the best model, one change at a time.
    run("ablation", x, y, "simple (mean imputation)", simple_prep(), model, rows)
    balanced = {
        f"reg. logistic, lambda={best:g}, undersampled": (
            undersampled(reg_logistic),
            logistic_scores,
        )
    }
    run("ablation", x, y, "BRFSS", brfss_prep(header), balanced, rows)
    variants = {
        "BRFSS + target-corr selection": brfss_prep(header, target_corr=True),
        "BRFSS + squares": brfss_prep(header, squares=True),
        "BRFSS + one-hot": brfss_prep(header, onehot=True),
        "BRFSS + one-hot + squares": brfss_prep(header, onehot=True, squares=True),
    }
    for name, prep in variants.items():
        run("ablation", x, y, name, prep, model, rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "cv_results.csv"
    if log_path.exists():
        log_path.unlink()  # one file per run, so the tables match the code
    log_results(log_path, rows)
    write_markdown(rows, RESULTS / "cv_results.md")
    print(f"\nwritten: {log_path} and {RESULTS / 'cv_results.md'}")

    convergence_figure(x, y, header)
    print(f"total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
