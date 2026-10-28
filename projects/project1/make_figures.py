import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from cross_validation import f1_score, stratified_kfold
from experiments import FIGURES, K, RESULTS, SEED, brfss_prep, header_and_data
from training import logistic_scores, train_reg_logistic

FIGURES.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.pdf")
    plt.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close()


# Read the log; if an experiment appears twice (test runs), keep the last row.
latest = {}
for r in csv.DictReader(open(RESULTS / "cv_results.csv")):
    for k in ("f1_mean", "f1_std", "f1_default_threshold"):
        r[k] = float(r[k])
    latest[(r["stage"], r["preprocessing"], r["model"])] = r
rows = list(latest.values())

# Figure 1: model comparison, tuned threshold vs default 0.5
models = [r for r in rows if r["stage"] == "models"]
names = [
    r["model"].replace("lambda", "λ").replace("reg. logistic", "reg. log.")
    for r in models
]
pos = np.arange(len(models))
plt.figure(figsize=(4.2, 3.2))
plt.barh(pos + 0.2, [r["f1_mean"] for r in models], 0.4,
         xerr=[r["f1_std"] for r in models], capsize=2, label="tuned threshold")
plt.barh(pos - 0.2, [r["f1_default_threshold"] for r in models], 0.4,
         label="default threshold 0.5")
plt.yticks(pos, names, fontsize=6)
plt.xlabel("F1 (5-fold CV)")
plt.legend(fontsize=6)
save("models_f1")

# Figure 2: ablation (feature variants, MLP, blend) vs the baseline
base = next(r for r in models if r["model"] == "reg. logistic, lambda=0.001")
abl = [r for r in rows if r["stage"] in ("ablation", "mlp", "blend")]


def label(r):
    if r["stage"] == "mlp":
        return "MLP (64 hidden, 3 seeds)"
    if r["stage"] == "blend":
        return "logistic + MLP blend (one-hot)"
    return r["preprocessing"] + (" + undersampling" if "undersampled" in r["model"] else "")


abl.sort(key=lambda r: r["f1_mean"])
plt.figure(figsize=(4.2, 3.0))
plt.barh(range(len(abl)), [r["f1_mean"] for r in abl],
         xerr=[r["f1_std"] for r in abl], capsize=2)
plt.axvline(base["f1_mean"], color="k", ls="--", lw=0.8,
            label="baseline (BRFSS, reg. logistic)")
plt.yticks(range(len(abl)), [label(r) for r in abl], fontsize=6)
plt.xlim(0.40, max(r["f1_mean"] + r["f1_std"] for r in abl) + 0.01)
plt.xlabel("F1 (5-fold CV)")
plt.legend(fontsize=6)
save("ablation_f1")

# Figure 3: F1 as a function of the decision threshold (fold 1, reg. logistic)
x, y, header = header_and_data()
fit, apply = brfss_prep(header)
train, val = stratified_kfold(y, K, SEED)[0]
params, tx_tr = fit(x[train], y[train])
w = train_reg_logistic(y[train], tx_tr, 1e-3)
s = logistic_scores(apply(x[val], params), w)
ts = np.linspace(0.02, 0.7, 120)
f1s = [f1_score(y[val], (s >= t).astype(int)) for t in ts]
best_t = ts[int(np.argmax(f1s))]
plt.figure(figsize=(4.2, 2.6))
plt.plot(ts, f1s)
plt.axvline(0.5, color="gray", ls=":", label="default 0.5")
plt.axvline(best_t, color="r", ls="--", label=f"best t = {best_t:.2f}")
plt.xlabel("decision threshold")
plt.ylabel("validation F1")
plt.legend(fontsize=7)
save("f1_vs_threshold")

print("saved to", FIGURES)