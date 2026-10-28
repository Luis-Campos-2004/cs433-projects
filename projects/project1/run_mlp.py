import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiments import RESULTS, brfss_prep, header_and_data, run
from cross_validation import log_results
from mlp import mlp_scores, train_mlp

x, y, header = header_and_data()
rows = []
run("mlp", x, y, "BRFSS", brfss_prep(header),
    {"MLP 64 hidden (3 seeds)": (train_mlp, mlp_scores)}, rows)
log_results(RESULTS / "cv_results.csv", rows)