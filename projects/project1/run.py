"""Pipeline de ponta a ponta: dados -> modelo -> ficheiro de submissão.

Regra de ouro: se isto não corre num clone limpo do repo, a submissão não é
reprodutível. Os notebooks servem para explorar; o run.py é a verdade.
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
    """TODO: carregar train/test a partir de DATA."""
    raise NotImplementedError


def preprocess(x_train, x_test):
    """TODO: tratar NaNs, standardizar (estatísticas do TREINO), expandir features."""
    raise NotImplementedError


def train(y, tx):
    """TODO: escolher modelo + hiperparâmetros por cross-validation."""
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
    print(f"escrito: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
