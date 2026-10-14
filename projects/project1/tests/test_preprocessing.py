import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (
    apply_codes,
    detect_codes,
    duplicate_row_mask,
    fit_preprocessor,
    load_feature_names,
    select_by_target_corr,
    transform,
    undersample_majority,
)

HEADER = ["GENHLTH", "PHYSHLTH", "CONT", "CONT_COPY", "CONSTANT", "SPARSE", "IDATE"]


def make_data(n=2000, seed=0):
    """Small synthetic matrix mimicking the BRFSS quirks, columns as in HEADER."""
    rng = np.random.default_rng(seed)
    genhlth = rng.integers(1, 6, n).astype(float)
    genhlth[rng.random(n) < 0.05] = 9  # "refused"
    physhlth = rng.integers(0, 31, n).astype(float)
    physhlth[rng.random(n) < 0.5] = 88  # "none"
    physhlth[rng.random(n) < 0.02] = 77  # "don't know"
    cont = rng.normal(50, 10, n)
    cont[:5] = 1000  # outliers
    cont[rng.random(n) < 0.1] = np.nan
    sparse = rng.normal(size=n)
    sparse[rng.random(n) < 0.8] = np.nan
    return np.column_stack(
        [genhlth, physhlth, cont, 2 * cont + 1, np.full(n, 3.0), sparse, np.arange(n)]
    )


def test_apply_codes():
    col = np.array([1.0, 88.0, 77.0, 5.0])
    out = apply_codes(col, nan_codes=[77], zero_codes=[88])
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2]) and out[3] == 5.0
    assert col[1] == 88.0  # input is not modified


def test_detect_codes_heuristic_and_explicit():
    answers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, np.nan])
    assert detect_codes("ANYTHING", answers, 15) == ([7, 9], [])
    full_scale = np.arange(1.0, 10.0)  # 1..9 complete: only 9 is a code
    assert detect_codes("ANYTHING", full_scale, 15) == ([9], [])
    scale_77 = np.array([1.0, 2.0, 3.0, 77.0, 99.0])
    assert detect_codes("ANYTHING", scale_77, 15) == ([77, 99], [])
    many_values = np.arange(0.0, 100.0)  # continuous-like: untouched
    assert detect_codes("ANYTHING", many_values, 15) == ([], [])
    assert detect_codes("PHYSHLTH", answers, 15) == ([77, 99], [88])


def test_fit_preprocessor_drops_and_cleans():
    x = make_data()
    params, z = fit_preprocessor(x, HEADER)
    kept = [c["name"] for c in params["columns"]]
    assert kept == ["GENHLTH", "PHYSHLTH", "CONT"]
    reasons = " | ".join(params["dropped"])
    assert "constant" in reasons and "NaN" in reasons and "correlated" in reasons
    assert z.shape == (len(x), 3)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z.mean(axis=0, dtype=np.float64), 0, atol=1e-3)
    np.testing.assert_allclose(z.std(axis=0, dtype=np.float64), 1, atol=1e-3)


def test_transform_matches_fit_and_uses_train_statistics():
    x = make_data()
    params, z = fit_preprocessor(x, HEADER)
    np.testing.assert_allclose(transform(x, HEADER, params), z, atol=1e-5)
    shifted = make_data(seed=1)
    shifted[:, 2] += 100  # new data must NOT be re-standardized
    z_new = transform(shifted, HEADER, params)
    assert np.isfinite(z_new).all()
    assert z_new[:, 2].mean() > 1  # still far from the training mean


def test_transform_accepts_extra_columns_and_float64():
    x = make_data()
    params, _ = fit_preprocessor(x, HEADER)
    wider = np.column_stack([np.arange(len(x)), x])
    z = transform(wider, ["Id", *HEADER], params, dtype=np.float64)
    assert z.dtype == np.float64 and z.shape == (len(x), 3)


def test_transform_missing_column_raises():
    x = make_data()
    params, _ = fit_preprocessor(x, HEADER)
    with pytest.raises(ValueError):
        transform(x[:, :2], HEADER[:2], params)


def test_load_feature_names_drops_id(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Id,A,B\n1,2,3\n")
    assert load_feature_names(path) == ["A", "B"]


def test_duplicate_row_mask():
    a = np.array([[1.0, np.nan], [2.0, 3.0], [1.0, np.nan], [2.0, 3.0]])
    assert duplicate_row_mask(a).tolist() == [False, False, True, True]


def test_target_selection_and_undersampling():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(float)
    informative = (y - y.mean()) / y.std()
    noise = rng.normal(size=1000)
    noise = (noise - noise.mean()) / noise.std()
    z = np.column_stack([noise, informative])
    assert select_by_target_corr(z, y, 0.5).tolist() == [1]
    idx = undersample_majority(y, 1.0, np.random.default_rng(0))
    assert len(idx) == 2 * int(y.sum()) and y[idx].mean() == 0.5
