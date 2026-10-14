"""Preprocessing of the BRFSS 2015 data for Project 1

The pipeline is split in two steps so that the test set is transformed with
statistics learned on the training set only (no data leakage):

    header = load_feature_names("data/x_train.csv")
    params, z_train = fit_preprocessor(x_train, header)  # learn on train
    z_test = transform(x_test, header, params)  # reuse on any other data

`x` is the raw matrix returned by `helpers.load_csv_data` (no `Id` column) and
`header` the matching column names. If the `Id` column is still present in both
`x` and `header`, it is simply ignored.

Steps applied by `fit_preprocessor`, for every column and in this order:
    1. BRFSS special codes -> NaN ("refused", "don't know") or 0 ("none").
    2. Drop identifier columns, mixed-unit columns, columns with too many NaN
       and constant columns.
    3. Clip continuous columns at two quantiles (limits the effect of outliers).
    4. Replace NaN by the median (continuous) or the most frequent value
       (categorical).
    5. Standardize: z = (x - mean) / std.
    6. Drop columns almost perfectly correlated with an earlier kept column.

`params` is a plain dict that can be saved with `json.dump`.

The remaining helpers (`select_by_target_corr`, `undersample_majority`,
`duplicate_row_mask`) are the steps that need the labels or the full data set.
"""

import numpy as np

DEFAULT_CONFIG = {
    "nan_col_max": 0.50,  # drop columns with more than 50% NaN
    "cat_max_unique": 15,  # <= 15 distinct values: categorical, else continuous
    "clip_quantiles": [0.005, 0.995],  # clipping bounds of continuous columns
    "corr_x_max": 0.95,  # |corr| above this with a kept column: drop
}

# Identifiers, dates, survey bookkeeping and nominal codes that a linear model
# would wrongly read as ordered numbers.
DROP_ADMIN = [
    "IDATE",
    "SEQNO",
    "_PSU",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "FMONTH",
    "DISPCODE",
    "QSTVER",
    "_STSTR",
    "_STATE",
    "EXRACT11",
    "EXRACT21",
]

# Raw columns mixing several units in one field (e.g. ALCDAY5: 1xx = days per
# week, 2xx = days per month). Derived "_" variables already summarize them.
DROP_MIXED_UNITS = [
    "WEIGHT2",
    "HEIGHT3",
    "ALCDAY5",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    "STRENGTH",
    "BLDSUGAR",
    "FEETCHK2",
    "LONGWTCH",
    "FLSHTMY2",
    "HIVTSTD3",
    "ASACTLIM",
]

# Known special codes. "nan": refused / don't know. "zero": "none" (-> 0).
EXPLICIT_CODES = {
    "PHYSHLTH": {"zero": [88], "nan": [77, 99]},
    "MENTHLTH": {"zero": [88], "nan": [77, 99]},
    "POORHLTH": {"zero": [88], "nan": [77, 99]},
    "CHILDREN": {"zero": [88], "nan": [99]},
    "HHADULT": {"nan": [77, 99]},
    "DIABAGE2": {"nan": [77, 99]},
    "DRNK3GE5": {"zero": [88], "nan": [77, 99]},
    "AVEDRNK2": {"nan": [77, 99]},
    "MAXDRNKS": {"nan": [77, 99]},
    "DOCTDIAB": {"zero": [88], "nan": [77, 99]},
    "CHKHEMO3": {"zero": [88], "nan": [77, 99]},
    "FEETCHK": {"zero": [88], "nan": [77, 99]},
    "DROCDY3_": {"nan": [900]},
    "_DRNKWEK": {"nan": [99900]},
    "MAXVO2_": {"nan": [999]},
    "FC60_": {"nan": [999]},
}


def load_feature_names(csv_path):
    """Column names of a data CSV, without the leading `Id` column.

    `helpers.load_csv_data` drops `Id` and does not return the names, so use
    this to get the `header` matching its `x_train` / `x_test`.
    """
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    return header[1:] if header[0] == "Id" else header


def detect_codes(name, col, cat_max_unique):
    """Return `(nan_codes, zero_codes)` for one column.

    Known columns use `EXPLICIT_CODES`. For other columns with few values, a
    heuristic flags 7/9 (or 77/99) when they lie beyond the normal answer scale
    (e.g. answers 1..5 followed by 7 and 9). It is a heuristic: check the BRFSS
    2015 codebook when in doubt.
    """
    if name in EXPLICIT_CODES:
        known = EXPLICIT_CODES[name]
        return list(known.get("nan", [])), list(known.get("zero", []))
    values = np.unique(col[~np.isnan(col)])
    if (
        values.size == 0
        or values.size > cat_max_unique
        or not np.all(values == np.round(values))
    ):
        return [], []
    codes = []
    if values.max() <= 9:
        run = 0  # largest k such that 1..k are all present (the "normal" scale)
        while (run + 1) in values:
            run += 1
        codes = [int(v) for v in values if v in (7, 9) and v > run]
        if values.max() == 9 and 9 not in codes:
            codes.append(9)  # 9 means "refused" even when 1..9 is complete
    elif 77 in values or 99 in values:
        codes = [int(v) for v in (77, 99) if v in values]
    return sorted(codes), []


def apply_codes(col, nan_codes, zero_codes):
    """Copy of `col` with `zero_codes` replaced by 0 and `nan_codes` by NaN."""
    out = col.astype(np.float64, copy=True)
    if zero_codes:
        out[np.isin(out, zero_codes)] = 0.0
    if nan_codes:
        out[np.isin(out, nan_codes)] = np.nan
    return out


def transform(x, header, params, dtype=np.float32):
    """Apply a fitted preprocessor (`params`) to a raw matrix `x`.

    Returns the standardized matrix with one column per entry of
    `params["columns"]`, in the same order. Nothing is learned here.
    """
    index = {name: i for i, name in enumerate(header)}
    missing = [c["name"] for c in params["columns"] if c["name"] not in index]
    if missing:
        raise ValueError(f"columns missing from header: {missing[:5]}")
    z = np.empty((x.shape[0], len(params["columns"])), dtype=dtype)
    for k, p in enumerate(params["columns"]):
        col = apply_codes(x[:, index[p["name"]]], p["nan_codes"], p["zero_codes"])
        col = np.clip(col, p["lo"], p["hi"])  # NaN stay NaN
        col = np.where(np.isnan(col), p["fill"], col)
        z[:, k] = (col - p["mean"]) / p["std"]
    return z


def fit_preprocessor(x, header, cfg=None, dtype=np.float32):
    """Learn the preprocessing on `x` (training rows only).

    Returns `(params, z)`: the JSON-serializable parameters, and the
    preprocessed `x` (same result as `transform(x, header, params)`).
    `params["dropped"]` maps each reason to the columns removed for it.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    n = x.shape[0]
    dropped, columns, parts = {}, [], []

    def drop(reason, name):
        dropped.setdefault(reason, []).append(name)

    for j, name in enumerate(header):
        if name == "Id":
            continue
        if name in DROP_ADMIN:
            drop("administrative / unordered nominal code", name)
            continue
        if name in DROP_MIXED_UNITS:
            drop("mixed units (replaced by a derived variable)", name)
            continue
        nan_codes, zero_codes = detect_codes(name, x[:, j], cfg["cat_max_unique"])
        col = apply_codes(x[:, j], nan_codes, zero_codes)
        if np.isnan(col).mean() > cfg["nan_col_max"]:
            drop(f"more than {cfg['nan_col_max']:.0%} NaN", name)
            continue
        observed = col[~np.isnan(col)]
        n_unique = np.unique(observed).size
        if n_unique <= 1:
            drop("constant", name)
            continue
        kind = "cont" if n_unique > cfg["cat_max_unique"] else "cat"
        if kind == "cont":
            lo, hi = (float(v) for v in np.quantile(observed, cfg["clip_quantiles"]))
        else:
            lo, hi = float(observed.min()), float(observed.max())
        observed = np.clip(observed, lo, hi)
        if kind == "cont":
            fill = float(np.median(observed))
        else:
            values, counts = np.unique(observed, return_counts=True)
            fill = float(values[np.argmax(counts)])
        col = np.clip(col, lo, hi)
        col = np.where(np.isnan(col), fill, col)
        std = float(col.std())
        if std < 1e-8:
            drop("zero variance after cleaning", name)
            continue
        mean = float(col.mean())
        parts.append(((col - mean) / std).astype(dtype))
        columns.append(
            {
                "name": name,
                "kind": kind,
                "nan_codes": nan_codes,
                "zero_codes": zero_codes,
                "lo": lo,
                "hi": hi,
                "fill": fill,
                "mean": mean,
                "std": std,
            }
        )

    if not parts:
        raise ValueError("no column left after preprocessing")

    # Redundant columns: after standardization, corr(i, j) = (z_i . z_j) / n.
    z = np.column_stack(parts)
    corr = (z.T @ z) / n
    keep = np.ones(len(columns), dtype=bool)
    for j in range(1, len(columns)):
        earlier = np.where(keep[:j])[0]
        r = np.abs(corr[earlier, j])
        if r.max() > cfg["corr_x_max"]:
            keep[j] = False
            twin = columns[earlier[r.argmax()]]["name"]
            drop(
                f"correlated above {cfg['corr_x_max']} with a kept column",
                f"{columns[j]['name']} (~ {twin})",
            )
    params = {
        "config": cfg,
        "columns": [columns[i] for i in np.where(keep)[0]],
        "dropped": dropped,
    }
    return params, z[:, keep]


def duplicate_row_mask(a):
    """True for each row that duplicates an earlier row (first one is kept)."""
    a = np.ascontiguousarray(a)
    rows = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, first = np.unique(rows, return_index=True)
    mask = np.ones(len(a), dtype=bool)
    mask[first] = False
    return mask


def target_correlation(z, y):
    """Pearson correlation between each standardized column of `z` and `y`."""
    yc = y - y.mean()
    return (z.T @ yc) / (len(y) * y.std())


def select_by_target_corr(z, y, threshold):
    """Indices of the columns with |corr(column, y)| >= threshold.

    Uses the labels: to get unbiased validation scores, call it on the
    training part of each fold only.
    """
    return np.where(np.abs(target_correlation(z, y)) >= threshold)[0]


def undersample_majority(y, ratio, rng):
    """Shuffled row indices: all positives and `ratio` x n_pos random negatives.

    Apply it to training rows only, after the validation split.
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg = rng.choice(neg, size=min(len(neg), int(ratio * len(pos))), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx
