"""Feature expansion on top of the preprocessing of src/preprocessing.py.

Both steps are learned on the training rows only and then reused, unchanged,
on validation and test rows:

    params = fit_feature_expansion(z_train, kinds)
    z_train_big = apply_feature_expansion(z_train, params)
    z_test_big = apply_feature_expansion(z_test, params)

`kinds` gives "cat" or "cont" for each column of z; it comes from the
preprocessing params: [c["kind"] for c in params["columns"]].
"""

import numpy as np


def fit_feature_expansion(z, kinds, onehot=True, squares=True):
    """Learn the extra features on the (preprocessed) training rows z.

    onehot  : one indicator per level of each categorical column. The most
              frequent level is the reference and gets no indicator. A linear
              model can then give any weight to each answer of a question,
              instead of assuming that the answer codes are ordered numbers.
    squares : the square of each continuous column, so that the model can
              capture U-shaped effects (both very low and very high values).

    The new columns are standardized with their training mean and std.
    """
    kinds = np.asarray(kinds)
    levels = []
    if onehot:
        for j in np.where(kinds == "cat")[0]:
            values, counts = np.unique(np.round(z[:, j], 4), return_counts=True)
            levels.append((int(j), values[np.argsort(-counts)][1:]))
    cont = np.where(kinds == "cont")[0] if squares else np.array([], dtype=int)
    params = {"levels": levels, "cont": cont}
    extra = extra_features(z, params)
    std = extra.std(axis=0)
    params["mean"] = extra.mean(axis=0)
    params["std"] = np.where(std < 1e-8, 1.0, std)
    return params


def extra_features(z, params):
    """The new columns described by params, before standardization."""
    parts = [np.empty((len(z), 0))]
    for j, values in params["levels"]:
        col = np.round(z[:, j], 4)  # same rounding as when the levels were found
        parts.append((col[:, None] == values[None, :]).astype(np.float64))
    if len(params["cont"]):
        parts.append(z[:, params["cont"]].astype(np.float64) ** 2)
    return np.hstack(parts)


def apply_feature_expansion(z, params):
    """z followed by its standardized extra columns."""
    extra = (extra_features(z, params) - params["mean"]) / params["std"]
    return np.hstack([z.astype(np.float64), extra])
