"""Two model families behind one interface. Both consume ONLY the audited feature columns."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from .. import config
from .dataset import CATEGORICAL, NUMERIC


def make_logistic(c: float = 0.3) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                      random_state=config.SEED)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, list(NUMERIC)),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), list(CATEGORICAL)),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=600, random_state=config.SEED))])


def make_hgb(max_leaf_nodes: int = 15, learning_rate: float = 0.05, max_iter: int = 250,
             min_samples_leaf: int = 100, numeric=NUMERIC, categorical=CATEGORICAL) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", list(numeric)),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), list(categorical)),
    ])
    mask = [False] * len(numeric) + [True] * len(categorical)
    clf = HistGradientBoostingClassifier(
        max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate, max_iter=max_iter,
        min_samples_leaf=min_samples_leaf, l2_regularization=1.0, categorical_features=mask,
        random_state=config.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


CANDIDATES = {
    "logistic_regression": lambda: make_logistic(),
    "gradient_boosting_small": lambda: make_hgb(max_leaf_nodes=8, max_iter=200),
    "gradient_boosting_deeper": lambda: make_hgb(max_leaf_nodes=24, max_iter=300, learning_rate=0.04),
}


def score(model: Pipeline, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
