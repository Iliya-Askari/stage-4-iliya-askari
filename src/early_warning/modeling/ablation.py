"""Does the external signal add anything? Honest with / without comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..audit import FEATURES
from .dataset import TimeSplit, inner_split
from .evaluate import capture_rate
from .models import make_hgb, score

EXTERNAL = "n_holidays_in_promise_window"


def _fit(train: pd.DataFrame, drop: tuple[str, ...]):
    num = tuple(f.name for f in FEATURES if f.kind != "categorical" and f.name not in drop)
    cat = tuple(f.name for f in FEATURES if f.kind == "categorical" and f.name not in drop)
    model = make_hgb(max_leaf_nodes=24, max_iter=300, learning_rate=0.04, numeric=num, categorical=cat)
    model.fit(train[list(num + cat)], train["is_negative"].astype(int))
    return model, list(num + cat)


def external_signal_ablation(split: TimeSplit, frac: float = config.CAPACITY, n_boot: int = 300) -> dict:
    inner_train, validation = inner_split(split.train)
    out = {}
    for label, drop in (("with_external", ()), ("without_external", (EXTERNAL,))):
        m_in, cols = _fit(inner_train, drop)
        m_full, _ = _fit(split.train, drop)
        out[label] = {
            "validation_capture": capture_rate(validation["is_negative"].astype(int).to_numpy(),
                                               score(m_in, validation[cols]), frac),
            "test_scores": score(m_full, split.test[cols]),
        }
    y = split.test["is_negative"].astype(int).to_numpy()
    a, b = out["with_external"]["test_scores"], out["without_external"]["test_scores"]
    rng = np.random.default_rng(config.SEED)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(capture_rate(y[idx], a[idx], frac) - capture_rate(y[idx], b[idx], frac))
    return {
        "validation_capture_with": out["with_external"]["validation_capture"],
        "validation_capture_without": out["without_external"]["validation_capture"],
        "test_capture_with": capture_rate(y, a, frac), "test_capture_without": capture_rate(y, b, frac),
        "test_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }
