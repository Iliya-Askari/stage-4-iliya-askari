"""Rule baselines the operations team could apply by hand (no model)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import capture_rate

RULES = {
    "longest_promised_delivery": ("Follow up the orders with the longest promised delivery time after approval.",
                                  lambda d: d["promised_days_from_approval"].fillna(0).to_numpy()),
    "worst_seller_reputation": ("Follow up the orders of sellers with the highest share of bad reviews so far.",
                                lambda d: d["s_neg_rate_known"].fillna(d["s_neg_rate_known"].median()).to_numpy()),
    "farthest_shipping_distance": ("Follow up the orders travelling the longest distance.",
                                   lambda d: d["distance_km"].fillna(d["distance_km"].median()).to_numpy()),
}


def rule_scores(df: pd.DataFrame, rule: str) -> np.ndarray:
    return RULES[rule][1](df)


def choose_baseline(train: pd.DataFrame, frac: float) -> tuple[str, pd.DataFrame]:
    """Pick the rule that works best on the TRAINING period only."""
    y = train["is_negative"].astype(int).to_numpy()
    rows = [{"rule": name, "description": desc, "train_capture_rate": capture_rate(y, fn(train), frac)}
            for name, (desc, fn) in RULES.items()]
    table = pd.DataFrame(rows).sort_values("train_capture_rate", ascending=False).reset_index(drop=True)
    return str(table.loc[0, "rule"]), table
