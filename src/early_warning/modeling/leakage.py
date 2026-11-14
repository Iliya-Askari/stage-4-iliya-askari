"""The price of leakage: a deliberately cheating model next to the honest one."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .. import config
from ..audit import FEATURE_NAMES
from .dataset import CATEGORICAL, NUMERIC, TimeSplit
from .evaluate import full_report
from .models import make_hgb, score

LEAKY_GROUPS: dict[str, dict[str, list[str]]] = {
    "post_approval_operations (status, delivery delay, handover)": {
        "numeric": ["delivery_delay_days", "carrier_handoff_days", "days_to_deliver"],
        "categorical": ["order_status"]},
    "naive_all_time_seller_negative_rate": {"numeric": ["seller_neg_rate_all_time"], "categorical": []},
    "review_comment_length": {"numeric": ["review_comment_len"], "categorical": []},
}


def load_leaky_columns(engine: Engine, leaky: str = config.SCHEMA_LEAKY) -> pd.DataFrame:
    df = pd.read_sql(text(f"SELECT * FROM {leaky}.v_leaky_columns"), engine)
    for col in df.columns:
        if col not in ("order_id", "order_status"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["order_status"] = df["order_status"].astype("string").fillna("missing").astype("object")
    return df


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, num: list[str], cat: list[str], label: str,
              frac: float) -> dict:
    model = make_hgb(max_leaf_nodes=24, max_iter=300, learning_rate=0.04, numeric=tuple(num),
                     categorical=tuple(cat))
    cols = num + cat
    model.fit(train[cols], train["is_negative"].astype(int))
    s = score(model, test[cols])
    rep = full_report(test["is_negative"].astype(int).to_numpy(), s, frac)
    return {"model": label, "extra_columns": len(cols) - len(FEATURE_NAMES),
            "capture_rate": rep["capture_rate"], "roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"],
            "precision": rep["precision"]}


def leakage_cost(split: TimeSplit, leaky: pd.DataFrame, frac: float = config.CAPACITY) -> pd.DataFrame:
    """Same learner, same split, same rows: the only difference is which columns it may see."""
    train = split.train.merge(leaky, on="order_id", how="left")
    test = split.test.merge(leaky, on="order_id", how="left")
    base_num, base_cat = list(NUMERIC), list(CATEGORICAL)
    rows = [_fit_eval(train, test, base_num, base_cat, "honest model (audited columns only)", frac)]
    all_num, all_cat = list(base_num), list(base_cat)
    for group, cols in LEAKY_GROUPS.items():
        rows.append(_fit_eval(train, test, base_num + cols["numeric"], base_cat + cols["categorical"],
                              f"honest + {group}", frac))
        all_num += cols["numeric"]
        all_cat += cols["categorical"]
    rows.append(_fit_eval(train, test, all_num, all_cat, "CHEAT model (all leaky columns)", frac))
    out = pd.DataFrame(rows)
    out["capture_rate_gain_vs_honest"] = out["capture_rate"] - out.loc[0, "capture_rate"]
    out["roc_auc_gain_vs_honest"] = out["roc_auc"] - out.loc[0, "roc_auc"]
    return out


