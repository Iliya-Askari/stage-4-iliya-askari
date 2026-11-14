"""Distribution shift between the training period and the evaluation period."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..audit import FEATURES

EPS = 1e-4


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """Population stability index with quantile bins from the training period; NaN is its own bin."""
    tr = pd.to_numeric(train, errors="coerce")
    te = pd.to_numeric(test, errors="coerce")
    edges = np.unique(np.nanquantile(tr.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (near) constant feature
        e_share = np.array([tr.isna().mean(), 1 - tr.isna().mean()])
        a_share = np.array([te.isna().mean(), 1 - te.isna().mean()])
        return _psi_from_shares(e_share, a_share)
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(s: pd.Series) -> np.ndarray:
        cut = pd.cut(s, edges, labels=False)
        counts = np.bincount(cut.dropna().astype(int), minlength=len(edges) - 1).astype(float)
        counts = np.append(counts, s.isna().sum())
        return counts / max(len(s), 1)

    return _psi_from_shares(shares(tr), shares(te))


def psi_categorical(train: pd.Series, test: pd.Series, top: int = 30) -> float:
    keep = train.value_counts().head(top).index
    tr = train.where(train.isin(keep), "__other__")
    te = test.where(test.isin(keep), "__other__")
    cats = sorted(set(tr.unique()) | set(te.unique()))
    e = tr.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = te.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi_from_shares(e, a)


def severity(psi: float) -> str:
    return "major" if psi >= 0.25 else "moderate" if psi >= 0.10 else "stable"


def drift_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        if f.kind == "categorical":
            value = psi_categorical(train[f.name].astype(str), test[f.name].astype(str))
        else:
            value = psi_numeric(train[f.name], test[f.name])
        shift = float("nan")
        if f.kind != "categorical":
            tr_med = pd.to_numeric(train[f.name], errors="coerce").median()
            te_med = pd.to_numeric(test[f.name], errors="coerce").median()
            shift = float(te_med - tr_med) if pd.notna(tr_med) and pd.notna(te_med) else float("nan")
        rows.append({"feature": f.name, "psi": value, "severity": severity(value),
                     "median_shift": shift, "aggregate_history_feature": f.aggregate})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def monthly_target_rate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(month=df["decision_ts"].dt.to_period("M").astype(str)).groupby("month")["is_negative"]
    out = g.agg(["mean", "size"]).rename(columns={"mean": "negative_rate", "size": "orders"})
    return out.reset_index()
