"""Capacity driven evaluation. The unit of decision is a quota (top X%), never a probability threshold."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config


def top_k_mask(score: np.ndarray, frac: float) -> np.ndarray:
    """Flag the highest scores: exactly ceil(frac * n) orders (stable for ties)."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = max(1, math.ceil(frac * n)) if n else 0
    order = np.argsort(-score, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def capture_rate(y: np.ndarray, score: np.ndarray, frac: float = config.CAPACITY) -> float:
    """Share of ALL negative reviews that fall inside the top `frac` of the ranking (the headline metric)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    mask = top_k_mask(score, frac)
    return float(y[mask].sum() / y.sum())


def confusion_at(y: np.ndarray, score: np.ndarray, frac: float = config.CAPACITY) -> dict:
    y = np.asarray(y).astype(int)
    mask = top_k_mask(score, frac)
    tp = int((y[mask] == 1).sum())
    fp = int((y[mask] == 0).sum())
    fn = int((y[~mask] == 1).sum())
    tn = int((y[~mask] == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": int(mask.sum()),
            "precision": precision, "recall": recall, "f1": f1}


def capacity_table(y: np.ndarray, score: np.ndarray, fracs=config.CAPACITY_GRID) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    base = y.mean()
    rows = []
    for frac in fracs:
        c = confusion_at(y, score, frac)
        rows.append({
            "capacity": f"{int(frac * 100)}%", "orders_followed_up": c["flagged"],
            "negatives_caught": c["tp"], "negatives_total": int(y.sum()),
            "share_of_negatives_caught": c["recall"], "precision": c["precision"],
            "lift_vs_random": c["precision"] / base if base else float("nan"),
        })
    return pd.DataFrame(rows)


def full_report(y: np.ndarray, score: np.ndarray, frac: float = config.CAPACITY) -> dict:
    y = np.asarray(y).astype(int)
    c = confusion_at(y, score, frac)
    return {**c, "capacity": frac, "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)), "base_rate": float(y.mean()),
            "capture_rate": capture_rate(y, score, frac)}


def monthly_table(df: pd.DataFrame, score: np.ndarray, frac: float = config.CAPACITY,
                  min_rows: int = 500) -> pd.DataFrame:
    """Apply the same quota inside each calendar month (what the support team would do each month)."""
    frame = pd.DataFrame({"month": df["decision_ts"].dt.to_period("M").astype(str).to_numpy(),
                          "y": df["is_negative"].astype(int).to_numpy(), "score": np.asarray(score)})
    rows = []
    for month, g in frame.groupby("month"):
        if len(g) < min_rows or g["y"].nunique() < 2:
            continue
        c = confusion_at(g["y"].to_numpy(), g["score"].to_numpy(), frac)
        rows.append({"month": month, "orders": len(g), "negative_rate": g["y"].mean(),
                     "capture_rate": c["recall"], "precision": c["precision"],
                     "roc_auc": roc_auc_score(g["y"], g["score"])})
    return pd.DataFrame(rows)


def bootstrap_capture(y: np.ndarray, score: np.ndarray, frac: float = config.CAPACITY, n_boot: int = 300,
                      seed: int = config.SEED) -> tuple[float, float]:
    """95% interval of the capture rate by resampling orders."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    score = np.asarray(score)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(capture_rate(y[idx], score[idx], frac))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
