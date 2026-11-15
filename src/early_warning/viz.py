"""Small plotting helpers so the notebooks stay narrative (no plotting logic inside cells)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def capacity_curves(curves: dict[str, tuple], title: str = "Share of negative reviews caught vs follow-up capacity"):
    """curves: label -> (y_true, score)."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 4))
    grid = np.linspace(0.01, 0.5, 50)
    from .modeling.evaluate import capture_rate
    for label, (y, s) in curves.items():
        ax.plot(grid * 100, [capture_rate(y, s, g) * 100 for g in grid], label=label)
    ax.plot(grid * 100, grid * 100, "--", color="grey", label="random")
    ax.axvline(10, color="black", lw=0.8, ls=":")
    ax.set_xlabel("% of orders followed up")
    ax.set_ylabel("% of negative reviews caught")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def monthly_bars(model: pd.DataFrame, baseline: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(model))
    ax.bar([i - 0.2 for i in x], model["capture_rate"] * 100, width=0.4, label="model")
    ax.bar([i + 0.2 for i in x], baseline["capture_rate"] * 100, width=0.4, label="rule baseline")
    ax.axhline(10, color="grey", ls="--", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(model["month"], rotation=45)
    ax.set_ylabel("% of negatives caught (top 10%)")
    ax.legend()
    fig.tight_layout()
    return fig


def psi_bars(drift: pd.DataFrame, top: int = 15):
    d = drift.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = d["severity"].map({"major": "#c0392b", "moderate": "#e67e22", "stable": "#7f8c8d"})
    ax.barh(d["feature"], d["psi"], color=colors)
    ax.axvline(0.10, color="grey", ls=":", lw=0.8)
    ax.axvline(0.25, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("population stability index (train vs evaluation period)")
    fig.tight_layout()
    return fig


def leakage_bars(cost: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    d = cost.iloc[::-1]
    ax.barh(d["model"].str.slice(0, 48), d["capture_rate"] * 100,
            color=["#c0392b" if "CHEAT" in m else "#2c7fb8" for m in d["model"]])
    ax.set_xlabel("% of negative reviews caught at 10% capacity")
    fig.tight_layout()
    return fig


def volume_and_rate(monthly: pd.DataFrame, boundary: str):
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(monthly["month"], monthly["orders"], color="#bdc3c7")
    ax.set_ylabel("labelled orders")
    ax.tick_params(axis="x", rotation=60)
    ax2 = ax.twinx()
    ax2.plot(monthly["month"], monthly["negative_rate"] * 100, color="#c0392b", marker="o")
    ax2.set_ylabel("negative review rate %")
    idx = list(monthly["month"]).index(boundary[:7]) if boundary[:7] in set(monthly["month"]) else None
    if idx is not None:
        ax.axvline(idx - 0.5, color="black", ls="--")
    fig.tight_layout()
    return fig
