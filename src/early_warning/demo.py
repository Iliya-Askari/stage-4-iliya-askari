"""Helpers that turn the raw CSVs into a small 'fresh orders' drop for demonstrating the prediction path."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .ingestion.contract import CONTRACTS
from .ingestion.loader import read_and_validate


def make_demo_slice(raw_dir: Path = config.DATA_RAW, out_dir: Path | None = None,
                    approved_from: str = "2018-08-15") -> Path:
    """Write orders (+ customers, items, payments) approved on/after a date into a drop directory."""
    out_dir = Path(out_dir or config.DATA_CURATED / "demo_new_orders")
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {t: read_and_validate(Path(raw_dir) / CONTRACTS[t].filename, t).clean
             for t in ("orders", "customers", "order_items", "order_payments")}
    orders = clean["orders"]
    orders = orders[orders["order_approved_at"] >= pd.Timestamp(approved_from)]
    ids = set(orders["order_id"])
    parts = {
        "orders": orders,
        "customers": clean["customers"][clean["customers"]["customer_id"].isin(set(orders["customer_id"]))],
        "order_items": clean["order_items"][clean["order_items"]["order_id"].isin(ids)],
        "order_payments": clean["order_payments"][clean["order_payments"]["order_id"].isin(ids)],
    }
    for name, df in parts.items():
        df.to_csv(out_dir / CONTRACTS[name].filename, index=False)
    broken = orders.drop(columns=["order_status"]).head(50)
    broken.to_csv(out_dir.parent / "demo_orders_missing_status.csv", index=False)
    return out_dir


def quarantine_demo(raw_dir: Path = config.DATA_RAW, n: int = 2000) -> pd.DataFrame:
    """Corrupt a few rows of the real reviews file in memory and show what the contract does with them."""
    from .ingestion.loader import read_raw, validate

    contract = CONTRACTS["order_reviews"]
    raw = read_raw(Path(raw_dir) / contract.filename, contract).head(n).copy()
    raw.loc[0, "review_score"] = "7"                      # outside 1..5
    raw.loc[1, "review_answer_timestamp"] = "31-31-2018"  # impossible date
    raw.loc[2, "order_id"] = ""                           # missing key
    res = validate(raw, contract)
    out = res.quarantine[["order_id", "review_score", "review_answer_timestamp", "_reason"]].copy()
    out["rows_in"], out["rows_clean"] = res.rows_in, len(res.clean)
    return out
