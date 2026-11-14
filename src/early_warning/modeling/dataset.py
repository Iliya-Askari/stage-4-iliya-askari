"""Modelling frame: features from the database + label, and the temporal train / test split."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy.engine import Engine

from .. import config
from ..audit import FEATURES, FEATURE_NAMES
from ..features import read_labels, read_snapshot

CATEGORICAL = tuple(f.name for f in FEATURES if f.kind == "categorical")
NUMERIC = tuple(f.name for f in FEATURES if f.kind in ("numeric", "boolean"))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast database types (Decimal, bool, None) to model friendly dtypes. Only audited columns survive."""
    out = df.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in CATEGORICAL:
        out[col] = out[col].astype("object").where(out[col].notna(), None)
        out[col] = out[col].astype("string").fillna("missing").astype("object")
    return out


def build_modeling_frame(engine: Engine, dst: str = config.SCHEMA_FEAT) -> pd.DataFrame:
    """Features (snapshot) joined with the label. The ONLY way the model layer touches data: via the DB."""
    feats = read_snapshot(engine, dst)
    labels = read_labels(engine, dst)
    df = feats.merge(labels[["order_id", "is_negative", "label_known_ts"]], on="order_id", how="left")
    df["decision_ts"] = pd.to_datetime(df["decision_ts"])
    df["label_known_ts"] = pd.to_datetime(df["label_known_ts"])
    df["decision_ts_imputed"] = df["decision_ts_imputed"].astype(bool)
    df = prepare_features(df)
    return df


@dataclass
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    summary: dict = field(default_factory=dict)


def labelled_population(df: pd.DataFrame) -> pd.DataFrame:
    """Orders the system is asked about (approved) whose outcome is known (a review exists)."""
    pop = df.loc[~df["decision_ts_imputed"]]
    return pop.loc[pop["is_negative"].notna()].copy()


def time_split(df: pd.DataFrame, boundary: str = config.SPLIT_BOUNDARY,
               embargo_days: int = config.LABEL_EMBARGO_DAYS) -> TimeSplit:
    """Train on the past, test on the future, with an embargo so no training label comes from the future.

    train : decision_ts < boundary - embargo  AND  label_known_ts < boundary   (purge)
    test  : decision_ts >= boundary
    """
    boundary_ts = pd.Timestamp(boundary)
    cutoff = boundary_ts - pd.Timedelta(days=embargo_days)
    lab = labelled_population(df)
    before = lab.loc[lab["decision_ts"] < cutoff]
    purged = before.loc[before["label_known_ts"] >= boundary_ts]
    train = before.loc[before["label_known_ts"] < boundary_ts]
    test = lab.loc[lab["decision_ts"] >= boundary_ts]
    embargoed = lab.loc[(lab["decision_ts"] >= cutoff) & (lab["decision_ts"] < boundary_ts)]
    summary = {
        "boundary": boundary_ts.date().isoformat(), "embargo_days": embargo_days,
        "train_cutoff": cutoff.date().isoformat(),
        "orders_total": int(len(df)),
        "excluded_never_approved": int(df["decision_ts_imputed"].sum()),
        "excluded_no_review": int((~df["decision_ts_imputed"] & df["is_negative"].isna()).sum()),
        "train_rows": int(len(train)), "train_negative_rate": round(float(train["is_negative"].astype(float).mean()), 4),
        "embargoed_rows": int(len(embargoed)), "purged_rows": int(len(purged)),
        "test_rows": int(len(test)), "test_negative_rate": round(float(test["is_negative"].astype(float).mean()), 4),
        "train_decision_range": [str(train["decision_ts"].min().date()), str(train["decision_ts"].max().date())],
        "test_decision_range": [str(test["decision_ts"].min().date()), str(test["decision_ts"].max().date())],
    }
    return TimeSplit(train.reset_index(drop=True), test.reset_index(drop=True), summary)


def inner_split(train: pd.DataFrame, validation_days: int = 60,
                embargo_days: int = config.LABEL_EMBARGO_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation block carved out of the TRAINING period only (the test period is never used to choose)."""
    edge = train["decision_ts"].max() - pd.Timedelta(days=validation_days)
    inner_train = train.loc[(train["decision_ts"] < edge - pd.Timedelta(days=embargo_days))
                            & (train["label_known_ts"] < edge)]
    validation = train.loc[train["decision_ts"] >= edge]
    return inner_train.reset_index(drop=True), validation.reset_index(drop=True)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs are restricted to the audited feature list, in a fixed order."""
    return df[list(FEATURE_NAMES)], df["is_negative"].astype(int)
