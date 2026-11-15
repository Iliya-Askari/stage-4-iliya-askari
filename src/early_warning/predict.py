"""Prediction path for unseen orders: validate -> same feature SQL -> same model -> ranked list.

Fresh rows are loaded into an isolated scratch copy of the source schema, so the production tables
are never modified. History features (seller reputation, ...) still see everything that was known
before each fresh order was approved, because the feature SQL is time-gated, not data-gated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config
from .db import copy_dataframe, run_sql_file
from .features import build_feature_layer
from .ingestion.contract import CONTRACTS, LOAD_ORDER
from .ingestion.external import load_holidays
from .ingestion.loader import SchemaError, normalise_header, read_raw, validate, write_quarantine
from .modeling.dataset import prepare_features, xy
from .modeling.evaluate import top_k_mask
from .modeling.models import score
from .modeling.train import TrainedModel, load_model

REPLACE_KEY = {  # table -> column that identifies the rows a fresh file replaces
    "orders": "order_id", "order_items": "order_id", "order_payments": "order_id", "order_reviews": "order_id",
    "customers": "customer_id", "products": "product_id", "sellers": "seller_id",
}
OUTPUT_COLUMNS = ["rank", "order_id", "risk_score", "follow_up", "decision_ts", "approval_imputed"]


class PredictionInputError(ValueError):
    """The input cannot be turned into a ranked list (clear message for the operator)."""


def _read_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    return [normalise_header(c) for c in first.strip().split(",")]


def collect_input(path: Path) -> dict[str, pd.DataFrame]:
    """Return validated fresh tables. A single file must be the orders table; a directory may hold any
    subset of the source tables (orders is mandatory)."""
    path = Path(path)
    if not path.exists():
        raise PredictionInputError(f"input not found: {path.name}")
    files: dict[str, Path] = {}
    if path.is_dir():
        for name in LOAD_ORDER:
            candidate = path / CONTRACTS[name].filename
            if candidate.exists():
                files[name] = candidate
        if "orders" not in files:
            raise SchemaError(f"directory '{path.name}' has no {CONTRACTS['orders'].filename}")
    else:
        header = _read_header(path)
        needed = set(CONTRACTS["orders"].column_names)
        if not needed <= set(header):
            raise SchemaError(
                f"'{path.name}' does not match the orders schema; missing columns "
                f"{sorted(needed - set(header))}. Expected columns: {CONTRACTS['orders'].column_names}")
        files["orders"] = path
    fresh: dict[str, pd.DataFrame] = {}
    for name, file in files.items():
        contract = CONTRACTS[name]
        result = validate(read_raw(file, contract), contract)
        if len(result.quarantine):
            write_quarantine(f"predict_{name}", result.quarantine, config.QUARANTINE_DIR)
        fresh[name] = result.clean
        if name == "orders" and result.clean.empty:
            raise PredictionInputError("no valid order rows in the input (all rows were quarantined)")
    return fresh


def _drop_foreign_keys(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT conrelid::regclass::text, conname FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace WHERE c.contype = 'f' AND n.nspname = :s"),
            {"s": schema}).all()
        for table, name in rows:
            conn.execute(text(f'ALTER TABLE {table} DROP CONSTRAINT "{name}"'))


def _load_scratch(engine: Engine, fresh: dict[str, pd.DataFrame], external_mode: str) -> None:
    src, infer = config.SCHEMA_SRC, config.SCHEMA_INFER_SRC
    run_sql_file(engine, "01_schema.sql", SRC=infer)
    _drop_foreign_keys(engine, infer)
    with engine.begin() as conn:
        for name in LOAD_ORDER:
            cols = ", ".join(c for c in CONTRACTS[name].column_names)
            if name == "geolocation":
                continue                                    # centroids are copied below
            where = ""
            if name in fresh and name in REPLACE_KEY:
                key = REPLACE_KEY[name]
                ids = tuple(set(fresh[name][key]))
                if ids:
                    where = f" WHERE {key} NOT IN :ids"
            stmt = text(f"INSERT INTO {infer}.{name} ({cols}) SELECT {cols} FROM {src}.{name}{where}")
            if where:
                stmt = stmt.bindparams(ids=ids)
            conn.execute(stmt)
        conn.execute(text(f"INSERT INTO {infer}.geo_zip SELECT * FROM {src}.geo_zip"))
    for name, df in fresh.items():
        if name == "geolocation":
            inside = df[df.geolocation_lat.between(*config.BRAZIL_LAT) & df.geolocation_lng.between(*config.BRAZIL_LNG)]
            cent = inside.groupby("geolocation_zip_code_prefix", as_index=False)[
                ["geolocation_lat", "geolocation_lng"]].mean()
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {infer}.geo_zip WHERE zip_prefix IN :z").bindparams(
                    z=tuple(cent.geolocation_zip_code_prefix) or ("",)))
            copy_dataframe(engine, f"{infer}.geo_zip", cent.rename(columns={
                "geolocation_zip_code_prefix": "zip_prefix", "geolocation_lat": "lat", "geolocation_lng": "lng"}))
        elif name == "product_category_translation":
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {infer}.product_category_translation"))
            copy_dataframe(engine, f"{infer}.{name}", df)
        else:
            copy_dataframe(engine, f"{infer}.{name}", df)
    _load_holidays(engine, fresh["orders"], external_mode)


def _load_holidays(engine: Engine, orders: pd.DataFrame, external_mode: str) -> None:
    infer_ext = "ext_infer"
    run_sql_file(engine, "03_external.sql", EXT=infer_ext)
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO {infer_ext}.holidays SELECT * FROM {config.SCHEMA_EXT}.holidays"))
        have = {int(r[0]) for r in conn.execute(text(
            f"SELECT DISTINCT extract(year FROM holiday_date) FROM {infer_ext}.holidays"))}
    need = set(range(int(orders["order_purchase_timestamp"].dt.year.min()),
                     int(orders["order_estimated_delivery_date"].dt.year.max()) + 1))
    missing = sorted(need - have)
    if missing:
        extra, _ = load_holidays(missing, mode=external_mode)
        copy_dataframe(engine, f"{infer_ext}.holidays", extra, ["holiday_date", "name", "source"])


def rank_orders(order_ids: list[str], features: pd.DataFrame, model: TrainedModel,
                capacity: float = config.CAPACITY) -> pd.DataFrame:
    frame = prepare_features(features)
    X, _ = xy(frame.assign(is_negative=0))
    s = score(model.pipeline, X)
    mask = top_k_mask(s, capacity)
    out = pd.DataFrame({
        "order_id": frame["order_id"].to_numpy(), "risk_score": s, "follow_up": mask,
        "decision_ts": frame["decision_ts"].to_numpy(),
        "approval_imputed": frame["decision_ts_imputed"].astype(bool).to_numpy()})
    out = out.sort_values(["risk_score", "order_id"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[OUTPUT_COLUMNS]


def predict(input_path: Path, engine: Engine, model: TrainedModel | None = None,
            capacity: float = config.CAPACITY, external_mode: str = "auto") -> pd.DataFrame:
    """Entry point used by the CLI and the tests. Returns the ranked follow-up list."""
    if not 0 < capacity <= 1:
        raise PredictionInputError("capacity must be in (0, 1]")
    fresh = collect_input(Path(input_path))      # schema problems must surface before anything else
    model = model or load_model()
    infer_src, infer_feat = config.SCHEMA_INFER_SRC, config.SCHEMA_INFER_FEAT
    try:
        _load_scratch(engine, fresh, external_mode)
        build_feature_layer(engine, src=infer_src, ext="ext_infer", dst=infer_feat)
        ids = list(fresh["orders"]["order_id"])
        with engine.connect() as conn:
            feats = pd.read_sql(text(f"SELECT * FROM {infer_feat}.feature_snapshot WHERE order_id IN :ids")
                                .bindparams(ids=tuple(ids)), conn)
    finally:
        with engine.begin() as conn:
            for schema in (infer_src, infer_feat, "ext_infer"):
                conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    if feats.empty:
        raise PredictionInputError("none of the input orders could be featurised")
    return rank_orders(ids, feats, model, capacity)


