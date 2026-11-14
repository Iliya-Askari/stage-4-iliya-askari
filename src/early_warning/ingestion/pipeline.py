"""Ingestion orchestration: validate -> quarantine -> load into PostgreSQL -> external signal."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .. import config
from ..db import copy_dataframe, run_sql_file
from .contract import CONTRACTS, LOAD_ORDER
from .external import load_holidays
from .loader import (SchemaError, anomaly_counts, apply_foreign_keys, read_and_validate, sha256_of,
                     write_quarantine)
from .manifest import Manifest


def validate_directory(raw_dir: Path, manifest: Manifest | None = None,
                       quarantine_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every contract table from raw_dir, apply the contract and the foreign keys."""
    raw_dir = Path(raw_dir)
    quarantine_dir = Path(quarantine_dir or config.QUARANTINE_DIR)
    tables, stats, quarantined = {}, {}, {}
    for name in LOAD_ORDER:
        contract = CONTRACTS[name]
        path = raw_dir / contract.filename
        if not path.exists():
            if contract.required:
                raise SchemaError(f"required file missing: {contract.filename} (looked in {raw_dir.name}/)")
            continue
        result = read_and_validate(path, name)
        tables[name], stats[name], quarantined[name] = result.clean, result.stats, result.quarantine
        stats[name]["_path"] = path
    tables, orphans = apply_foreign_keys(tables)
    for name, bad in orphans.items():
        quarantined[name] = pd.concat([quarantined[name], bad], ignore_index=True)
    for name in tables:
        write_quarantine(name, quarantined[name], quarantine_dir)
        s = stats[name]
        s["rows_clean"] = int(len(tables[name]))
        s["rows_quarantined"] = int(len(quarantined[name]))
        for label in bad_labels(orphans.get(name)):
            s["reasons"][label] = s["reasons"].get(label, 0) + int((orphans[name]["_reason"] == label + ";").sum())
        if manifest is not None:
            path = s.pop("_path")
            manifest.add_table(name, path.name, sha256_of(path), s, len(tables[name]))
        else:
            s.pop("_path", None)
    if manifest is not None:
        manifest.data["anomalies"] = anomaly_counts(tables)
    return tables


def bad_labels(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return []
    return sorted({r.rstrip(";") for r in frame["_reason"].unique()})


def load_tables(engine: Engine, tables: dict[str, pd.DataFrame], schema: str = config.SCHEMA_SRC) -> None:
    run_sql_file(engine, "01_schema.sql", SRC=schema)
    for name in LOAD_ORDER:
        if name in tables:
            copy_dataframe(engine, f"{schema}.{name}", tables[name])
    with engine.begin() as conn:
        if "geolocation" in tables:
            conn.execute(text(f"""
                INSERT INTO {schema}.geo_zip
                SELECT geolocation_zip_code_prefix, avg(geolocation_lat), avg(geolocation_lng)
                FROM {schema}.geolocation
                WHERE geolocation_lat BETWEEN :la AND :lb AND geolocation_lng BETWEEN :ga AND :gb
                GROUP BY 1"""), {"la": config.BRAZIL_LAT[0], "lb": config.BRAZIL_LAT[1],
                                 "ga": config.BRAZIL_LNG[0], "gb": config.BRAZIL_LNG[1]})
    run_sql_file(engine, "02_indexes.sql", SRC=schema)


def load_external(engine: Engine, orders: pd.DataFrame, mode: str, manifest: Manifest | None) -> pd.DataFrame:
    first = int(orders["order_purchase_timestamp"].dt.year.min())
    last = int(orders["order_estimated_delivery_date"].dt.year.max())
    holidays, info = load_holidays(list(range(first, last + 1)), mode=mode)
    run_sql_file(engine, "03_external.sql", EXT=config.SCHEMA_EXT)
    copy_dataframe(engine, f"{config.SCHEMA_EXT}.holidays", holidays,
                   ["holiday_date", "name", "source"])
    if manifest is not None:
        manifest.add_external("public_holidays_br", info)
        if info["used_fallback"]:
            manifest.note("External API unreachable for at least one year: bundled holiday snapshot used.")
    return holidays


def ingest_all(raw_dir: Path, engine: Engine, manifest: Manifest | None = None,
               external_mode: str = "auto") -> dict[str, pd.DataFrame]:
    manifest = manifest or Manifest()
    with manifest.stage("ingest_validate"):
        tables = validate_directory(raw_dir, manifest)
    with manifest.stage("ingest_load_postgres"):
        load_tables(engine, tables)
    with manifest.stage("ingest_external_api"):
        load_external(engine, tables["orders"], external_mode, manifest)
    return tables


