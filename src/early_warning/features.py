"""Feature layer driver: builds the SQL views, snapshots them, measures the query."""
from __future__ import annotations

import statistics
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config
from .db import run_sql_file


def sql_tokens(src: str = config.SCHEMA_SRC, ext: str = config.SCHEMA_EXT,
               dst: str = config.SCHEMA_FEAT) -> dict[str, str]:
    return {
        "SRC": src, "EXT": ext, "DST": dst,
        "NEG_MAX": str(config.NEGATIVE_MAX_SCORE),
        "IMPUTED_LAG": str(config.IMPUTED_APPROVAL_LAG_MINUTES),
        "PRIOR_STRENGTH": repr(config.PRIOR_STRENGTH),
        "PRIOR_NEG": repr(config.PRIOR_NEG_RATE),
        "PRIOR_LATE": repr(config.PRIOR_LATE_RATE),
    }


def build_feature_layer(engine: Engine, src: str = config.SCHEMA_SRC, ext: str = config.SCHEMA_EXT,
                        dst: str = config.SCHEMA_FEAT) -> None:
    """Create views + materialised roll-ups, then take the snapshot through the stored procedure."""
    tokens = sql_tokens(src, ext, dst)
    run_sql_file(engine, "04_features.sql", **{k: v for k, v in tokens.items()})
    with engine.begin() as conn:
        conn.execute(text(f"CALL {dst}.sp_build_feature_snapshot()"))


def build_leaky_layer(engine: Engine, src: str = config.SCHEMA_SRC, leaky: str = config.SCHEMA_LEAKY) -> None:
    tokens = sql_tokens(src=src)
    run_sql_file(engine, "05_leaky.sql", LEAKY=leaky, SRC=src, NEG_MAX=tokens["NEG_MAX"])


def feature_columns(engine: Engine, dst: str = config.SCHEMA_FEAT, view: str = "v_features") -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = :s AND table_name = :t "
            "ORDER BY ordinal_position"), {"s": dst, "t": view}).all()
    return [r[0] for r in rows]


SESSION_WORK_MEM = "256MB"


def explain_feature_query(engine: Engine, dst: str = config.SCHEMA_FEAT) -> str:
    with engine.connect() as conn:
        conn.execute(text(f"SET work_mem = '{SESSION_WORK_MEM}'"))
        rows = conn.execute(text(f"EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) SELECT * FROM {dst}.v_features")).all()
    return "\n".join(r[0] for r in rows)


def time_feature_query(engine: Engine, dst: str = config.SCHEMA_FEAT, repeats: int = 5) -> dict:
    """Server-side execution time (EXPLAIN ANALYZE) and client-side time including the fetch."""
    server, client = [], []
    for _ in range(repeats):
        plan = explain_feature_query(engine, dst)
        for line in plan.splitlines():
            if line.strip().startswith("Execution Time:"):
                server.append(float(line.split(":")[1].replace("ms", "").strip()))
        t0 = time.perf_counter()
        with engine.connect() as conn:
            conn.execute(text(f"SET work_mem = '{SESSION_WORK_MEM}'"))
            conn.execute(text(f"SELECT * FROM {dst}.v_features")).fetchall()
        client.append((time.perf_counter() - t0) * 1000)
    return {
        "server_ms_median": round(statistics.median(server), 1), "server_ms_all": [round(x, 1) for x in server],
        "client_fetch_ms_median": round(statistics.median(client), 1),
        "budget_ms": config.FEATURE_QUERY_BUDGET_MS,
    }


def read_snapshot(engine: Engine, dst: str = config.SCHEMA_FEAT) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {dst}.feature_snapshot"), engine)


def read_labels(engine: Engine, dst: str = config.SCHEMA_FEAT) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT order_id, review_score, label_known_ts, is_negative FROM {dst}.v_labels"),
                       engine)
