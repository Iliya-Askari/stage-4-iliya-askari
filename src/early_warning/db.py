
"""Database helpers: engine, database creation, SQL templating, fast COPY loading."""
from __future__ import annotations

import csv
import io
from importlib import resources

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine, make_url

from . import config


class DatabaseUnavailable(RuntimeError):
    """Raised with an actionable message when PostgreSQL cannot be reached."""


def make_engine(url: str | None = None) -> Engine:
    return create_engine(
        url or config.database_url(),
        future=True,
        pool_pre_ping=True
    )


def ensure_database(url: str | None = None) -> None:
    """Create the target database if it does not exist (connects to the 'postgres' database)."""
    target = make_url(url or config.database_url())

    admin = create_engine(
        target.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        future=True
    )

    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": target.database}
            ).scalar()

            if not exists:
                conn.execute(
                    text(f'CREATE DATABASE "{target.database}"')
                )

    except Exception as exc:
        raise DatabaseUnavailable(
            f"Cannot reach PostgreSQL at {_safe(target)}: {exc.__class__.__name__}. "
            "Start PostgreSQL and set the DATABASE_URL environment variable, e.g. "
            "postgresql+psycopg://USER:PASSWORD@localhost:5432/early_warning"
        ) from exc

    finally:
        admin.dispose()


def _safe(url: URL) -> str:
    return url.render_as_string(hide_password=True)


def drop_database(url: str) -> None:
    target = make_url(url)

    admin = create_engine(
        target.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        future=True
    )

    try:
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": target.database}
            )

            conn.execute(
                text(f'DROP DATABASE IF EXISTS "{target.database}"')
            )

    finally:
        admin.dispose()


def read_sql_file(name: str, **tokens: str) -> str:
    sql = (
        resources.files("early_warning")
        .joinpath("sql", name)
        .read_text(encoding="utf-8")
    )

    for key, value in tokens.items():
        sql = sql.replace(f"__{key}__", value)

    return sql


def run_sql_file(engine: Engine, name: str, **tokens: str) -> None:
    sql = read_sql_file(name, **tokens)

    raw = engine.raw_connection()

    try:
        with raw.cursor() as cur:
            cur.execute(sql)

        raw.commit()

    except Exception:
        raw.rollback()
        raise

    finally:
        raw.close()


def copy_dataframe(
    engine: Engine,
    table: str,
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> int:
    """Bulk load a DataFrame with COPY using Psycopg3. Missing values become NULL."""

    if df.empty:
        return 0

    columns = columns or list(df.columns)

    buffer = io.StringIO()

    df[columns].to_csv(
        buffer,
        index=False,
        header=False,
        quoting=csv.QUOTE_MINIMAL,
        na_rep=""
    )

    buffer.seek(0)

    cols = ", ".join(f'"{c}"' for c in columns)

    raw = engine.raw_connection()

    try:
        with raw.cursor() as cur:
            with cur.copy(
                f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')"
            ) as copy:

                while data := buffer.read(8192):
                    copy.write(data)

        raw.commit()

    except Exception:
        raw.rollback()
        raise

    finally:
        raw.close()
        buffer.close()

    return len(df)
