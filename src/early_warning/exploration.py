"""Small query/summary helpers used by the exploration notebook (keeps notebook cells thin)."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config
from .modeling.dataset import labelled_population, time_split

S = config.SCHEMA_SRC


def _q(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def table_counts(engine: Engine) -> pd.DataFrame:
    tables = ["orders", "customers", "order_items", "order_payments", "order_reviews", "products", "sellers",
              "geolocation", "geo_zip", "product_category_translation"]
    sql = " UNION ALL ".join(f"SELECT '{t}' AS table_name, count(*) AS rows FROM {S}.{t}" for t in tables)
    return _q(engine, sql)


def approval_lag_summary(engine: Engine) -> pd.DataFrame:
    return _q(engine, f"""
        SELECT count(*) FILTER (WHERE order_approved_at IS NULL) AS never_approved,
               round(percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM order_approved_at - order_purchase_timestamp) / 60)::numeric, 1) AS median_lag_minutes,
               round(percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM order_approved_at - order_purchase_timestamp) / 3600)::numeric, 1) AS p95_lag_hours,
               round(percentile_cont(0.99) WITHIN GROUP (ORDER BY extract(epoch FROM order_approved_at - order_purchase_timestamp) / 3600)::numeric, 1) AS p99_lag_hours
        FROM {S}.orders""")


def never_approved_profile(engine: Engine) -> pd.DataFrame:
    return _q(engine, f"""
        SELECT order_status, count(*) AS orders,
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {S}.order_reviews r WHERE r.order_id = o.order_id)) AS with_review
        FROM {S}.orders o WHERE order_approved_at IS NULL GROUP BY 1 ORDER BY 2 DESC""")


def label_coverage(engine: Engine) -> pd.DataFrame:
    return _q(engine, f"""
        SELECT o.order_status, count(*) AS orders,
               count(*) FILTER (WHERE r.order_id IS NULL) AS without_review,
               round(100.0 * avg((r.review_score <= 2)::int), 1) AS negative_rate_pct
        FROM {S}.orders o
        LEFT JOIN (SELECT DISTINCT ON (order_id) order_id, review_score FROM {S}.order_reviews
                   ORDER BY order_id, review_answer_timestamp) r USING (order_id)
        GROUP BY 1 ORDER BY 2 DESC""")


def multi_review_orders(engine: Engine) -> pd.DataFrame:
    return _q(engine, f"""
        SELECT count(*) AS orders_with_several_reviews,
               count(*) FILTER (WHERE min_score <= 2 AND max_score >= 3) AS reviews_disagree_on_label
        FROM (SELECT order_id, min(review_score) AS min_score, max(review_score) AS max_score
              FROM {S}.order_reviews GROUP BY order_id HAVING count(*) > 1) x""")


def review_lag_quantiles(engine: Engine) -> pd.DataFrame:
    return _q(engine, f"""
        SELECT q, round(percentile_cont(q) WITHIN GROUP (ORDER BY
               extract(epoch FROM r.review_answer_timestamp - o.order_approved_at) / 86400.0)::numeric, 1) AS days_after_approval
        FROM {S}.order_reviews r JOIN {S}.orders o USING (order_id),
             unnest(ARRAY[0.5, 0.9, 0.95, 0.99, 0.999]) AS q
        WHERE o.order_approved_at IS NOT NULL GROUP BY q ORDER BY q""")


def boundary_sensitivity(frame: pd.DataFrame, boundaries=("2018-02-01", "2018-04-01", "2018-06-01")) -> pd.DataFrame:
    rows = []
    for b in boundaries:
        s = time_split(frame, boundary=b).summary
        rows.append({"boundary": b, "train_rows": s["train_rows"], "test_rows": s["test_rows"],
                     "train_negative_rate": s["train_negative_rate"], "test_negative_rate": s["test_negative_rate"],
                     "test_share_of_labelled": round(s["test_rows"] / (s["train_rows"] + s["test_rows"]
                                                                       + s["embargoed_rows"] + s["purged_rows"]), 3)})
    return pd.DataFrame(rows)


def monthly_volume_and_rate(frame: pd.DataFrame) -> pd.DataFrame:
    lab = labelled_population(frame)
    g = lab.assign(month=lab["decision_ts"].dt.to_period("M").astype(str)).groupby("month")["is_negative"]
    return g.agg(orders="size", negative_rate="mean").reset_index()


def rate_by_bucket(df: pd.DataFrame, column: str, q: int = 5) -> pd.DataFrame:
    lab = labelled_population(df)
    bucket = pd.qcut(lab[column], q=q, duplicates="drop") if lab[column].nunique() > q else lab[column]
    out = lab.groupby(bucket, observed=True)["is_negative"].agg(orders="size", negative_rate="mean")
    return out.reset_index().rename(columns={column: column + "_bucket"})
