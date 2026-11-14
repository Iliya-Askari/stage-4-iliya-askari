"""Temporal audit: for every model column, WHEN in the life of an order does it exist?

Decision moment = order approval (orders.order_approved_at). A column is allowed only if its value
is known at, or before, that moment. The audit is machine readable on purpose: the anti-leak tests
compare the real feature columns against this registry, so a column cannot reach the model
without somebody writing down when it becomes available.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config

META_COLUMNS = ("order_id", "decision_ts", "decision_ts_imputed")


@dataclass(frozen=True)
class FeatureAudit:
    name: str
    kind: str                 # numeric | categorical | boolean
    sources: str              # raw columns / tables it derives from
    available_at: str         # moment in the order life when the value exists
    at_approval: bool
    evidence: str             # why we are certain (logic + the empirical check id)
    hypothesis: str           # what the feature tests
    aggregate: bool = False
    leak_control: str = ""    # how the time limit is enforced (mandatory for aggregates)


P = "purchase (customer places the order)"
A = "approval (decision moment)"
H = "history strictly before approval"

FEATURES: tuple[FeatureAudit, ...] = (
    FeatureAudit("purchase_hour", "numeric", "orders.order_purchase_timestamp", P, True,
                 "Timestamp is written when the order is placed; approval cannot precede it (E1).",
                 "Orders placed late at night wait longer for approval and handling."),
    FeatureAudit("purchase_dow", "numeric", "orders.order_purchase_timestamp", P, True,
                 "Same as purchase_hour (E1).", "Weekend orders are processed more slowly."),
    FeatureAudit("approval_lag_hours", "numeric", "orders.order_approved_at - order_purchase_timestamp", A, True,
                 "Both ends exist at the decision moment by definition (E1). Imputed for never-approved orders.",
                 "A slow payment approval signals a fragile order and an unhappy customer."),
    FeatureAudit("promised_days_from_purchase", "numeric",
                 "orders.order_estimated_delivery_date - order_purchase_timestamp", P, True,
                 "The estimate is shown to the customer at checkout; it is never earlier than purchase (E2).",
                 "A long promise irritates customers; a short one is easier to break."),
    FeatureAudit("promised_days_from_approval", "numeric",
                 "orders.order_estimated_delivery_date - order_approved_at", A, True,
                 "Estimate exists since purchase, approval is the decision moment (E2).",
                 "The time left for the seller after approval drives lateness risk."),
    FeatureAudit("ship_limit_slack_days", "numeric", "order_items.shipping_limit_date - order_approved_at", A, True,
                 "Seller deadline is derived from the purchase: it is never earlier than purchase (E3).",
                 "A tight handover deadline makes lateness more likely."),
    FeatureAudit("n_items", "numeric", "order_items (count)", P, True,
                 "The basket is fixed when the order is placed (E4).", "Larger baskets have more ways to go wrong."),
    FeatureAudit("n_sellers", "numeric", "order_items.seller_id (distinct)", P, True,
                 "Basket composition is fixed at purchase (E4).", "Multi-seller orders arrive in pieces."),
    FeatureAudit("n_products", "numeric", "order_items.product_id (distinct)", P, True,
                 "Basket composition is fixed at purchase (E4).", "Heterogeneous baskets are harder to fulfil."),
    FeatureAudit("total_price", "numeric", "order_items.price (sum)", P, True,
                 "Prices are shown at checkout (E4).", "Expensive orders raise customer expectations."),
    FeatureAudit("total_freight", "numeric", "order_items.freight_value (sum)", P, True,
                 "Freight is quoted at checkout (E4).", "High shipping cost makes customers harsher judges."),
    FeatureAudit("freight_ratio", "numeric", "order_items.freight_value / (price + freight_value)", P, True,
                 "Ratio of two checkout quantities (E4).", "Customers paying mostly for shipping punish slow delivery."),
    FeatureAudit("customer_state", "categorical", "customers.customer_state", P, True,
                 "Delivery address is entered at checkout (E5).", "Remote states have worse delivery performance."),
    FeatureAudit("c_prior_orders", "numeric", "orders of the same customer_unique_id", H, True,
                 "Counts only earlier approvals (strictly before this decision timestamp).",
                 "Repeat customers know the platform and complain less.", True,
                 "Event stream window: only events with ts < decision_ts of this order are summed."),
    FeatureAudit("c_neg_rate_known", "numeric", "reviews of the customer's earlier orders", H, True,
                 "Uses a review only after its ANSWER timestamp, not its creation date.",
                 "Customers who complained before complain again.", True,
                 "Review counted only if review_answer_timestamp < decision_ts; shrunk to a fixed prior."),
    FeatureAudit("seller_state", "categorical", "sellers.seller_state", P, True,
                 "Seller of the first item is known at purchase (E4).", "Seller region drives transit time."),
    FeatureAudit("s_prior_orders", "numeric", "earlier orders of the primary seller", H, True,
                 "Counts only earlier approvals.", "Experienced sellers fulfil more reliably.", True,
                 "Event stream window, strict inequality on decision_ts."),
    FeatureAudit("s_known_reviews", "numeric", "answered reviews of the seller before this decision", H, True,
                 "Counts a review only after review_answer_timestamp.",
                 "The amount of evidence behind the seller's reputation.", True,
                 "Review counted only if review_answer_timestamp < decision_ts."),
    FeatureAudit("s_neg_rate_known", "numeric", "review_score of the seller's earlier orders", H, True,
                 "Shrunken share of negative reviews already ANSWERED before the decision.",
                 "Sellers with a bad recent reputation will disappoint again.", True,
                 "Review counted only if review_answer_timestamp < decision_ts; prior is a constant."),
    FeatureAudit("s_known_deliveries", "numeric", "delivered orders of the seller before this decision", H, True,
                 "Counts a delivery only after order_delivered_customer_date.",
                 "The amount of evidence behind the seller's punctuality.", True,
                 "Delivery counted only if order_delivered_customer_date < decision_ts."),
    FeatureAudit("s_late_rate_known", "numeric", "delivery vs estimate of the seller's earlier orders", H, True,
                 "Share of orders already DELIVERED before the decision that arrived after the estimate.",
                 "Sellers that were late recently will be late again.", True,
                 "Delivery counted only if order_delivered_customer_date < decision_ts; prior is a constant."),
    FeatureAudit("product_category", "categorical", "products.product_category_name (+ translation)", P, True,
                 "Catalogue attribute of the first item (E6: static snapshot, caveat in decisions).",
                 "Some categories (furniture, electronics) disappoint more often."),
    FeatureAudit("product_weight_g", "numeric", "products.product_weight_g", P, True, "Catalogue attribute (E6).",
                 "Heavy items are harder to ship on time."),
    FeatureAudit("product_volume_cm3", "numeric", "products length x height x width", P, True,
                 "Catalogue attribute (E6).", "Bulky items break more often in transit."),
    FeatureAudit("product_photos_qty", "numeric", "products.product_photos_qty", P, True,
                 "Catalogue attribute (E6).", "Poor listings create wrong expectations."),
    FeatureAudit("product_desc_len", "numeric", "products.product_description_lenght", P, True,
                 "Catalogue attribute (E6).", "Thin descriptions create wrong expectations."),
    FeatureAudit("product_name_len", "numeric", "products.product_name_lenght", P, True,
                 "Catalogue attribute (E6).", "Listing quality proxy."),
    FeatureAudit("payment_type_primary", "categorical", "order_payments.payment_type (largest payment)", A, True,
                 "Payment is captured before approval; approval depends on it (E7).",
                 "Boleto payers wait for the bank slip and are more fragile."),
    FeatureAudit("payment_installments_max", "numeric", "order_payments.payment_installments", A, True,
                 "Chosen at checkout (E7).", "Long instalment plans signal price-sensitive customers."),
    FeatureAudit("n_payments", "numeric", "order_payments (count)", A, True, "Captured before approval (E7).",
                 "Split payments (vouchers) complicate orders."),
    FeatureAudit("has_voucher", "boolean", "order_payments.payment_type = 'voucher'", A, True,
                 "Captured before approval (E7).", "Voucher users are promotion driven and less loyal."),
    FeatureAudit("total_payment_value", "numeric", "order_payments.payment_value (sum)", A, True,
                 "Captured before approval (E7).", "Order size in money terms."),
    FeatureAudit("distance_km", "numeric", "geolocation centroids of seller and customer zip prefixes", P, True,
                 "Both zip prefixes are known at purchase; the geolocation table is a static reference (E6).",
                 "Long distances mean long transit and more lateness."),
    FeatureAudit("same_state", "boolean", "customers.customer_state = sellers.seller_state", P, True,
                 "Both states known at purchase (E5).", "Cross-state shipping is slower."),
    FeatureAudit("n_holidays_in_promise_window", "numeric",
                 "external API: public holidays between approval date and estimated delivery date", A, True,
                 "Calendar is public and fixed in advance; the window ends at the estimate (E2). EXTERNAL SIGNAL.",
                 "Holidays close carriers and post offices, so promises spanning them are broken more often."),
)

FEATURE_NAMES = tuple(f.name for f in FEATURES)


@dataclass(frozen=True)
class Excluded:
    column: str
    available_at: str
    reason: str
    tempting_because: str


EXCLUDED: tuple[Excluded, ...] = (
    Excluded("orders.order_status", "after approval (final status of a finished order)",
             "The table stores the FINAL status. 'canceled' and 'unavailable' are decided after approval.",
             "Negative-review rate is 76% for canceled and 85% for unavailable orders against 13% for delivered (E8)."),
    Excluded("orders.order_delivered_customer_date", "after delivery",
             "Exists only once the parcel arrives, weeks after the decision. Its NULLs also encode status.",
             "Delivery delay is the strongest single driver of a bad review."),
    Excluded("orders.order_delivered_carrier_date", "after the seller hands over the parcel",
             "Seller handover happens after approval; its NULLs also encode status.",
             "Handover delay predicts delivery delay."),
    Excluded("order_reviews.review_score / comment fields", "after the customer answers the survey",
             "This is the label itself (or text written together with it).", "Perfect correlation with the target."),
    Excluded("order_reviews.review_creation_date", "survey send date, mostly after delivery (E9)",
             "Only 8% of surveys are created before delivery; it is a post-approval timestamp that also "
             "proves a review exists.", "Looks like an innocent order timestamp."),
    Excluded("naive all-time seller negative rate", "needs reviews that do not exist yet",
             "A GROUP BY over all reviews of a seller includes future reviews AND the order's own review.",
             "The most natural way to build a reputation feature; the classic leak the brief warns about."),
)


def audit_table() -> pd.DataFrame:
    """The audit table required by the brief: one row per model column."""
    return pd.DataFrame([{
        "column": f.name, "available_at": f.available_at,
        "available_at_approval": "yes" if f.at_approval else "no",
        "source_of_certainty": f.evidence, "derived_from": f.sources,
        "aggregate": "yes" if f.aggregate else "no",
    } for f in FEATURES])


def excluded_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "column": e.column, "available_at": e.available_at, "why_excluded": e.reason,
        "why_tempting": e.tempting_because} for e in EXCLUDED])


def hypothesis_table() -> pd.DataFrame:
    return pd.DataFrame([{"feature": f.name, "hypothesis": f.hypothesis,
                          "temporal_control": f.leak_control or "row-local, no history involved"}
                         for f in FEATURES])


def undeclared_columns(columns: list[str]) -> list[str]:
    """Columns that are neither audited features nor allowed meta columns."""
    allowed = set(FEATURE_NAMES) | set(META_COLUMNS)
    return sorted(c for c in columns if c not in allowed)


def assert_columns_audited(columns: list[str]) -> None:
    bad = undeclared_columns(columns)
    if bad:
        raise AssertionError(f"columns without a temporal audit entry: {bad}")
    not_ok = [f.name for f in FEATURES if f.name in columns and not f.at_approval]
    if not_ok:
        raise AssertionError(f"columns marked as NOT available at approval: {not_ok}")


EVIDENCE_QUERIES: dict[str, tuple[str, str]] = {
    "E1": ("orders approved before purchase (must be 0)",
           "SELECT count(*) FROM {s}.orders WHERE order_approved_at < order_purchase_timestamp"),
    "E2": ("orders whose estimated delivery date is before the purchase date (must be 0)",
           "SELECT count(*) FROM {s}.orders WHERE order_estimated_delivery_date::date < order_purchase_timestamp::date"),
    "E3": ("items whose shipping limit is before the purchase (must be 0)",
           "SELECT count(*) FROM {s}.order_items i JOIN {s}.orders o USING (order_id) "
           "WHERE i.shipping_limit_date < o.order_purchase_timestamp"),
    "E3b": ("items whose shipping limit is before the approval (tolerated, tiny share)",
            "SELECT count(*) FROM {s}.order_items i JOIN {s}.orders o USING (order_id) "
            "WHERE i.shipping_limit_date < o.order_approved_at"),
    "E4": ("share of orders that have item rows, percent",
           "SELECT round(100.0 * count(DISTINCT order_id) / (SELECT count(*) FROM {s}.orders), 2) FROM {s}.order_items"),
    "E7": ("share of orders that have at least one payment row",
           "SELECT round(100.0 * count(DISTINCT order_id) / (SELECT count(*) FROM {s}.orders), 2) "
           "FROM {s}.order_payments"),
    "E8": ("negative-review rate of canceled orders, percent",
           "SELECT round(100.0 * avg((r.review_score <= 2)::int), 1) FROM {s}.orders o "
           "JOIN {s}.order_reviews r USING (order_id) WHERE o.order_status = 'canceled'"),
    "E8b": ("negative-review rate of delivered orders, percent",
            "SELECT round(100.0 * avg((r.review_score <= 2)::int), 1) FROM {s}.orders o "
            "JOIN {s}.order_reviews r USING (order_id) WHERE o.order_status = 'delivered'"),
    "E9": ("share of surveys created BEFORE the parcel was delivered, percent",
           "SELECT round(100.0 * avg((r.review_creation_date < o.order_delivered_customer_date)::int), 1) "
           "FROM {s}.order_reviews r JOIN {s}.orders o USING (order_id) "
           "WHERE o.order_delivered_customer_date IS NOT NULL"),
    "E10": ("reviews answered before the order approval (data anomaly, tiny)",
            "SELECT count(*) FROM {s}.order_reviews r JOIN {s}.orders o USING (order_id) "
            "WHERE r.review_answer_timestamp < o.order_approved_at"),
    "E11": ("p99 of days between approval and review answer (drives the label embargo)",
            "SELECT round(percentile_cont(0.99) WITHIN GROUP (ORDER BY "
            "extract(epoch FROM (r.review_answer_timestamp - o.order_approved_at)) / 86400.0)::numeric, 1) "
            "FROM {s}.order_reviews r JOIN {s}.orders o USING (order_id) WHERE o.order_approved_at IS NOT NULL"),
    "E12": ("orders with no approval timestamp",
            "SELECT count(*) FROM {s}.orders WHERE order_approved_at IS NULL"),
    "E13": ("orders without any review (label unknown)",
            "SELECT count(*) FROM {s}.orders o WHERE NOT EXISTS (SELECT 1 FROM {s}.order_reviews r WHERE r.order_id = o.order_id)"),
}


def run_evidence(engine: Engine, src: str = config.SCHEMA_SRC) -> pd.DataFrame:
    rows = []
    with engine.connect() as conn:
        for key, (desc, sql) in EVIDENCE_QUERIES.items():
            value = conn.execute(text(sql.format(s=src))).scalar()
            rows.append({"check": key, "what": desc, "value": float(value) if value is not None else None})
    return pd.DataFrame(rows)
