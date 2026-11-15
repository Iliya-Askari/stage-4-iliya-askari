-- Feature layer. Tokens: __SRC__ source schema, __EXT__ external schema, __DST__ feature schema.
-- GOLDEN RULE: every column below may only use information that exists at order approval time.
-- History aggregates use an event-stream window: an order sees only events with ts STRICTLY BEFORE
-- its own decision timestamp (query rows sort before events that share the same timestamp).
SET work_mem = '256MB';   -- the history roll-ups sort ~300k event rows; keep them in memory
DROP SCHEMA IF EXISTS __DST__ CASCADE;
CREATE SCHEMA __DST__;

-- ---------------------------------------------------------------- per-order roll-ups
CREATE MATERIALIZED VIEW __DST__.mv_order_items AS
SELECT i.order_id,
       count(*)                                   AS n_items,
       count(DISTINCT i.seller_id)                AS n_sellers,
       count(DISTINCT i.product_id)               AS n_products,
       sum(i.price)                               AS total_price,
       sum(i.freight_value)                       AS total_freight,
       min(i.shipping_limit_date)                 AS first_ship_limit_ts,
       (array_agg(i.seller_id  ORDER BY i.order_item_id))[1] AS primary_seller_id,
       (array_agg(i.product_id ORDER BY i.order_item_id))[1] AS primary_product_id
FROM __SRC__.order_items i
GROUP BY i.order_id;
CREATE UNIQUE INDEX ON __DST__.mv_order_items (order_id);

CREATE MATERIALIZED VIEW __DST__.mv_order_payments AS
SELECT p.order_id,
       count(*)                              AS n_payments,
       max(p.payment_installments)           AS payment_installments_max,
       sum(p.payment_value)                  AS total_payment_value,
       bool_or(p.payment_type = 'voucher')   AS has_voucher,
       (array_agg(p.payment_type ORDER BY p.payment_value DESC, p.payment_sequential))[1] AS payment_type_primary
FROM __SRC__.order_payments p
GROUP BY p.order_id;
CREATE UNIQUE INDEX ON __DST__.mv_order_payments (order_id);

-- Cumulative holiday count per calendar day: replaces a per-order correlated sub-query
-- (99k index probes, >1 s) by two cheap hash joins.
CREATE MATERIALIZED VIEW __DST__.mv_holiday_cum AS
SELECT d::date AS day,
       (SELECT count(*) FROM __EXT__.holidays h WHERE h.holiday_date <= d::date)::int AS cum
FROM generate_series(
        (SELECT min(order_purchase_timestamp)::date - 1 FROM __SRC__.orders),
        (SELECT max(order_estimated_delivery_date)::date FROM __SRC__.orders),
        interval '1 day') AS d;
CREATE UNIQUE INDEX ON __DST__.mv_holiday_cum (day);

-- ---------------------------------------------------------------- decision moment
CREATE VIEW __DST__.v_core AS
SELECT o.order_id,
       o.customer_id,
       o.order_purchase_timestamp                                    AS purchase_ts,
       COALESCE(o.order_approved_at,
                o.order_purchase_timestamp + make_interval(mins => __IMPUTED_LAG__)) AS decision_ts,
       (o.order_approved_at IS NULL)                                 AS decision_ts_imputed,
       o.order_estimated_delivery_date                               AS estimated_ts,
       o.order_delivered_customer_date                               AS delivered_ts
FROM __SRC__.orders o;

-- ---------------------------------------------------------------- label (NOT a feature)
-- The first review that was ANSWERED. Its knowledge time is the answer timestamp, not the
-- creation date: creation is when the survey is sent, the score only exists once answered.
CREATE VIEW __DST__.v_labels AS
SELECT DISTINCT ON (r.order_id)
       r.order_id,
       r.review_score,
       r.review_answer_timestamp                       AS label_known_ts,
       (r.review_score <= __NEG_MAX__)                 AS is_negative
FROM __SRC__.order_reviews r
ORDER BY r.order_id, r.review_answer_timestamp, r.review_creation_date, r.review_row_id;

-- ---------------------------------------------------------------- seller history (as-of)
-- Materialised because the event-stream sort is the expensive part; the serving query reads it.
CREATE MATERIALIZED VIEW __DST__.mv_seller_history AS
WITH pairs AS (
    SELECT DISTINCT order_id, seller_id FROM __SRC__.order_items
), prim AS (
    SELECT order_id, primary_seller_id AS seller_id FROM __DST__.mv_order_items
), events AS (
    SELECT p.seller_id, c.decision_ts AS ts, 0 AS prio, c.order_id,
           0 AS d_orders, 0 AS d_known, 0 AS d_neg, 0 AS d_deliv, 0 AS d_late
    FROM prim p JOIN __DST__.v_core c USING (order_id)
  UNION ALL
    SELECT p.seller_id, c.decision_ts, 1, c.order_id, 1, 0, 0, 0, 0
    FROM pairs p JOIN __DST__.v_core c USING (order_id)
  UNION ALL
    SELECT p.seller_id, l.label_known_ts, 1, l.order_id, 0, 1, l.is_negative::int, 0, 0
    FROM pairs p JOIN __DST__.v_labels l USING (order_id)
  UNION ALL
    SELECT p.seller_id, c.delivered_ts, 1, c.order_id, 0, 0, 0, 1,
           (c.delivered_ts::date > c.estimated_ts::date)::int
    FROM pairs p JOIN __DST__.v_core c USING (order_id)
    WHERE c.delivered_ts IS NOT NULL
), running AS (
    SELECT seller_id, order_id, prio,
           sum(d_orders) OVER w AS s_prior_orders,
           sum(d_known)  OVER w AS s_known_reviews,
           sum(d_neg)    OVER w AS s_known_neg,
           sum(d_deliv)  OVER w AS s_known_deliveries,
           sum(d_late)   OVER w AS s_known_late
    FROM events
    WINDOW w AS (PARTITION BY seller_id ORDER BY ts, prio ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT order_id, s_prior_orders, s_known_reviews, s_known_neg, s_known_deliveries, s_known_late
FROM running WHERE prio = 0;
CREATE UNIQUE INDEX ON __DST__.mv_seller_history (order_id);

-- ---------------------------------------------------------------- customer history (as-of)
CREATE MATERIALIZED VIEW __DST__.mv_customer_history AS
WITH cust AS (
    SELECT c.order_id, u.customer_unique_id
    FROM __DST__.v_core c JOIN __SRC__.customers u USING (customer_id)
), events AS (
    SELECT k.customer_unique_id, c.decision_ts AS ts, 0 AS prio, c.order_id, 0 AS d_orders, 0 AS d_known, 0 AS d_neg
    FROM cust k JOIN __DST__.v_core c USING (order_id)
  UNION ALL
    SELECT k.customer_unique_id, c.decision_ts, 1, c.order_id, 1, 0, 0
    FROM cust k JOIN __DST__.v_core c USING (order_id)
  UNION ALL
    SELECT k.customer_unique_id, l.label_known_ts, 1, l.order_id, 0, 1, l.is_negative::int
    FROM cust k JOIN __DST__.v_labels l USING (order_id)
), running AS (
    SELECT order_id, prio,
           sum(d_orders) OVER w AS c_prior_orders,
           sum(d_known)  OVER w AS c_known_reviews,
           sum(d_neg)    OVER w AS c_known_neg
    FROM events
    WINDOW w AS (PARTITION BY customer_unique_id ORDER BY ts, prio ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT order_id, c_prior_orders, c_known_reviews, c_known_neg FROM running WHERE prio = 0;
CREATE UNIQUE INDEX ON __DST__.mv_customer_history (order_id);

-- ---------------------------------------------------------------- row-local features (materialised)
-- Everything here is a function of ONE order and static dimension tables, so it can be computed
-- once per refresh. All inputs exist at approval time (see the temporal audit).
CREATE MATERIALIZED VIEW __DST__.mv_order_static AS
SELECT
    c.order_id,
    c.decision_ts,
    c.decision_ts_imputed,
    extract(hour FROM c.purchase_ts)::int                                   AS purchase_hour,
    extract(isodow FROM c.purchase_ts)::int                                 AS purchase_dow,
    extract(epoch FROM (c.decision_ts - c.purchase_ts)) / 3600.0            AS approval_lag_hours,
    (c.estimated_ts::date - c.purchase_ts::date)                            AS promised_days_from_purchase,
    (c.estimated_ts::date - c.decision_ts::date)                            AS promised_days_from_approval,
    extract(epoch FROM (i.first_ship_limit_ts - c.decision_ts)) / 86400.0   AS ship_limit_slack_days,
    i.n_items, i.n_sellers, i.n_products, i.total_price, i.total_freight,
    i.total_freight / NULLIF(i.total_price + i.total_freight, 0)            AS freight_ratio,
    cu.customer_state,
    s.seller_state,
    COALESCE(t.product_category_name_english, p.product_category_name)     AS product_category,
    p.product_weight_g,
    p.product_length_cm * p.product_height_cm * p.product_width_cm          AS product_volume_cm3,
    p.product_photos_qty,
    p.product_description_lenght                                            AS product_desc_len,
    p.product_name_lenght                                                   AS product_name_len,
    pay.payment_type_primary, pay.payment_installments_max, pay.n_payments, pay.has_voucher,
    pay.total_payment_value,
    CASE WHEN gc.lat IS NULL OR gs.lat IS NULL THEN NULL ELSE
      2 * 6371.0 * asin(sqrt(LEAST(1.0,
          power(sin(radians(gs.lat - gc.lat) / 2), 2)
        + cos(radians(gc.lat)) * cos(radians(gs.lat)) * power(sin(radians(gs.lng - gc.lng) / 2), 2))))
    END                                                                     AS distance_km,
    (cu.customer_state = s.seller_state)                                    AS same_state,
    -- external signal: public holidays inside the promised delivery window
    GREATEST(he.cum - hb.cum, 0)                                            AS n_holidays_in_promise_window
FROM __DST__.v_core c
LEFT JOIN __SRC__.customers cu       ON cu.customer_id = c.customer_id
LEFT JOIN __DST__.mv_order_items i   ON i.order_id = c.order_id
LEFT JOIN __SRC__.products p         ON p.product_id = i.primary_product_id
LEFT JOIN __SRC__.product_category_translation t ON t.product_category_name = p.product_category_name
LEFT JOIN __SRC__.sellers s          ON s.seller_id = i.primary_seller_id
LEFT JOIN __DST__.mv_order_payments pay ON pay.order_id = c.order_id
LEFT JOIN __SRC__.geo_zip gc         ON gc.zip_prefix = cu.customer_zip_code_prefix
LEFT JOIN __SRC__.geo_zip gs         ON gs.zip_prefix = s.seller_zip_code_prefix
LEFT JOIN __DST__.mv_holiday_cum he  ON he.day = c.estimated_ts::date
LEFT JOIN __DST__.mv_holiday_cum hb  ON hb.day = c.decision_ts::date - 1;
CREATE UNIQUE INDEX ON __DST__.mv_order_static (order_id);

-- ---------------------------------------------------------------- the serving query
-- One row per order: row-local features + as-of history aggregates (with shrinkage).
CREATE VIEW __DST__.v_features AS
SELECT
    st.order_id, st.decision_ts, st.decision_ts_imputed,
    st.purchase_hour, st.purchase_dow, st.approval_lag_hours,
    st.promised_days_from_purchase, st.promised_days_from_approval, st.ship_limit_slack_days,
    st.n_items, st.n_sellers, st.n_products, st.total_price, st.total_freight, st.freight_ratio,
    st.customer_state,
    ch.c_prior_orders,
    (ch.c_known_neg + __PRIOR_STRENGTH__ * __PRIOR_NEG__) / (ch.c_known_reviews + __PRIOR_STRENGTH__)
                                                                            AS c_neg_rate_known,
    st.seller_state,
    sh.s_prior_orders,
    sh.s_known_reviews,
    (sh.s_known_neg + __PRIOR_STRENGTH__ * __PRIOR_NEG__) / (sh.s_known_reviews + __PRIOR_STRENGTH__)
                                                                            AS s_neg_rate_known,
    sh.s_known_deliveries,
    (sh.s_known_late + __PRIOR_STRENGTH__ * __PRIOR_LATE__) / (sh.s_known_deliveries + __PRIOR_STRENGTH__)
                                                                            AS s_late_rate_known,
    st.product_category, st.product_weight_g, st.product_volume_cm3, st.product_photos_qty,
    st.product_desc_len, st.product_name_len,
    st.payment_type_primary, st.payment_installments_max, st.n_payments, st.has_voucher,
    st.total_payment_value,
    st.distance_km, st.same_state,
    st.n_holidays_in_promise_window
FROM __DST__.mv_order_static st
LEFT JOIN __DST__.mv_seller_history sh   ON sh.order_id = st.order_id
LEFT JOIN __DST__.mv_customer_history ch ON ch.order_id = st.order_id;

-- ---------------------------------------------------------------- procedure: refresh + snapshot
CREATE PROCEDURE __DST__.sp_build_feature_snapshot()
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM set_config('work_mem', '256MB', true);
    REFRESH MATERIALIZED VIEW __DST__.mv_holiday_cum;
    REFRESH MATERIALIZED VIEW __DST__.mv_order_items;
    REFRESH MATERIALIZED VIEW __DST__.mv_order_payments;
    REFRESH MATERIALIZED VIEW __DST__.mv_order_static;
    REFRESH MATERIALIZED VIEW __DST__.mv_seller_history;
    REFRESH MATERIALIZED VIEW __DST__.mv_customer_history;
    DROP TABLE IF EXISTS __DST__.feature_snapshot;
    CREATE TABLE __DST__.feature_snapshot AS SELECT * FROM __DST__.v_features;
    ALTER TABLE __DST__.feature_snapshot ADD PRIMARY KEY (order_id);
    ANALYZE __DST__.feature_snapshot;
END $$;
