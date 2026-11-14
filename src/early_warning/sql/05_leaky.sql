-- DELIBERATELY LEAKY columns. Used only to train the "cheat model" that measures the cost of leakage.
-- Nothing in the feature layer may reference this schema (a test enforces it).
DROP SCHEMA IF EXISTS __LEAKY__ CASCADE;
CREATE SCHEMA __LEAKY__;
CREATE VIEW __LEAKY__.v_leaky_columns AS
SELECT o.order_id,
       -- (1) post-approval operational facts
       o.order_status,
       (o.order_delivered_customer_date::date - o.order_estimated_delivery_date::date)     AS delivery_delay_days,
       extract(epoch FROM (o.order_delivered_carrier_date - o.order_purchase_timestamp)) / 86400.0
                                                                                           AS carrier_handoff_days,
       extract(epoch FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) / 86400.0
                                                                                           AS days_to_deliver,
       -- (2) naive all-time seller aggregate: uses future reviews AND the order's own review
       sa.seller_neg_rate_all_time,
       -- (3) review-derived text signal
       COALESCE(length(r.review_comment_message), 0)                                       AS review_comment_len
FROM __SRC__.orders o
LEFT JOIN (
    SELECT DISTINCT ON (order_id) order_id, review_comment_message
    FROM __SRC__.order_reviews ORDER BY order_id, review_answer_timestamp, review_row_id
) r ON r.order_id = o.order_id
LEFT JOIN (
    SELECT p.order_id, sa.seller_neg_rate_all_time
    FROM (SELECT DISTINCT ON (order_id) order_id, seller_id FROM __SRC__.order_items ORDER BY order_id, order_item_id) p
    JOIN (SELECT i2.seller_id, avg((rv.review_score <= __NEG_MAX__)::int) AS seller_neg_rate_all_time
          FROM __SRC__.order_reviews rv JOIN __SRC__.order_items i2 USING (order_id)
          GROUP BY i2.seller_id) sa USING (seller_id)
) sa ON sa.order_id = o.order_id;
