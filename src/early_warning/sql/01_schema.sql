-- Typed source tables. Schema name is injected by the loader (token __SRC__).
DROP SCHEMA IF EXISTS __SRC__ CASCADE;
CREATE SCHEMA __SRC__;

CREATE TABLE __SRC__.product_category_translation (
    product_category_name         text PRIMARY KEY,
    product_category_name_english text NOT NULL
);
CREATE TABLE __SRC__.customers (
    customer_id              text PRIMARY KEY,
    customer_unique_id       text NOT NULL,
    customer_zip_code_prefix char(5) NOT NULL,
    customer_city            text,
    customer_state           char(2) NOT NULL
);
CREATE TABLE __SRC__.sellers (
    seller_id              text PRIMARY KEY,
    seller_zip_code_prefix char(5) NOT NULL,
    seller_city            text,
    seller_state           char(2) NOT NULL
);
CREATE TABLE __SRC__.products (
    product_id                 text PRIMARY KEY,
    product_category_name      text,
    product_name_lenght        double precision,
    product_description_lenght double precision,
    product_photos_qty         double precision,
    product_weight_g           double precision,
    product_length_cm          double precision,
    product_height_cm          double precision,
    product_width_cm           double precision
);
CREATE TABLE __SRC__.orders (
    order_id                      text PRIMARY KEY,
    customer_id                   text NOT NULL REFERENCES __SRC__.customers,
    order_status                  text NOT NULL,
    order_purchase_timestamp      timestamp NOT NULL,
    order_approved_at             timestamp,
    order_delivered_carrier_date  timestamp,
    order_delivered_customer_date timestamp,
    order_estimated_delivery_date timestamp NOT NULL
);
CREATE TABLE __SRC__.order_items (
    order_id            text NOT NULL REFERENCES __SRC__.orders,
    order_item_id       integer NOT NULL,
    product_id          text NOT NULL REFERENCES __SRC__.products,
    seller_id           text NOT NULL REFERENCES __SRC__.sellers,
    shipping_limit_date timestamp,
    price               double precision NOT NULL,
    freight_value       double precision NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);
CREATE TABLE __SRC__.order_payments (
    order_id             text NOT NULL REFERENCES __SRC__.orders,
    payment_sequential   integer NOT NULL,
    payment_type         text NOT NULL,
    payment_installments integer NOT NULL,
    payment_value        double precision NOT NULL,
    PRIMARY KEY (order_id, payment_sequential)
);
CREATE TABLE __SRC__.order_reviews (
    review_row_id           bigserial PRIMARY KEY,
    review_id               text NOT NULL,
    order_id                text NOT NULL REFERENCES __SRC__.orders,
    review_score            integer NOT NULL,
    review_comment_title    text,
    review_comment_message  text,
    review_creation_date    timestamp NOT NULL,
    review_answer_timestamp timestamp NOT NULL
);
CREATE TABLE __SRC__.geolocation (
    geolocation_zip_code_prefix char(5) NOT NULL,
    geolocation_lat             double precision NOT NULL,
    geolocation_lng             double precision NOT NULL,
    geolocation_city            text,
    geolocation_state           text
);
-- Zip centroid table: filled by the loader from geolocation (only points inside Brazil)
CREATE TABLE __SRC__.geo_zip (
    zip_prefix char(5) PRIMARY KEY,
    lat        double precision NOT NULL,
    lng        double precision NOT NULL
);
