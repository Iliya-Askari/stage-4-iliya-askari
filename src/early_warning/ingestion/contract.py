"""Schema contract for every source table.

The contract is the single place that says what a valid row looks like. A row that
violates it is never loaded: it is written to the quarantine with a machine readable reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ORDER_STATUSES = (
    "created", "approved", "invoiced", "processing", "shipped",
    "delivered", "unavailable", "canceled",
)
PAYMENT_TYPES = ("credit_card", "boleto", "voucher", "debit_card", "not_defined")


@dataclass(frozen=True)
class Col:
    name: str
    kind: str                       # str | int | float | ts | zip
    nullable: bool = True
    min: float | None = None
    max: float | None = None
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TableContract:
    name: str
    filename: str
    columns: tuple[Col, ...]
    key: tuple[str, ...] = ()
    required: bool = True
    dedupe_exact: bool = False
    aliases: dict[str, str] = field(default_factory=dict)   # alternative header -> canonical

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


CONTRACTS: dict[str, TableContract] = {
    "orders": TableContract(
        "orders", "olist_orders_dataset.csv",
        (
            Col("order_id", "str", False),
            Col("customer_id", "str", False),
            Col("order_status", "str", False, allowed=ORDER_STATUSES),
            Col("order_purchase_timestamp", "ts", False),
            Col("order_approved_at", "ts"),
            Col("order_delivered_carrier_date", "ts"),
            Col("order_delivered_customer_date", "ts"),
            Col("order_estimated_delivery_date", "ts", False),
        ),
        key=("order_id",),
    ),
    "customers": TableContract(
        "customers", "olist_customers_dataset.csv",
        (
            Col("customer_id", "str", False),
            Col("customer_unique_id", "str", False),
            Col("customer_zip_code_prefix", "zip", False),
            Col("customer_city", "str"),
            Col("customer_state", "str", False),
        ),
        key=("customer_id",),
    ),
    "order_items": TableContract(
        "order_items", "olist_order_items_dataset.csv",
        (
            Col("order_id", "str", False),
            Col("order_item_id", "int", False, min=1),
            Col("product_id", "str", False),
            Col("seller_id", "str", False),
            Col("shipping_limit_date", "ts"),
            Col("price", "float", False, min=0),
            Col("freight_value", "float", False, min=0),
        ),
        key=("order_id", "order_item_id"),
    ),
    "order_payments": TableContract(
        "order_payments", "olist_order_payments_dataset.csv",
        (
            Col("order_id", "str", False),
            Col("payment_sequential", "int", False, min=1),
            Col("payment_type", "str", False, allowed=PAYMENT_TYPES),
            Col("payment_installments", "int", False, min=0),
            Col("payment_value", "float", False, min=0),
        ),
        key=("order_id", "payment_sequential"),
    ),
    "order_reviews": TableContract(
        "order_reviews", "olist_order_reviews_dataset.csv",
        (
            Col("review_id", "str", False),
            Col("order_id", "str", False),
            Col("review_score", "int", False, min=1, max=5),
            Col("review_comment_title", "str"),
            Col("review_comment_message", "str"),
            Col("review_creation_date", "ts", False),
            Col("review_answer_timestamp", "ts", False),
        ),
    ),
    "products": TableContract(
        "products", "olist_products_dataset.csv",
        (
            Col("product_id", "str", False),
            Col("product_category_name", "str"),
            Col("product_name_lenght", "float", min=0),
            Col("product_description_lenght", "float", min=0),
            Col("product_photos_qty", "float", min=0),
            Col("product_weight_g", "float", min=0),
            Col("product_length_cm", "float", min=0),
            Col("product_height_cm", "float", min=0),
            Col("product_width_cm", "float", min=0),
        ),
        key=("product_id",),
        aliases={"product_name_length": "product_name_lenght",
                 "product_description_length": "product_description_lenght"},
    ),
    "sellers": TableContract(
        "sellers", "olist_sellers_dataset.csv",
        (
            Col("seller_id", "str", False),
            Col("seller_zip_code_prefix", "zip", False),
            Col("seller_city", "str"),
            Col("seller_state", "str", False),
        ),
        key=("seller_id",),
    ),
    "geolocation": TableContract(
        "geolocation", "olist_geolocation_dataset.csv",
        (
            Col("geolocation_zip_code_prefix", "zip", False),
            Col("geolocation_lat", "float", False, min=-90, max=90),
            Col("geolocation_lng", "float", False, min=-180, max=180),
            Col("geolocation_city", "str"),
            Col("geolocation_state", "str"),
        ),
        required=False,
    ),
    "product_category_translation": TableContract(
        "product_category_translation", "product_category_name_translation.csv",
        (
            Col("product_category_name", "str", False),
            Col("product_category_name_english", "str", False),
        ),
        key=("product_category_name",),
        required=False,
    ),
}

# Foreign keys checked at ingestion time: (child table, child column, parent table, parent column)
FOREIGN_KEYS = (
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("order_payments", "order_id", "orders", "order_id"),
    ("order_reviews", "order_id", "orders", "order_id"),
)

# Load order respects the foreign keys
LOAD_ORDER = (
    "product_category_translation", "customers", "sellers", "products", "orders",
    "order_items", "order_payments", "order_reviews", "geolocation",
)
