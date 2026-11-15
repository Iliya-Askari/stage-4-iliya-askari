"""Shared test fixtures: a tiny synthetic Olist world and a throw-away database."""
from __future__ import annotations

import os
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import make_url

from early_warning import config
from early_warning.db import drop_database, ensure_database, make_engine

FMT = "%Y-%m-%d %H:%M:%S"


def temp_database_url() -> str:
    base = make_url(config.database_url())
    return base.set(database=f"ew_test_{uuid.uuid4().hex[:10]}").render_as_string(hide_password=False)


class DatabaseTestCase(unittest.TestCase):
    """Each test class gets its own empty database and drops it afterwards."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("EW_SKIP_DB_TESTS") == "1":
            raise unittest.SkipTest("EW_SKIP_DB_TESTS=1")
        cls.url = temp_database_url()
        os.environ["_EW_TEST_URL"] = cls.url
        ensure_database(cls.url)
        cls.engine = make_engine(cls.url)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        drop_database(cls.url)


def t(base: str, **delta) -> str:
    return (datetime.strptime(base, FMT) + timedelta(**delta)).strftime(FMT)


def tiny_world(n_orders: int = 40, seed_start: str = "2017-01-01 10:00:00") -> dict[str, pd.DataFrame]:
    """Deterministic mini dataset: 2 sellers, 40 orders, a review for most of them.

    Seller S_BAD gets only 1-star reviews, S_GOOD only 5-star, so history features are predictable.
    """
    orders, customers, items, pays, reviews = [], [], [], [], []
    for i in range(n_orders):
        oid, cid = f"o{i:03d}", f"c{i:03d}"
        seller = "S_BAD" if i % 2 == 0 else "S_GOOD"
        purchase = t(seed_start, days=i * 3)
        approved = t(purchase, minutes=30)
        delivered = t(purchase, days=12)
        estimated = t(purchase, days=10)
        orders.append({
            "order_id": oid, "customer_id": cid, "order_status": "delivered",
            "order_purchase_timestamp": purchase, "order_approved_at": approved,
            "order_delivered_carrier_date": t(purchase, days=3), "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated[:10] + " 00:00:00"})
        customers.append({"customer_id": cid, "customer_unique_id": f"u{i:03d}",
                          "customer_zip_code_prefix": "01000", "customer_city": "sao paulo",
                          "customer_state": "SP"})
        items.append({"order_id": oid, "order_item_id": 1, "product_id": "p1", "seller_id": seller,
                      "shipping_limit_date": t(purchase, days=4), "price": 50.0 + i, "freight_value": 10.0})
        pays.append({"order_id": oid, "payment_sequential": 1, "payment_type": "credit_card",
                     "payment_installments": 2, "payment_value": 60.0 + i})
        if i % 7 != 6:  # a few orders never get a review
            score = 1 if seller == "S_BAD" else 5
            answered = t(delivered, days=3)
            reviews.append({"review_id": f"r{i:03d}", "order_id": oid, "review_score": score,
                            "review_comment_title": "", "review_comment_message": "",
                            "review_creation_date": t(delivered, days=1), "review_answer_timestamp": answered})
    return {
        "orders": pd.DataFrame(orders), "customers": pd.DataFrame(customers),
        "order_items": pd.DataFrame(items), "order_payments": pd.DataFrame(pays),
        "order_reviews": pd.DataFrame(reviews),
        "products": pd.DataFrame([{"product_id": "p1", "product_category_name": "toys",
                                   "product_name_lenght": 20, "product_description_lenght": 100,
                                   "product_photos_qty": 2, "product_weight_g": 500, "product_length_cm": 10,
                                   "product_height_cm": 10, "product_width_cm": 10}]),
        "sellers": pd.DataFrame([{"seller_id": s, "seller_zip_code_prefix": "01000", "seller_city": "sao paulo",
                                  "seller_state": "SP"} for s in ("S_BAD", "S_GOOD")]),
        "geolocation": pd.DataFrame([{"geolocation_zip_code_prefix": "01000", "geolocation_lat": -23.5,
                                      "geolocation_lng": -46.6, "geolocation_city": "sao paulo",
                                      "geolocation_state": "SP"}]),
        "product_category_translation": pd.DataFrame([{"product_category_name": "toys",
                                                       "product_category_name_english": "toys"}]),
    }


def write_world(world: dict[str, pd.DataFrame], directory: Path) -> Path:
    from early_warning.ingestion.contract import CONTRACTS
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in world.items():
        df.to_csv(directory / CONTRACTS[name].filename, index=False)
    return directory
