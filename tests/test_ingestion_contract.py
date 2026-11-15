import tempfile
import unittest
from pathlib import Path

import pandas as pd

from early_warning.ingestion.contract import CONTRACTS
from early_warning.ingestion.loader import (SchemaError, apply_foreign_keys, parse_timestamps, read_raw,
                                            validate)
from early_warning.ingestion.manifest import Manifest
from tests.helpers import tiny_world


def as_raw(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(str).replace({"nan": "", "None": ""})


class ContractTests(unittest.TestCase):
    def test_header_is_case_insensitive_and_bom_tolerant(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "orders.csv"
            df = tiny_world()["orders"].rename(columns=str.upper)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            out = read_raw(path, CONTRACTS["orders"])
        self.assertEqual(list(out.columns), CONTRACTS["orders"].column_names)

    def test_missing_column_raises_clear_schema_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "orders.csv"
            tiny_world()["orders"].drop(columns=["order_status"]).to_csv(path, index=False)
            with self.assertRaises(SchemaError) as ctx:
                read_raw(path, CONTRACTS["orders"])
        self.assertIn("order_status", str(ctx.exception))

    def test_invalid_rows_are_quarantined_with_reason(self):
        reviews = as_raw(tiny_world()["order_reviews"])
        reviews.loc[0, "review_score"] = "9"
        reviews.loc[1, "review_score"] = "abc"
        res = validate(reviews, CONTRACTS["order_reviews"])
        self.assertEqual(len(res.quarantine), 2)
        self.assertIn("above_max_review_score", res.stats["reasons"])
        self.assertIn("invalid_review_score", res.stats["reasons"])
        self.assertEqual(len(res.clean) + len(res.quarantine), len(reviews))

    def test_negative_price_and_duplicate_key_are_quarantined(self):
        items = as_raw(tiny_world()["order_items"])
        items.loc[0, "price"] = "-5"
        items = pd.concat([items, items.iloc[[3]]], ignore_index=True)
        res = validate(items, CONTRACTS["order_items"])
        self.assertEqual(res.stats["reasons"].get("below_min_price"), 1)
        self.assertEqual(res.stats["reasons"].get("duplicate_key"), 1)

    def test_timestamp_layouts_parse_to_the_same_instant(self):
        iso = parse_timestamps(pd.Series(["2017-10-02 10:56:33", "2018-01-05 08:00:00"]))
        dmy = parse_timestamps(pd.Series(["02-10-2017 10:56", "05-01-2018 08:00"]))
        self.assertEqual(iso[0].floor("min"), dmy[0])
        self.assertEqual(iso[1], dmy[1])

    def test_unparseable_timestamp_is_quarantined_not_guessed(self):
        orders = as_raw(tiny_world()["orders"])
        orders.loc[0, "order_purchase_timestamp"] = "not a date"
        res = validate(orders, CONTRACTS["orders"])
        self.assertIn("invalid_order_purchase_timestamp", res.stats["reasons"])

    def test_zip_prefix_is_zero_padded(self):
        customers = as_raw(tiny_world()["customers"])
        customers["customer_zip_code_prefix"] = "1037"
        res = validate(customers, CONTRACTS["customers"])
        self.assertEqual(res.clean["customer_zip_code_prefix"].iloc[0], "01037")

    def test_orphan_child_rows_are_quarantined(self):
        world = tiny_world()
        tables = {k: validate(as_raw(world[k]), CONTRACTS[k]).clean for k in ("customers", "orders", "order_items")}
        tables["order_items"].loc[0, "order_id"] = "ghost"
        clean, bad = apply_foreign_keys(tables)
        self.assertEqual(len(bad["order_items"]), 1)
        self.assertEqual(len(clean["order_items"]), len(tables["order_items"]) - 1)

    def test_manifest_records_stage_timings(self):
        m = Manifest()
        with m.stage("demo"):
            pass
        with tempfile.TemporaryDirectory() as d:
            path = m.write(Path(d) / "m.json")
            self.assertIn("demo", path.read_text())
        self.assertIn("demo", m.data["stage_seconds"])


if __name__ == "__main__":
    unittest.main()
