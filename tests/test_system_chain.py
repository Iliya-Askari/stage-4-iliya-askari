"""Whole chain on a tiny synthetic dataset: ingest -> features -> split -> model -> prediction on unseen orders."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from sqlalchemy import text

from early_warning import config, features as F
from early_warning.ingestion.loader import SchemaError
from early_warning.ingestion.pipeline import ingest_all
from early_warning.modeling.dataset import build_modeling_frame, time_split, xy
from early_warning.modeling.models import make_hgb
from early_warning.modeling.train import TrainedModel
from early_warning.predict import PredictionInputError, predict
from tests.helpers import DatabaseTestCase, t, tiny_world, write_world


def fresh_orders(n=10, start="2017-09-01 09:00:00") -> dict[str, pd.DataFrame]:
    base = tiny_world(n_orders=n, seed_start=start)
    base["orders"]["order_id"] = base["orders"]["order_id"].str.replace("o", "new")
    base["orders"]["customer_id"] = base["orders"]["customer_id"].str.replace("c", "newc")
    for key in ("order_items", "order_payments"):
        base[key]["order_id"] = base[key]["order_id"].str.replace("o", "new")
    base["customers"]["customer_id"] = base["customers"]["customer_id"].str.replace("c", "newc")
    base["customers"]["customer_unique_id"] = base["customers"]["customer_unique_id"].str.replace("u", "newu")
    # future information that must be ignored: delivery facts and reviews do not exist yet
    for col in ("order_delivered_carrier_date", "order_delivered_customer_date"):
        base["orders"][col] = None
    base["orders"]["order_status"] = "approved"
    return {k: base[k] for k in ("orders", "customers", "order_items", "order_payments")}


class ChainTests(DatabaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_world(tiny_world(n_orders=60), cls.dir / "raw")
        cls.patch = mock.patch.object(config, "QUARANTINE_DIR", cls.dir / "q")
        cls.patch.start()
        ingest_all(cls.dir / "raw", cls.engine, external_mode="offline")
        F.build_feature_layer(cls.engine)
        frame = build_modeling_frame(cls.engine)
        cls.split = time_split(frame, boundary="2017-04-15", embargo_days=5)
        X, y = xy(cls.split.train)
        pipe = make_hgb(max_leaf_nodes=4, max_iter=30, min_samples_leaf=3).fit(X, y)
        cls.model = TrainedModel("tiny_hgb", pipe, pd.DataFrame(), {"model": "tiny_hgb"})

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()
        super().tearDownClass()

    def test_split_has_both_classes_in_train_and_a_future_test(self):
        self.assertGreater(len(self.split.train), 10)
        self.assertGreater(len(self.split.test), 5)
        self.assertEqual(self.split.train.is_negative.nunique(), 2)
        self.assertLess(self.split.train.decision_ts.max(), self.split.test.decision_ts.min())

    def test_unseen_orders_are_ranked_from_a_directory_of_csvs(self):
        fresh = write_world(fresh_orders(), self.dir / "fresh")
        out = predict(fresh, self.engine, model=self.model, external_mode="offline")
        self.assertEqual(len(out), 10)
        self.assertEqual(list(out["rank"]), list(range(1, 11)))
        self.assertTrue(out["risk_score"].is_monotonic_decreasing)
        self.assertEqual(int(out["follow_up"].sum()), 1)               # ceil(10% of 10)
        self.assertEqual(set(out.order_id), set(fresh_orders()["orders"].order_id))

    def test_orders_from_the_unreliable_seller_rank_higher(self):
        fresh = write_world(fresh_orders(), self.dir / "fresh2")
        out = predict(fresh, self.engine, model=self.model, external_mode="offline")
        items = fresh_orders()["order_items"][["order_id", "seller_id"]]
        m = out.merge(items, on="order_id")
        self.assertGreater(m[m.seller_id == "S_BAD"].risk_score.mean(), m[m.seller_id == "S_GOOD"].risk_score.mean())

    def test_prediction_does_not_modify_production_tables_or_leave_scratch_schemas(self):
        with self.engine.connect() as conn:
            before = conn.execute(text("SELECT count(*) FROM olist.orders")).scalar()
        predict(write_world(fresh_orders(), self.dir / "fresh3"), self.engine, model=self.model,
                external_mode="offline")
        with self.engine.connect() as conn:
            after = conn.execute(text("SELECT count(*) FROM olist.orders")).scalar()
            leftovers = conn.execute(text(
                "SELECT count(*) FROM information_schema.schemata WHERE schema_name IN "
                "('olist_infer','feat_infer','ext_infer')")).scalar()
        self.assertEqual((before, leftovers), (after, 0))

    def test_orders_only_file_is_accepted(self):
        path = self.dir / "orders_only.csv"
        fresh_orders()["orders"].to_csv(path, index=False)
        out = predict(path, self.engine, model=self.model, external_mode="offline")
        self.assertEqual(len(out), 10)

    def test_wrong_schema_is_rejected_with_a_clear_error(self):
        path = self.dir / "bad.csv"
        fresh_orders()["orders"].drop(columns=["customer_id"]).to_csv(path, index=False)
        with self.assertRaises(SchemaError) as ctx:
            predict(path, self.engine, model=self.model, external_mode="offline")
        self.assertIn("customer_id", str(ctx.exception))

    def test_directory_without_orders_and_bad_capacity_are_rejected(self):
        empty = self.dir / "empty"
        empty.mkdir()
        with self.assertRaises(SchemaError):
            predict(empty, self.engine, model=self.model)
        with self.assertRaises(PredictionInputError):
            predict(empty, self.engine, model=self.model, capacity=0)


if __name__ == "__main__":
    unittest.main()
