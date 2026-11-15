import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from early_warning import config, features as F
from early_warning.ingestion.pipeline import ingest_all
from tests.helpers import DatabaseTestCase, tiny_world, write_world


class FeatureLayerTests(DatabaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.world = tiny_world()
        write_world(cls.world, Path(cls.tmp.name))
        with mock.patch.object(config, "QUARANTINE_DIR", Path(cls.tmp.name) / "q"):
            ingest_all(Path(cls.tmp.name), cls.engine, external_mode="offline")
        F.build_feature_layer(cls.engine)
        cls.feats = F.read_snapshot(cls.engine).sort_values("decision_ts").reset_index(drop=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        super().tearDownClass()

    def test_one_row_per_order(self):
        self.assertEqual(len(self.feats), len(self.world["orders"]))
        self.assertTrue(self.feats.order_id.is_unique)

    def test_first_order_of_each_seller_has_no_history(self):
        first = self.feats.iloc[0]
        self.assertEqual(first.s_prior_orders, 0)
        self.assertEqual(first.s_known_reviews, 0)

    def test_history_counts_only_reviews_answered_before_the_decision(self):
        rev = self.world["order_reviews"].merge(self.world["order_items"][["order_id", "seller_id"]], on="order_id")
        rev["ans"] = pd.to_datetime(rev.review_answer_timestamp)
        merged = self.feats.merge(self.world["order_items"][["order_id", "seller_id"]], on="order_id")
        for _, row in merged.iterrows():
            expected = ((rev.seller_id == row.seller_id) & (rev.ans < row.decision_ts)).sum()
            self.assertEqual(row.s_known_reviews, expected, row.order_id)

    def test_reputation_follows_the_seller_not_the_order_itself(self):
        merged = self.feats.merge(self.world["order_items"][["order_id", "seller_id"]], on="order_id")
        late = merged[merged.s_known_reviews >= 5]
        bad = late[late.seller_id == "S_BAD"].s_neg_rate_known.min()
        good = late[late.seller_id == "S_GOOD"].s_neg_rate_known.max()
        self.assertGreater(bad, good)

    def test_serving_query_is_fast_and_the_plan_is_available(self):
        timing = F.time_feature_query(self.engine, repeats=2)
        self.assertLess(timing["server_ms_median"], config.FEATURE_QUERY_BUDGET_MS)
        self.assertIn("Execution Time", F.explain_feature_query(self.engine))


if __name__ == "__main__":
    unittest.main()
