"""Temporal-leakage guards. If somebody adds a post-approval column to the features, these go red."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import text

from early_warning import audit, config, features as F
from early_warning.ingestion.pipeline import ingest_all
from tests.helpers import DatabaseTestCase, tiny_world, write_world

FORBIDDEN_IN_STATIC_OBJECTS = ("order_status", "order_delivered", "order_reviews", "v_labels", "leaky",
                               "review_score", "review_comment")


class AntiLeakTests(DatabaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.world = tiny_world()
        write_world(cls.world, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        super().tearDownClass()

    def rebuild(self) -> pd.DataFrame:
        with mock.patch.object(config, "QUARANTINE_DIR", Path(self.tmp.name) / "q"):
            ingest_all(Path(self.tmp.name), self.engine, external_mode="offline")
        return self._snapshot()

    def _snapshot(self) -> pd.DataFrame:
        F.build_feature_layer(self.engine)
        return F.read_snapshot(self.engine).sort_values("order_id").reset_index(drop=True)

    def execute(self, sql: str, **params):
        with self.engine.begin() as conn:
            conn.execute(text(sql), params)

    # ---- structural guards -------------------------------------------------------------
    def test_feature_view_columns_equal_the_audit_registry_exactly(self):
        self.rebuild()
        cols = F.feature_columns(self.engine)
        audit.assert_columns_audited(cols)
        self.assertEqual(set(cols) - set(audit.META_COLUMNS), set(audit.FEATURE_NAMES))

    def test_tripwire_an_unaudited_column_makes_the_guard_fail(self):
        self.rebuild()
        self.execute("CREATE VIEW feat.v_tampered AS SELECT f.*, o.order_status FROM feat.v_features f "
                     "JOIN olist.orders o USING (order_id)")
        cols = F.feature_columns(self.engine, view="v_tampered")
        with self.assertRaises(AssertionError) as ctx:
            audit.assert_columns_audited(cols)
        self.assertIn("order_status", str(ctx.exception))

    def test_serving_and_static_objects_never_read_post_approval_sources(self):
        self.rebuild()
        with self.engine.connect() as conn:
            for obj in ("feat.v_features", "feat.mv_order_static"):
                definition = conn.execute(text("SELECT pg_get_viewdef(CAST(:o AS regclass))"), {"o": obj}).scalar().lower()
                for token in FORBIDDEN_IN_STATIC_OBJECTS:
                    self.assertNotIn(token, definition, f"{obj} references {token}")

    def test_feature_sql_file_does_not_touch_the_leaky_schema(self):
        sql = (Path(config.PROJECT_ROOT) / "src/early_warning/sql/04_features.sql").read_text().lower()
        self.assertNotIn("leaky", sql)

    # ---- behavioural guards (perturbation) -----------------------------------------------
    def test_features_are_invariant_to_columns_that_exist_only_after_approval(self):
        base = self.rebuild()
        self.execute("UPDATE olist.orders SET order_status = 'canceled', order_delivered_carrier_date = NULL")
        self.execute("UPDATE olist.order_reviews SET review_comment_message = 'x', "
                     "review_creation_date = review_creation_date + interval '90 days'")
        assert_frame_equal(base, self._snapshot())

    def test_features_do_not_change_when_the_future_changes(self):
        base = self.rebuild()
        cut = base.decision_ts.sort_values().iloc[20]
        self.execute("UPDATE olist.order_reviews SET review_score = 6 - review_score "
                     "WHERE review_answer_timestamp >= :c", c=cut)
        self.execute("UPDATE olist.orders SET order_delivered_customer_date = order_delivered_customer_date "
                     "+ interval '30 days' WHERE order_delivered_customer_date >= :c", c=cut)
        after = self._snapshot()
        before_cut = base.decision_ts <= cut
        assert_frame_equal(base[before_cut].reset_index(drop=True), after[before_cut].reset_index(drop=True))

    def test_sensitivity_changing_the_past_does_change_later_features(self):
        base = self.rebuild()
        cut = base.decision_ts.sort_values().iloc[20]
        self.execute("UPDATE olist.order_reviews SET review_score = 6 - review_score "
                     "WHERE review_answer_timestamp < :c", c=cut)
        after = self._snapshot()
        later = base.decision_ts > cut
        self.assertFalse(base.loc[later, "s_neg_rate_known"].equals(after.loc[later, "s_neg_rate_known"]))

    def test_naive_all_time_seller_rate_is_leaky_and_differs_from_the_as_of_feature(self):
        base = self.rebuild().merge(self.world["order_items"][["order_id", "seller_id"]], on="order_id")
        with self.engine.connect() as conn:
            naive = pd.read_sql(text(
                "SELECT i.order_id, avg((r.review_score <= 2)::int) OVER (PARTITION BY i.seller_id) AS naive "
                "FROM olist.order_items i JOIN olist.order_reviews r USING (order_id)"), conn)
        m = base.merge(naive, on="order_id").sort_values("decision_ts")
        first_bad = m[m.seller_id == "S_BAD"].iloc[0]
        self.assertEqual(first_bad.naive, 1.0)                    # knows its own (future) bad review
        self.assertLess(first_bad.s_neg_rate_known, 0.5)          # the honest feature does not
        self.assertFalse(m.naive.round(4).equals(m.s_neg_rate_known.round(4)))


if __name__ == "__main__":
    unittest.main()
