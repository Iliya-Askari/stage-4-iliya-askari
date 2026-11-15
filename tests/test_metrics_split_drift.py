import unittest

import numpy as np
import pandas as pd

from early_warning.modeling import evaluate as E
from early_warning.modeling.dataset import inner_split, time_split
from early_warning.modeling.drift import psi_categorical, psi_numeric


class MetricTests(unittest.TestCase):
    def test_quota_flags_exactly_ceil_of_capacity(self):
        mask = E.top_k_mask(np.arange(101, dtype=float), 0.10)
        self.assertEqual(mask.sum(), 11)
        self.assertTrue(mask[-11:].all())

    def test_capture_rate_of_a_perfect_ranking_is_capped_by_capacity(self):
        y = np.array([1] * 10 + [0] * 90)
        self.assertAlmostEqual(E.capture_rate(y, y.astype(float), 0.10), 1.0)
        self.assertAlmostEqual(E.capture_rate(y, y.astype(float), 0.05), 0.5)

    def test_random_scores_capture_about_the_capacity(self):
        rng = np.random.default_rng(0)
        y = (rng.random(20000) < 0.15).astype(int)
        self.assertAlmostEqual(E.capture_rate(y, rng.random(20000), 0.10), 0.10, delta=0.02)

    def test_confusion_matrix_adds_up_and_matches_capture(self):
        rng = np.random.default_rng(1)
        y = (rng.random(1000) < 0.2).astype(int)
        s = y + rng.normal(0, 1, 1000)
        c = E.confusion_at(y, s, 0.10)
        self.assertEqual(c["tp"] + c["fp"] + c["fn"] + c["tn"], 1000)
        self.assertEqual(c["tp"] + c["fn"], int(y.sum()))
        self.assertAlmostEqual(c["recall"], E.capture_rate(y, s, 0.10))

    def test_capacity_table_is_monotone(self):
        rng = np.random.default_rng(2)
        y = (rng.random(5000) < 0.15).astype(int)
        t = E.capacity_table(y, y + rng.normal(0, 1.5, 5000))
        self.assertEqual(list(t.capacity), ["5%", "10%", "20%"])
        self.assertTrue(t.share_of_negatives_caught.is_monotonic_increasing)

    def test_monthly_table_applies_quota_inside_each_month(self):
        rng = np.random.default_rng(3)
        n = 3000
        df = pd.DataFrame({"decision_ts": pd.date_range("2018-01-01", periods=n, freq="35min"),
                           "is_negative": (rng.random(n) < 0.2).astype(int)})
        t = E.monthly_table(df, rng.random(n), min_rows=100)
        self.assertGreaterEqual(len(t), 2)
        self.assertTrue(((t.capture_rate >= 0) & (t.capture_rate <= 1)).all())


def frame(n=4000):
    ts = pd.date_range("2017-06-01", periods=n, freq="6h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"order_id": [f"o{i}" for i in range(n)], "decision_ts": ts,
                         "decision_ts_imputed": False, "is_negative": (rng.random(n) < 0.15).astype(float),
                         "label_known_ts": ts + pd.Timedelta(days=15)})


class SplitTests(unittest.TestCase):
    def test_split_is_temporal_with_embargo_and_no_future_labels(self):
        df = frame()
        sp = time_split(df, boundary="2018-01-01", embargo_days=30)
        self.assertLess(sp.train.decision_ts.max(), pd.Timestamp("2017-12-02"))
        self.assertGreaterEqual(sp.test.decision_ts.min(), pd.Timestamp("2018-01-01"))
        self.assertTrue((sp.train.label_known_ts < pd.Timestamp("2018-01-01")).all())
        self.assertEqual(set(sp.train.order_id) & set(sp.test.order_id), set())
        self.assertGreater(sp.summary["embargoed_rows"], 0)

    def test_unlabelled_and_never_approved_orders_are_excluded(self):
        df = frame()
        df.loc[:50, "is_negative"] = np.nan
        df.loc[100:120, "decision_ts_imputed"] = True
        sp = time_split(df, boundary="2018-01-01", embargo_days=30)
        used = set(sp.train.order_id) | set(sp.test.order_id)
        self.assertEqual(used & set(df.order_id[:51]), set())
        self.assertEqual(used & set(df.order_id[100:121]), set())
        self.assertEqual(sp.summary["excluded_never_approved"], 21)

    def test_inner_validation_comes_only_from_the_training_period(self):
        sp = time_split(frame(), boundary="2018-01-01", embargo_days=30)
        inner, val = inner_split(sp.train, validation_days=40, embargo_days=30)
        self.assertLess(inner.decision_ts.max(), val.decision_ts.min())
        self.assertLessEqual(val.decision_ts.max(), sp.train.decision_ts.max())


class DriftTests(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        x = pd.Series(np.random.default_rng(0).normal(size=5000))
        self.assertLess(psi_numeric(x.iloc[:2500], x.iloc[2500:]), 0.05)

    def test_shifted_distribution_is_flagged_major(self):
        rng = np.random.default_rng(0)
        self.assertGreater(psi_numeric(pd.Series(rng.normal(0, 1, 4000)), pd.Series(rng.normal(2, 1, 4000))), 0.25)
        self.assertGreater(psi_categorical(pd.Series(["a"] * 900 + ["b"] * 100), pd.Series(["a"] * 100 + ["b"] * 900)), 0.25)


if __name__ == "__main__":
    unittest.main()
