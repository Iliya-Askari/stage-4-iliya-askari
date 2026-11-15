import unittest

from early_warning import audit
from early_warning.modeling.dataset import CATEGORICAL, NUMERIC


class AuditRegistryTests(unittest.TestCase):
    def test_every_feature_is_declared_available_at_approval(self):
        self.assertTrue(all(f.at_approval for f in audit.FEATURES))

    def test_registry_has_unique_names_and_at_least_ten_features(self):
        names = [f.name for f in audit.FEATURES]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 10)

    def test_every_feature_has_a_hypothesis_and_every_aggregate_a_leak_control(self):
        for f in audit.FEATURES:
            self.assertTrue(f.hypothesis.strip(), f.name)
            if f.aggregate:
                self.assertTrue(f.leak_control.strip(), f.name)

    def test_at_least_three_tempting_columns_are_documented_as_excluded(self):
        self.assertGreaterEqual(len(audit.EXCLUDED), 3)

    def test_model_input_lists_equal_the_registry(self):
        self.assertEqual(set(NUMERIC) | set(CATEGORICAL), set(audit.FEATURE_NAMES))

    def test_undeclared_column_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit.assert_columns_audited(list(audit.FEATURE_NAMES) + ["order_status"])

    def test_features_come_from_several_tables(self):
        text = " ".join(f.sources for f in audit.FEATURES)
        for table in ("orders", "customers", "sellers", "products", "order_payments", "geolocation", "external"):
            self.assertIn(table, text)


if __name__ == "__main__":
    unittest.main()
