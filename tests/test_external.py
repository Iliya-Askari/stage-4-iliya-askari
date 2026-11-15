import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from early_warning.ingestion import external


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))

    def json(self):
        return self.payload


class ExternalTests(unittest.TestCase):
    def test_live_payload_is_normalised_and_cached(self):
        payload = [{"date": "2017-09-07", "name": "Independence Day", "types": ["Public"]},
                   {"date": "2017-06-24", "name": "Local feast", "types": ["Optional"]}]
        session = mock.Mock()
        session.get.return_value = FakeResponse(payload)
        with tempfile.TemporaryDirectory() as d:
            df, info = external.load_holidays([2017], mode="auto", cache_dir=Path(d), session=session)
            self.assertTrue((Path(d) / "holidays_BR_2017.json").exists())
        self.assertEqual(len(df), 1)
        self.assertEqual(df.source.iloc[0], "live_api")
        self.assertFalse(info["used_fallback"])

    def test_failure_falls_back_to_bundled_snapshot_and_is_recorded(self):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("blocked")
        with tempfile.TemporaryDirectory() as d, mock.patch("time.sleep"):
            df, info = external.load_holidays([2017], mode="auto", cache_dir=Path(d), session=session)
        self.assertTrue(info["used_fallback"])
        self.assertEqual(set(df.source), {"bundled_fallback"})
        self.assertEqual(len(df), 10)

    def test_live_mode_raises_instead_of_hiding_the_failure(self):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("blocked")
        with tempfile.TemporaryDirectory() as d, mock.patch("time.sleep"):
            with self.assertRaises(external.ExternalSourceError):
                external.load_holidays([2017], mode="live", cache_dir=Path(d), session=session)

    def test_circuit_breaker_stops_calling_api_after_first_failed_year(self):
        session = mock.Mock()
        session.get.side_effect = requests.ConnectionError("blocked")
        with tempfile.TemporaryDirectory() as d, mock.patch("time.sleep"):
            external.load_holidays([2016, 2017, 2018], mode="auto", cache_dir=Path(d), session=session)
        self.assertEqual(session.get.call_count, 3)   # 3 retries for the first year only

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            external.load_holidays([2017], mode="magic")


if __name__ == "__main__":
    unittest.main()
