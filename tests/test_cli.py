import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from early_warning.cli import main
from tests.helpers import tiny_world


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(list(argv))
        return code, err.getvalue()

    def test_bad_schema_gives_exit_code_2_and_a_readable_message(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "x.csv"
            tiny_world()["orders"].drop(columns=["order_status"]).to_csv(bad, index=False)
            code, err = self.run_cli("predict", "--input", str(bad), "--output", str(Path(d) / "o.csv"))
        self.assertEqual(code, 2)
        self.assertIn("SCHEMA ERROR", err)
        self.assertIn("order_status", err)

    def test_missing_input_gives_exit_code_3(self):
        code, err = self.run_cli("predict", "--input", "does_not_exist.csv")
        self.assertEqual(code, 3)
        self.assertIn("INPUT ERROR", err)

    def test_a_command_is_required(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                main([])


if __name__ == "__main__":
    unittest.main()
