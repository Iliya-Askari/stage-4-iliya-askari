"""One command, empty machine -> follow-up list.

    python run_all.py                      # venv + pinned install + full chain
    python run_all.py --with-tests         # ... and the unit tests
    python run_all.py --with-notebooks     # ... and execute both notebooks (outputs are saved)

Needs: Python >= 3.11, a running PostgreSQL, DATABASE_URL (default
postgresql+psycopg2://postgres:postgres@localhost:5432/early_warning) and the nine Olist CSV files
in data/raw/. Only the standard library is used here, so it runs before anything is installed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
ART = ROOT / "data" / "curated" / "artifacts"
BUDGET_MINUTES = 45


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("--with-notebooks", action="store_true")
    parser.add_argument("--external-mode", default="auto", choices=("auto", "live", "cache", "offline"))
    parser.add_argument("--fresh-venv", action="store_true", help="delete .venv first (true 'empty environment')")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required.", file=sys.stderr)
        return 1
    if not (ROOT / "data" / "raw" / "olist_orders_dataset.csv").exists():
        print("Put the nine Olist CSV files into data/raw/ first (see README).", file=sys.stderr)
        return 1

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if args.fresh_venv and VENV.exists():
        import shutil
        shutil.rmtree(VENV)
    if not venv_python().exists():
        venv.EnvBuilder(with_pip=True).create(VENV)
    run([venv_python(), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", "requirements.txt"])
    run([venv_python(), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-deps", "-e", "."])
    timings["environment_setup"] = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    run([venv_python(), "-m", "early_warning", "run-all", "--external-mode", args.external_mode])
    timings["pipeline_run_all"] = round(time.perf_counter() - t1, 1)

    if args.with_tests:
        t2 = time.perf_counter()
        run([venv_python(), "-m", "unittest", "discover", "-s", "tests", "-t", "."])
        timings["unit_tests"] = round(time.perf_counter() - t2, 1)
    if args.with_notebooks:
        t3 = time.perf_counter()
        run([venv_python(), "tools/execute_notebooks.py"])
        timings["notebooks"] = round(time.perf_counter() - t3, 1)

    core = timings["environment_setup"] + timings["pipeline_run_all"]
    timings["total_to_final_list_seconds"] = round(core, 1)
    timings["budget_seconds"] = BUDGET_MINUTES * 60
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "bootstrap_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(json.dumps(timings, indent=2))
    print(f"empty environment -> final list: {core / 60:.1f} min (budget {BUDGET_MINUTES} min)")
    return 0 if core < BUDGET_MINUTES * 60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
