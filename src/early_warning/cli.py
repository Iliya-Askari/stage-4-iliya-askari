"""Command line: `python -m early_warning run-all | predict`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .db import DatabaseUnavailable, make_engine
from .ingestion.loader import SchemaError
from .predict import PredictionInputError, predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="early_warning")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all", help="empty database -> follow-up list, one command")
    run.add_argument("--raw-dir", type=Path, default=config.DATA_RAW)
    run.add_argument("--external-mode", choices=("auto", "live", "cache", "offline"), default="auto")
    pred = sub.add_parser("predict", help="rank new orders (orders CSV or a directory of Olist CSVs)")
    pred.add_argument("--input", type=Path, required=True)
    pred.add_argument("--output", type=Path, default=config.DATA_CURATED / "predictions.csv")
    pred.add_argument("--capacity", type=float, default=config.CAPACITY)
    pred.add_argument("--external-mode", choices=("auto", "live", "cache", "offline"), default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "run-all":
            from .pipeline import run_all
            res = run_all(args.raw_dir, args.external_mode)
            op = res["operating_point"]
            print(f"done in {res['total_seconds']:.0f}s | model={res['model']} | "
                  f"capture@10%={op['capture_rate']:.3f} | roc_auc={op['roc_auc']:.3f}")
            return 0
        ranked = predict(args.input, make_engine(), capacity=args.capacity, external_mode=args.external_mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output, index=False)
        print(f"wrote {len(ranked)} ranked orders ({int(ranked['follow_up'].sum())} flagged) to {args.output.name}")
        return 0
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return 2
    except (PredictionInputError, FileNotFoundError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 3
    except DatabaseUnavailable as exc:
        print(f"DATABASE ERROR: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
