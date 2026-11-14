"""Run manifest: what was read, what was kept, what was quarantined, how long each stage took."""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..config import SEED


class Manifest:
    def __init__(self) -> None:
        self.data: dict = {
            "package_version": __version__,
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.system(),
            "seed": SEED,
            "tables": {},
            "external_sources": {},
            "anomalies": {},
            "stage_seconds": {},
            "notes": [],
        }

    def add_table(self, name: str, source_file: str, sha256: str, stats: dict, loaded_rows: int) -> None:
        self.data["tables"][name] = {
            "source_file": source_file, "sha256": sha256, **stats, "rows_loaded": int(loaded_rows),
        }

    def add_external(self, name: str, info: dict) -> None:
        self.data["external_sources"][name] = info

    def note(self, text: str) -> None:
        self.data["notes"].append(text)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data["stage_seconds"][name] = round(
                self.data["stage_seconds"].get(name, 0.0) + time.perf_counter() - start, 3)

    def total_seconds(self) -> float:
        return round(sum(self.data["stage_seconds"].values()), 3)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.data["total_seconds"] = self.total_seconds()
        path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        return path
