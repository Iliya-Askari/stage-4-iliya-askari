"""External signal: Brazilian national public holidays from the free Nager.Date API.

Hypothesis: an order whose promised delivery window contains public holidays is more likely
to arrive late (carriers and post offices close), and late orders get negative reviews.
The signal links to an order through dates only (the API has no regional detail for BR).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd
import requests

from .. import config

API_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/BR"
MODES = ("auto", "live", "cache", "offline")


class ExternalSourceError(RuntimeError):
    pass


def _fallback_records() -> list[dict]:
    text = resources.files("early_warning").joinpath("ingestion", "holidays_br_fallback.json").read_text("utf-8")
    return json.loads(text)["holidays"]


def fetch_year_live(year: int, session: requests.Session | None = None, retries: int = 3,
                    timeout: float = 15.0) -> list[dict]:
    http = session or requests
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = http.get(API_URL.format(year=year), timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ExternalSourceError(f"unexpected payload type {type(payload).__name__} for {year}")
            return payload
        except (requests.RequestException, ValueError, ExternalSourceError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 4) * 0.5)
    raise ExternalSourceError(f"live API failed for {year} after {retries} attempts: {last}")


def normalise(payload: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in payload:
        types = rec.get("types") or ["Public"]
        if "Public" not in types:
            continue
        rows.append({"holiday_date": rec["date"], "name": rec.get("name") or rec.get("localName", "holiday")})
    df = pd.DataFrame(rows, columns=["holiday_date", "name"])
    df["holiday_date"] = pd.to_datetime(df["holiday_date"], format="%Y-%m-%d").dt.date
    return df.drop_duplicates("holiday_date")


def load_holidays(years: list[int], mode: str = "auto", cache_dir: Path | None = None,
                  session: requests.Session | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (holidays frame with a `source` column, info dict for the manifest)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cache_dir = Path(cache_dir or config.EXTERNAL_CACHE_DIR)
    frames, per_year = [], {}
    live_down = False   # circuit breaker: after one failed year do not wait for the others
    fallback = pd.DataFrame(_fallback_records()).rename(columns={"date": "holiday_date"})
    fallback["holiday_date"] = pd.to_datetime(fallback["holiday_date"]).dt.date

    for year in sorted(set(years)):
        source, payload = None, None
        cache_file = cache_dir / f"holidays_BR_{year}.json"
        if mode in ("auto", "live") and not live_down:
            try:
                payload = fetch_year_live(year, session=session)
                source = "live_api"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except ExternalSourceError:
                if mode == "live":
                    raise
                live_down = True
        if payload is None and mode in ("auto", "cache") and cache_file.exists():
            payload, source = json.loads(cache_file.read_text("utf-8")), "disk_cache"
        if payload is not None:
            df = normalise(payload)
        else:
            if mode == "live":
                raise ExternalSourceError(f"no data for {year}")
            df = fallback[pd.to_datetime(fallback["holiday_date"]).dt.year == year][["holiday_date", "name"]]
            source = "bundled_fallback"
        df = df.assign(source=source)
        frames.append(df)
        per_year[str(year)] = {
            "source": source, "records": int(len(df)),
            "content_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16],
        }
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["holiday_date", "name", "source"])
    info = {
        "name": "Nager.Date public holidays (BR)", "url_template": API_URL, "mode_requested": mode,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "years": per_year,
        "used_fallback": any(v["source"] == "bundled_fallback" for v in per_year.values()),
    }
    return out, info
