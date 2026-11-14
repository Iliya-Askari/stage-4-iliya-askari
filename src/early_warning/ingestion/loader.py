"""Read raw CSV files, enforce the schema contract, split clean rows from quarantined rows."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contract import CONTRACTS, FOREIGN_KEYS, Col, TableContract


class SchemaError(ValueError):
    """The file does not match the contract at the header level (nothing can be loaded)."""


# Timestamp layouts we accept. Ordered from most to least specific; the layout that parses the
# highest share of a column wins, so one file can never mix day-first and month-first.
TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalise_header(name: str) -> str:
    return name.replace("\ufeff", "").strip().strip('"').strip().lower()


def read_raw(path: Path, contract: TableContract) -> pd.DataFrame:
    """Read a CSV as strings, normalise headers, enforce that every contract column exists."""
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"{contract.name}: file not found: {path.name}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [contract.aliases.get(normalise_header(c), normalise_header(c)) for c in df.columns]
    missing = [c for c in contract.column_names if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{contract.name}: file '{path.name}' is missing required columns {missing}; "
            f"found columns {list(df.columns)}"
        )
    return df[contract.column_names].copy()


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse a string column using the single best matching layout. Unparseable -> NaT."""
    text = values.astype("string").str.strip()
    text = text.mask(text == "", pd.NA)
    present = text.notna()
    if not present.any():
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[us]")
    best, best_rate = None, -1.0
    sample = text[present].head(5000)
    for fmt in TS_FORMATS:
        rate = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean()
        if rate > best_rate:
            best, best_rate = fmt, rate
        if rate == 1.0:
            break
    parsed = pd.to_datetime(text, format=best, errors="coerce")
    return parsed.astype("datetime64[us]")


def _coerce(series: pd.Series, col: Col) -> tuple[pd.Series, pd.Series]:
    """Return (typed series, mask of values that were present but invalid)."""
    text = series.astype("string").str.strip()
    blank = text.isna() | (text == "")
    if col.kind in ("str", "zip"):
        out = text.mask(blank, pd.NA)
        if col.kind == "zip":
            digits = out.str.fullmatch(r"\d{1,5}").fillna(False)
            out = out.where(~digits, out.str.zfill(5))
            bad = ~blank & ~out.str.fullmatch(r"\d{5}").fillna(False)
        else:
            bad = pd.Series(False, index=series.index)
        return out, bad
    if col.kind == "ts":
        parsed = parse_timestamps(text)
        return parsed, ~blank & parsed.isna()
    num = pd.to_numeric(text.mask(blank, pd.NA), errors="coerce")
    bad = ~blank & num.isna()
    if col.kind == "int":
        frac = num.notna() & (num != num.round())
        bad = bad | frac
        num = num.where(~frac).round().astype("Int64")
    else:
        num = num.astype("float64")
    return num, bad


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    rows_in: int

    @property
    def stats(self) -> dict:
        reasons: dict[str, int] = {}
        if len(self.quarantine):
            for label in self.quarantine["_reason"].str.split(";").explode():
                if label:
                    reasons[label] = reasons.get(label, 0) + 1
        return {
            "rows_in": self.rows_in,
            "rows_clean": int(len(self.clean)),
            "rows_quarantined": int(len(self.quarantine)),
            "reasons": dict(sorted(reasons.items())),
        }


def validate(df: pd.DataFrame, contract: TableContract) -> ValidationResult:
    """Apply the contract. Never raises on bad rows: they go to the quarantine."""
    raw = df.reset_index(drop=True)
    typed = pd.DataFrame(index=raw.index)
    reasons = pd.Series("", index=raw.index, dtype="object")

    def flag(mask: pd.Series, label: str) -> None:
        nonlocal reasons
        mask = mask.fillna(False).astype(bool)
        reasons = reasons.mask(mask, reasons + label + ";")

    for col in contract.columns:
        values, bad = _coerce(raw[col.name], col)
        typed[col.name] = values
        flag(bad, f"invalid_{col.name}")
        if not col.nullable:
            flag(values.isna() & ~bad, f"null_{col.name}")
        if col.allowed is not None:
            flag(values.notna() & ~values.isin(col.allowed), f"not_allowed_{col.name}")
        if col.kind in ("int", "float"):
            if col.min is not None:
                flag(values.notna() & (values < col.min), f"below_min_{col.name}")
            if col.max is not None:
                flag(values.notna() & (values > col.max), f"above_max_{col.name}")

    if contract.key:
        dup = typed.duplicated(subset=list(contract.key), keep="first")
        flag(dup, "duplicate_key")
    elif contract.dedupe_exact:
        flag(typed.duplicated(keep="first"), "exact_duplicate")

    bad_rows = reasons != ""
    quarantine = raw.loc[bad_rows].copy()
    quarantine["_reason"] = reasons[bad_rows]
    return ValidationResult(typed.loc[~bad_rows].reset_index(drop=True), quarantine, len(raw))


def apply_foreign_keys(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Move child rows whose parent is missing to the quarantine (reason orphan_<column>)."""
    quarantined: dict[str, list[pd.DataFrame]] = {}
    out = dict(tables)
    for child, ccol, parent, pcol in FOREIGN_KEYS:
        if child not in out or parent not in out:
            continue
        orphan = ~out[child][ccol].isin(set(out[parent][pcol]))
        if orphan.any():
            bad = out[child].loc[orphan].copy()
            bad["_reason"] = f"orphan_{ccol}_in_{parent};"
            quarantined.setdefault(child, []).append(bad)
            out[child] = out[child].loc[~orphan].reset_index(drop=True)
    return out, {k: pd.concat(v, ignore_index=True) for k, v in quarantined.items()}


def read_and_validate(path: Path, table: str) -> ValidationResult:
    contract = CONTRACTS[table]
    return validate(read_raw(path, contract), contract)


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def write_quarantine(table: str, quarantine: pd.DataFrame, directory: Path) -> Path | None:
    if quarantine.empty:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{_SAFE.sub('_', table)}.csv"
    quarantine.to_csv(target, index=False)
    return target


def anomaly_counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Rows that are legal under the contract but suspicious. Counted, never dropped."""
    out: dict[str, int] = {}
    if "orders" in tables:
        o = tables["orders"]
        out["orders_approved_before_purchase"] = int((o.order_approved_at < o.order_purchase_timestamp).sum())
        out["orders_delivered_before_carrier"] = int(
            (o.order_delivered_customer_date < o.order_delivered_carrier_date).sum())
        out["orders_never_approved"] = int(o.order_approved_at.isna().sum())
    if "order_reviews" in tables:
        r = tables["order_reviews"]
        out["reviews_answered_before_creation"] = int((r.review_answer_timestamp < r.review_creation_date).sum())
    return out


__all__ = [
    "SchemaError", "ValidationResult", "validate", "read_raw", "read_and_validate",
    "apply_foreign_keys", "write_quarantine", "sha256_of", "parse_timestamps", "anomaly_counts",
]
