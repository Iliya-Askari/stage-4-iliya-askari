"""Central configuration. Every tunable lives here so decisions.md can point to one place."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CURATED = PROJECT_ROOT / "data" / "curated"
QUARANTINE_DIR = DATA_CURATED / "quarantine"
EXTERNAL_CACHE_DIR = DATA_RAW / "external"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = DATA_CURATED / "artifacts"

SEED = 42

# --- database -------------------------------------------------------------
DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/early_warning"
SCHEMA_SRC = "olist"        # typed, validated source tables
SCHEMA_EXT = "ext"          # external signals (public holidays)
SCHEMA_FEAT = "feat"        # feature views / snapshot
SCHEMA_LEAKY = "leaky"      # deliberately leaky columns, used ONLY by the cheat model
SCHEMA_INFER_SRC = "olist_infer"
SCHEMA_INFER_FEAT = "feat_infer"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


# --- problem definition ---------------------------------------------------
NEGATIVE_MAX_SCORE = 2            # review score 1 or 2 means "negative"
SPLIT_BOUNDARY = "2018-04-01"     # train before, evaluate on/after (see decisions.md D3)
LABEL_EMBARGO_DAYS = 46           # ceil of the p99 approval -> review answer lag (45.5 days, see notebook 01)
CAPACITY = 0.10                   # support team can follow up on 10% of orders
CAPACITY_GRID = (0.05, 0.10, 0.20)
FEATURE_QUERY_BUDGET_MS = 500
TOTAL_BUDGET_MINUTES = 45

# Bayesian shrinkage used by the seller / customer history features (see decisions.md D6)
PRIOR_NEG_RATE = 0.15
PRIOR_STRENGTH = 5.0
PRIOR_LATE_RATE = 0.08

# Brazil bounding box used to discard impossible geolocation points
BRAZIL_LAT = (-34.0, 6.0)
BRAZIL_LNG = (-74.0, -34.0)

# Decision moment fallback for the few orders that have no approval timestamp (see decisions.md D2)
IMPUTED_APPROVAL_LAG_MINUTES = 21   # median approval lag measured in notebook 01
