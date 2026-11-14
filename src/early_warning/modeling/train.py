"""Model selection on a validation block inside the training period, then a refit on all training data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from .. import config
from ..audit import FEATURE_NAMES
from .dataset import TimeSplit, inner_split, xy
from .evaluate import capture_rate
from .models import CANDIDATES, score


@dataclass
class TrainedModel:
    name: str
    pipeline: object
    selection: pd.DataFrame
    metadata: dict


def select_and_fit(split: TimeSplit, frac: float = config.CAPACITY) -> TrainedModel:
    inner_train, validation = inner_split(split.train)
    Xi, yi = xy(inner_train)
    Xv, yv = xy(validation)
    rows = []
    for name, factory in CANDIDATES.items():
        fitted = factory().fit(Xi, yi)
        s = score(fitted, Xv)
        rows.append({"model": name, "validation_capture_rate": capture_rate(yv, s, frac),
                     "validation_roc_auc": float(roc_auc_score(yv, s))})
    table = pd.DataFrame(rows).sort_values("validation_capture_rate", ascending=False).reset_index(drop=True)
    winner = str(table.loc[0, "model"])
    X, y = xy(split.train)
    final = CANDIDATES[winner]().fit(X, y)
    meta = {
        "model": winner, "features": list(FEATURE_NAMES), "capacity": frac,
        "split": split.summary, "selection": table.round(5).to_dict("records"),
        "sklearn_version": sklearn.__version__, "seed": config.SEED,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return TrainedModel(winner, final, table, meta)


def save_model(model: TrainedModel, directory: Path = config.MODEL_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "early_warning_model.joblib"
    joblib.dump({"pipeline": model.pipeline, "metadata": model.metadata}, path)
    (directory / "early_warning_model.json").write_text(json.dumps(model.metadata, indent=2), encoding="utf-8")
    return path


def load_model(directory: Path = config.MODEL_DIR) -> TrainedModel:
    path = Path(directory) / "early_warning_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no trained model at {path.name}; run the full pipeline first")
    blob = joblib.load(path)
    return TrainedModel(blob["metadata"]["model"], blob["pipeline"], pd.DataFrame(blob["metadata"]["selection"]),
                        blob["metadata"])
