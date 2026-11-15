"""The single chain: empty database -> ingest -> features -> model -> evaluation -> follow-up list."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from . import audit, config
from . import features as F
from .db import ensure_database, make_engine
from .ingestion.manifest import Manifest
from .ingestion.pipeline import ingest_all
from .modeling import baseline as B
from .modeling import drift as D
from .modeling import evaluate as E
from .modeling import leakage as L
from .modeling.ablation import external_signal_ablation
from .modeling.dataset import build_modeling_frame, time_split, xy
from .modeling.models import score
from .modeling.train import save_model, select_and_fit


def _write(df: pd.DataFrame, name: str, art: Path) -> None:
    df.to_csv(art / name, index=False)


def run_all(raw_dir: Path = config.DATA_RAW, external_mode: str = "auto", engine: Engine | None = None,
            artifact_dir: Path = config.ARTIFACT_DIR, with_experiments: bool = True) -> dict:
    """Run the whole system from an empty database. Every stage is timed in the manifest."""
    art = Path(artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    manifest = Manifest()
    with manifest.stage("database_init"):
        ensure_database()
        engine = engine or make_engine()
    ingest_all(raw_dir, engine, manifest, external_mode)

    with manifest.stage("features_build_views_and_snapshot"):
        F.build_feature_layer(engine)
    with manifest.stage("features_query_benchmark"):
        timing = F.time_feature_query(engine)
        (art / "feature_query_plan.txt").write_text(F.explain_feature_query(engine), encoding="utf-8")
        (art / "feature_query_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    with manifest.stage("model_load_frame_and_split"):
        frame = build_modeling_frame(engine)
        split = time_split(frame)
        audit.assert_columns_audited(F.feature_columns(engine))
    results: dict = {"split": split.summary, "feature_query": timing}

    with manifest.stage("model_baseline"):
        rule, rule_table = B.choose_baseline(split.train, config.CAPACITY)
        y_test = split.test["is_negative"].astype(int).to_numpy()
        base_scores = B.rule_scores(split.test, rule)
        results["baseline"] = {"rule": rule, "test_capture_rate": E.capture_rate(y_test, base_scores)}
        _write(rule_table, "baseline_rules_train.csv", art)

    with manifest.stage("model_train_and_select"):
        trained = select_and_fit(split)
        save_model(trained)
        _write(trained.selection, "model_selection.csv", art)
    results["model"] = trained.name

    with manifest.stage("model_evaluate"):
        s = score(trained.pipeline, xy(split.test)[0])
        results["operating_point"] = E.full_report(y_test, s)
        results["operating_point"]["capture_ci95"] = E.bootstrap_capture(y_test, s)
        _write(E.capacity_table(y_test, s), "capacity_table_model.csv", art)
        _write(E.capacity_table(y_test, base_scores), "capacity_table_baseline.csv", art)
        monthly = E.monthly_table(split.test, s)
        _write(monthly, "monthly_capture_model.csv", art)
        base_monthly = E.monthly_table(split.test, base_scores)
        _write(base_monthly, "monthly_capture_baseline.csv", art)
        last = monthly["month"].max()
        mask = split.test["decision_ts"].dt.to_period("M").astype(str) == last
        results["last_month"] = {"month": last, "capture_rate": E.capture_rate(y_test[mask.to_numpy()], s[mask.to_numpy()]),
                                 "orders": int(mask.sum())}
        full = frame.loc[(frame["decision_ts"] >= pd.Timestamp(config.SPLIT_BOUNDARY)) & ~frame["decision_ts_imputed"]]
        fs = score(trained.pipeline, xy(full.assign(is_negative=0))[0])
        out = full[["order_id", "decision_ts"]].assign(risk_score=fs)
        out = out.sort_values(["risk_score", "order_id"], ascending=[False, True]).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        out["follow_up"] = E.top_k_mask(out["risk_score"].to_numpy(), config.CAPACITY)
        _write(out, "follow_up_list_evaluation_period.csv", art)

    with manifest.stage("analysis_drift"):
        drift = D.drift_table(split.train, split.test)
        _write(drift, "drift_psi.csv", art)
        _write(D.monthly_target_rate(pd.concat([split.train, split.test])), "monthly_negative_rate.csv", art)

    if with_experiments:
        with manifest.stage("analysis_leakage_cost"):
            F.build_leaky_layer(engine)
            cost = L.leakage_cost(split, L.load_leaky_columns(engine))
            _write(cost, "leakage_cost.csv", art)
            results["leakage_cost"] = cost.round(4).to_dict("records")
        with manifest.stage("analysis_external_signal_ablation"):
            results["external_ablation"] = external_signal_ablation(split)

    manifest.data["results"] = results
    manifest.write(art / "run_manifest.json")
    (art / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    results["stage_seconds"] = manifest.data["stage_seconds"]
    results["total_seconds"] = manifest.total_seconds()
    return results
