"""Generate the two notebooks (narrative + thin code cells). Run tools/execute_notebooks.py afterwards."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from IPython.display import display
from early_warning import audit, config, features as F, viz
from early_warning.config import SEED          # the single seed of the whole project
np.random.seed(SEED)
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 100)
pd.options.display.float_format = "{:,.4f}".format
"""


def nb01():
    c = []
    c.append(md("""
# 01 - Exploration: the temporal audit, the label and the paths that failed

**Question.** At the moment an order is approved, which orders will end with a negative review (score 1 or 2)?
The support team can follow up on 10% of orders, so the system must produce a ranked list, not a yes/no.

**What this notebook does.** It is the lab notebook: how the data behaves, when each column becomes known,
why the label, the split and the embargo were chosen, and which attempts were abandoned.
The final system is narrated in `02_system.ipynb`.
"""))
    c.append(code(SETUP + """
from early_warning import exploration as X
from early_warning.pipeline import ensure_loaded
from early_warning.modeling import dataset as DS, baseline as B, evaluate as E, leakage as L
engine = ensure_loaded()
print("seed:", SEED)
"""))
    c.append(md("## 1. The data at a glance\nRow counts of the typed tables that the ingestion layer loaded into PostgreSQL (the model layer reads only from there)."))
    c.append(code("display(X.table_counts(engine))"))
    c.append(md("""
## 2. The decision moment: order approval
The brief fixes the decision moment at order approval. A few orders have no approval timestamp.
We look at who they are before deciding what to do with them (decision D2 in `decisions.md`).
"""))
    c.append(code("""
display(X.approval_lag_summary(engine))
display(X.never_approved_profile(engine))
"""))
    c.append(md("""
Most never-approved orders are cancelled before payment cleared. The system is triggered *by* an approval,
so such orders are outside its population: they are excluded from training and evaluation. The prediction
path still scores them with an imputed approval time (purchase + the median lag shown above) and flags them.
"""))
    c.append(md("""
## 3. The label and the orders without a review
An order has a negative label when its review score is 1 or 2. Orders that never received a review have an
*unknown* outcome - not a positive and not a negative. Dropping them is a decision with a side effect, so we measure it.
"""))
    c.append(code("""
display(X.label_coverage(engine))
display(X.multi_review_orders(engine))
"""))
    c.append(md("""
Only a small share of orders lacks a review, so excluding them changes little. Orders that were cancelled or never
delivered have a much higher negative rate: those outcomes are decided *after* approval, which is exactly why
`order_status` is a trap (section 4). A few hundred orders carry several reviews; we use the first one that was answered
and document the disagreement count above.
"""))
    c.append(md("""
## 4. Temporal audit
For every column we ask: *at the moment of approval, does this value already exist?*
First the empirical checks that back the claims (all queries run against the loaded data), then the audit table
that the code enforces.
"""))
    c.append(code("display(audit.run_evidence(engine))"))
    c.append(code("display(audit.audit_table()[['column','available_at','available_at_approval','source_of_certainty']])"))
    c.append(md("### Columns that look useful but were set aside"))
    c.append(code("display(audit.excluded_table())"))
    c.append(md("### One sentence per feature: the hypothesis it tests"))
    c.append(code("display(audit.hypothesis_table())"))
    c.append(md("""
## 5. When does a review become *known*? (label embargo)
A review is created when the survey is sent but the score only exists when the customer answers. History features and
training labels must therefore use the *answer* timestamp. The lag between approval and answer also tells us how long a
training order's label stays unknown, which sets the embargo between train and test.
"""))
    c.append(code("""
display(X.review_lag_quantiles(engine))
print("embargo used (days):", config.LABEL_EMBARGO_DAYS)
"""))
    c.append(md("""
The 99th percentile is about 45 days, so the embargo is 46 days: no training order is closer to the boundary than the
time it takes (for 99% of orders) for its label to arrive. A purge on top removes the few remaining rows whose label
arrived after the boundary.
"""))
    c.append(md("## 6. Where to cut in time?"))
    c.append(code("""
frame = DS.build_modeling_frame(engine)
monthly = X.monthly_volume_and_rate(frame)
display(monthly)
viz.volume_and_rate(monthly, config.SPLIT_BOUNDARY)
"""))
    c.append(code("display(X.boundary_sensitivity(frame))"))
    c.append(md("""
The boundary is 2018-04-01. It keeps a full Black-Friday/Christmas season and the early-2018 delivery problems in training,
and leaves the largest part of the data that still has complete labels (about a third) for evaluation. Cutting at
2018-02 would test on a period that is still very close to the shock months; cutting at 2018-06 leaves too little to
evaluate month by month. Note that the negative rate is lower in the evaluation period: the world changed (see drift).
"""))
    c.append(md("## 7. A rule the operations team could apply by hand (baseline)"))
    c.append(code("""
split = DS.time_split(frame)
print(split.summary)
rule, table = B.choose_baseline(split.train, config.CAPACITY)
display(table)
y_test = split.test["is_negative"].astype(int).to_numpy()
print("chosen on the TRAINING period:", rule)
print("capture rate on the evaluation period:", round(E.capture_rate(y_test, B.rule_scores(split.test, rule)), 4))
"""))
    c.append(md("""
## 8. Hypotheses: what do the features say before any model?
Negative-review rate by bucket of three audited features, on labelled training orders.
"""))
    c.append(code("""
for col in ["promised_days_from_approval", "s_neg_rate_known", "distance_km"]:
    display(X.rate_by_bucket(split.train, col))
"""))
    c.append(md("""
### The external signal (public holidays from a free API)
Hypothesis: if the promised delivery window contains public holidays, carriers and post offices are closed, deliveries
slip and customers complain. The signal links to an order by date only. First the raw relationship, then the honest test:
the same model with and without the feature, with a bootstrap interval for the difference on the evaluation period.
"""))
    c.append(code("""
display(X.rate_by_bucket(split.train, "n_holidays_in_promise_window", q=6))
from early_warning.modeling.ablation import external_signal_ablation
ablation = external_signal_ablation(split)
display(pd.Series(ablation, name="value").to_frame())
"""))
    c.append(md("""
**Result: negative.** The interval of the difference contains zero, so the holiday feature adds nothing measurable beyond
the other features (the promised lead time already absorbs most of the calendar effect). The hypothesis was tested and is
reported as rejected rather than hidden; the feature stays in the audited set because it is harmless, but the decision does
not depend on it.
"""))
    c.append(md("## 9. Paths that failed"))
    c.append(md("""
### 9.1 A random split looks better - and is wrong
Randomly splitting orders lets the model learn from the future (seller reputation, seasonal shocks) and inflates the metric.
"""))
    c.append(code("display(L.random_vs_temporal(frame, split))"))
    c.append(md("""
### 9.2 The naive all-time seller rate
The most natural reputation feature is a plain `GROUP BY seller` over all reviews. It contains the future *and the order's own
review*. The leakage table below prices it (and the other tempting columns); the full comparison is in `02_system.ipynb`.
"""))
    c.append(code("""
leaky = L.load_leaky_columns(engine)
display(L.leakage_cost(split, leaky)[["model", "capture_rate", "roc_auc", "capture_rate_gain_vs_honest"]])
"""))
    c.append(md("""
### 9.3 Feature query performance (development notes)
The first version computed the history windows and all joins inside one plain view. Measured during development
(not re-run here) it took several seconds against the 500 ms budget. Pre-computing the expensive event-stream windows and the
row-local roll-ups in materialised views brought the serving query under budget (`02_system.ipynb` shows the measured plan).
The cost moved into the refresh step, which is timed separately.
"""))
    return c


def nb02():
    c = []
    c.append(md("""
# 02 - The system: from raw files to the follow-up list

This notebook runs the **whole chain from an empty database** with one call and narrates each result:
ingestion with quarantine -> PostgreSQL -> features built in the database -> model -> capacity driven evaluation ->
drift -> price of leakage -> operational list -> prediction path for unseen orders.
Heavy logic lives in the `early_warning` package; cells only call it and show the result.
"""))
    c.append(code(SETUP + """
import json
from early_warning import demo
from early_warning.db import make_engine
from early_warning.pipeline import run_all
from early_warning.predict import predict
from early_warning.ingestion.loader import SchemaError
ART = config.ARTIFACT_DIR
print("seed:", SEED, "| capacity:", config.CAPACITY, "| boundary:", config.SPLIT_BOUNDARY)
"""))
    c.append(md("## 1. Run the chain\nOne call: it rebuilds the schemas, so nothing depends on manual state."))
    c.append(code("""
results = run_all(external_mode="auto")
engine = make_engine()
manifest = json.loads((ART / "run_manifest.json").read_text())
print("model selected:", results["model"], "| chain time (s):", results["total_seconds"])
"""))
    c.append(md("## 2. Time budget\nEvery stage is timed separately. The budget for the whole run is 45 minutes (environment setup is timed by `run_all.py`)."))
    c.append(code("""
timing = pd.Series(results["stage_seconds"], name="seconds").to_frame()
timing["share_%"] = 100 * timing["seconds"] / timing["seconds"].sum()
display(timing)
print(f"total {results['total_seconds']:.0f} s of {config.TOTAL_BUDGET_MINUTES * 60} s budget")
bootstrap = ART / "bootstrap_timing.json"
if bootstrap.exists():
    display(pd.Series(json.loads(bootstrap.read_text()), name="seconds").to_frame())
"""))
    c.append(md("## 3. Ingestion: contract, quarantine, manifest"))
    c.append(code("""
tables = pd.DataFrame(manifest["tables"]).T[["rows_in", "rows_clean", "rows_quarantined", "rows_loaded"]]
display(tables)
display(pd.Series(manifest["anomalies"], name="rows").to_frame())
display(pd.DataFrame(manifest["external_sources"]["public_holidays_br"]["years"]).T)
print("notes:", manifest["notes"])
"""))
    c.append(md("The real files are clean, so to show the contract working we corrupt three rows of the reviews file in memory:"))
    c.append(code("display(demo.quarantine_demo(config.DATA_RAW))"))
    c.append(md("""
Rows that break the contract would be written to `data/curated/quarantine/` with a machine-readable reason; the manifest
records counts per reason. Anomalies are legal rows that look suspicious: they are counted, not dropped.
The last table shows where the external signal came from for each year (live API, disk cache or the bundled snapshot).
"""))
    c.append(md("## 4. Features built in the database, and their temporal audit"))
    c.append(code("display(audit.audit_table()[['column','available_at','available_at_approval','aggregate','source_of_certainty']])"))
    c.append(code("""
timing_q = results["feature_query"]
display(pd.Series(timing_q, name="value").to_frame())
plan = (ART / "feature_query_plan.txt").read_text().splitlines()
print("\\n".join(plan[:6]), "\\n...\\n", "\\n".join(plan[-2:]))
"""))
    c.append(md("""
The serving query (`feat.v_features`) reads pre-computed roll-ups; its server-side execution time is compared to the
500 ms budget above, and the full `EXPLAIN (ANALYZE, BUFFERS)` plan is saved in `data/curated/artifacts/feature_query_plan.txt`.
The expensive part (sorting about 300k event rows per history stream) lives in the refresh procedure, timed in section 2
(`features_build_views_and_snapshot`). The 'client fetch' number includes moving 99k rows into Python and is reported for honesty.
"""))
    c.append(md("## 5. Split"))
    c.append(code("display(pd.Series(results['split'], name='value').to_frame())"))
    c.append(md("## 6. Baseline rule vs model at 5%, 10% and 20% capacity"))
    c.append(code("""
cap_model = pd.read_csv(ART / "capacity_table_model.csv")
cap_base = pd.read_csv(ART / "capacity_table_baseline.csv")
print("baseline rule:", results["baseline"]["rule"])
display(cap_base.assign(who="baseline rule"))
display(cap_model.assign(who=results["model"]))
display(pd.read_csv(ART / "model_selection.csv"))
"""))
    c.append(md("## 7. Operating point: the top 10% of the ranking"))
    c.append(code("""
op = results["operating_point"]
cm = pd.DataFrame([[op["tp"], op["fn"]], [op["fp"], op["tn"]]],
                  index=["flagged for follow-up", "not flagged"], columns=["negative review", "no negative review"])
display(cm)
display(pd.Series({k: op[k] for k in ["capture_rate", "precision", "recall", "f1", "roc_auc", "pr_auc", "base_rate"]}, name="value").to_frame())
print("95% interval of the capture rate:", [round(x, 4) for x in op["capture_ci95"]])
"""))
    c.append(code("""
from early_warning.modeling.dataset import build_modeling_frame, time_split, xy
from early_warning.modeling.train import load_model
from early_warning.modeling.models import score
frame = build_modeling_frame(engine)
split = time_split(frame)
model = load_model()
y_test = split.test["is_negative"].astype(int).to_numpy()
s_model = score(model.pipeline, xy(split.test)[0])
from early_warning.modeling import baseline as B
s_rule = B.rule_scores(split.test, results["baseline"]["rule"])
viz.capacity_curves({"model": (y_test, s_model), "rule baseline": (y_test, s_rule)})
"""))
    c.append(md("## 8. Spread over the months of the evaluation period"))
    c.append(code("""
monthly = pd.read_csv(ART / "monthly_capture_model.csv")
monthly_base = pd.read_csv(ART / "monthly_capture_baseline.csv")
display(monthly)
print("last month:", results["last_month"])
viz.monthly_bars(monthly, monthly_base)
"""))
    c.append(md("""
The monthly numbers apply the same 10% quota inside each month. The last month is reported on its own as required;
if it is lower than the pooled number, section 9 shows which features moved.
"""))
    c.append(md("## 9. Drift between the training and the evaluation period"))
    c.append(code("""
drift = pd.read_csv(ART / "drift_psi.csv")
display(drift)
viz.psi_bars(drift)
display(pd.read_csv(ART / "monthly_negative_rate.csv"))
"""))
    c.append(md("""
PSI above 0.25 is a major shift. The history features of sellers and the promised lead time moved the most: the platform
shortened its delivery promises and sellers' punctuality changed. The negative rate itself fell in the evaluation period, so the
ranking quality (capture rate) must be read against a lower base rate.
"""))
    c.append(md("## 10. What leakage would have bought: the cheat model"))
    c.append(code("""
cost = pd.read_csv(ART / "leakage_cost.csv")
display(cost[["model", "extra_columns", "capture_rate", "roc_auc", "pr_auc", "capture_rate_gain_vs_honest", "roc_auc_gain_vs_honest"]])
viz.leakage_bars(cost)
"""))
    c.append(md("""
Same learner, same rows, same split; the only difference is which columns the model may see. The gap between the first row
(honest) and the last row (cheat) is what leakage would have faked on this notebook - a model that is useless the moment it
meets a real, just-approved order.
"""))
    c.append(md("## 11. Does the external signal help?"))
    c.append(code("display(pd.Series(results['external_ablation'], name='value').to_frame())"))
    c.append(md("## 12. The operational output: the follow-up list"))
    c.append(code("""
follow = pd.read_csv(ART / "follow_up_list_evaluation_period.csv")
print("orders scored:", len(follow), "| flagged for follow-up:", int(follow["follow_up"].sum()))
display(follow.head(10))
"""))
    c.append(md("""
## 13. The prediction path for unseen orders
`python -m early_warning predict --input <file or folder> --output <csv>` validates the schema, applies the same SQL feature
preparation in an isolated scratch copy and returns a ranked list. Below: a folder of newly approved orders (last two weeks of
the data, written as plain CSV files), the ranked output, a consistency check against the main pipeline, and a wrong schema.
"""))
    c.append(code("""
drop = demo.make_demo_slice(config.DATA_RAW, approved_from="2018-08-15")
ranked = predict(drop, engine)
print("ranked orders:", len(ranked), "| flagged:", int(ranked["follow_up"].sum()))
display(ranked.head(8))
check = ranked.merge(follow[["order_id", "risk_score"]], on="order_id", suffixes=("_predict", "_pipeline"))
print("max |score difference| vs the main pipeline:", float((check.risk_score_predict - check.risk_score_pipeline).abs().max()))
"""))
    c.append(code("""
try:
    predict(config.DATA_CURATED / "demo_orders_missing_status.csv", engine)
except SchemaError as exc:
    print("rejected with a clear error ->", exc)
"""))
    c.append(md("""
## 14. Reading guide for the operations manager
See `SUMMARY_FOR_OPERATIONS.md`. Short version: the list is a *prioritisation aid*, it catches roughly twice as many unhappy
customers as picking orders at random, it is not a crystal ball, and it must be watched monthly because the platform changes.
"""))
    return c


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)


if __name__ == "__main__":
    write("01_exploration.ipynb", nb01())
    write("02_system.ipynb", nb02())
    print("notebooks written")
