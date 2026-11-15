# Early Warning System - negative-review predictor for Olist orders

At the moment an order is approved, rank it by the risk of ending with a review of 1 or 2 stars, so that the
support team (capacity: 10% of orders) calls the right customers. Everything runs from one command and is
checked against temporal leakage.

## Run it (about 10 lines)
1. Requirements: Python 3.11+, a running PostgreSQL (no Docker needed), internet for the first `pip install`.
2. Copy the nine Olist CSV files (`olist_*_dataset.csv`, `product_category_name_translation.csv`) into `data/raw/`.
3. Optional: `set DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost:5432/early_warning`
   (default: user `postgres`, password `postgres`, database `early_warning`, created automatically).
4. `python run_all.py` creates `.venv`, installs the pinned `requirements.txt` and runs the whole chain
   (ingest -> PostgreSQL -> features -> model -> evaluation -> follow-up list). Add `--with-tests` and/or
   `--with-notebooks` to also run the 55 unit tests and execute both notebooks. `--fresh-venv` starts from nothing.
5. Results land in `data/curated/artifacts/` (capacity table, monthly table, drift, leakage cost, manifest,
   `follow_up_list_evaluation_period.csv`, the feature query plan). Copies of the key evidence are tracked in `docs/evidence/`.

## Score unseen orders
```
python -m early_warning predict --input NEW_ORDERS.csv_or_FOLDER --output ranked.csv [--capacity 0.10]
```
`--input` is either one orders CSV (same schema as `olist_orders_dataset.csv`) or a folder holding any of the
Olist CSV files (orders is mandatory). Wrong schema -> exit code 2 and a message naming the missing columns.
Fresh rows are loaded into a scratch schema; production tables are untouched.

## Layout
```
run_all.py            one-command bootstrap (stdlib only)
src/early_warning/    package: ingestion/ (contract, quarantine, manifest, holiday API), sql/ (schema, features,
                      leaky columns), audit.py (temporal audit registry), modeling/, predict.py, pipeline.py, cli.py
tests/                55 unit/integration tests (unittest)
notebooks/            01_exploration.ipynb (lab notebook), 02_system.ipynb (the final system, saved outputs)
decisions.md          14 architecture decisions;  report.md  the reasoning;  SUMMARY_FOR_OPERATIONS.md  one page for the manager
docs/evidence/        saved plan, timings, tables
```
Tests: `python -m unittest discover -s tests -t .` (needs the same PostgreSQL; each test class uses a throw-away database).

## Honest notes
* The external signal (Brazilian public holidays, Nager.Date API) falls back to a bundled snapshot when the API is
  unreachable; the manifest records which source was used for each year.
* The feature query is fast because the expensive history windows are pre-computed by a stored procedure (timed separately).
