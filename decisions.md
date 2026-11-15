# Architecture decisions

Each decision: context, options, decision, accepted costs. Numbers refer to `notebooks/02_system.ipynb` unless stated.

## D1 - Prediction unit and label
**Context.** The brief wants one prediction per order; 547 orders have several reviews, 96 of them disagreeing on the label.
**Options.** (a) first answered review, (b) last review, (c) lowest score, (d) drop disagreeing orders.
**Decision.** (a): the first review that was *answered*; negative means score 1 or 2. It is the earliest outcome the support team could have prevented and it keeps "knowledge time" simple (answer timestamp).
**Accepted.** 96 orders are labelled by a review the customer later contradicted.

## D2 - Orders without an approval timestamp
**Context.** 160 orders have no approval time (mostly cancelled before payment cleared); the decision moment is approval.
**Options.** (a) drop everywhere, (b) impute, (c) treat purchase time as decision time.
**Decision.** Excluded from training and evaluation (the system is *triggered* by an approval). The prediction path still scores them with decision time = purchase + 21 min (the measured median lag) and flags `approval_imputed`.
**Accepted.** The model never learns from orders that die before approval; the 21-minute constant is a documented assumption.

## D3 - Orders without a review
**Context.** 763 approved orders have no review (646 are delivered).
**Options.** (a) treat as not negative, (b) treat as negative, (c) exclude.
**Decision.** Exclude. Their outcome is unknown, not "fine"; labelling them either way invents data.
**Accepted.** Selection bias: customers who never answer are invisible. The share is under 1% of approved orders, so the effect is small but real.

## D4 - Boundary date of the temporal split
**Context.** Train on the past, evaluate on the future; labelled data spans 2016-09 to 2018-09.
**Options.** 2018-02-01 (47% test), 2018-04-01 (33% test), 2018-06-01 (19% test).
**Decision.** 2018-04-01. Training keeps a full Black Friday / Christmas season and the early-2018 delivery problems; evaluation keeps five complete months (Apr-Aug 2018) for the monthly spread.
**Accepted.** The evaluation period has a lower negative rate (11.5% vs 14.8%), so absolute metrics are not comparable to the training period.

## D5 - Label embargo and purge
**Context.** A training order's label is unknown until the customer answers; p99 of approval -> answer is 45.5 days.
**Options.** no gap; 30-day gap; 46-day gap + purge.
**Decision.** Training only uses orders approved at least 46 days before the boundary, and drops any row whose label arrived after the boundary (97 rows purged, 11,050 embargoed).
**Accepted.** About 11k labelled orders are not used for training (the most recent ones before the boundary).

## D6 - History features: event stream, answer time, shrinkage
**Context.** Seller / customer history is valuable and the easiest feature to leak.
**Options.** (a) `GROUP BY seller` over everything, (b) self-join with `<` per order (quadratic), (c) event-stream window functions.
**Decision.** (c): review, delivery and order events are merged per seller in time order; each order reads the running sum of events strictly before its own decision timestamp (queries sort before events with equal time). A review counts from its *answer* timestamp, not its creation date. Rates are shrunk to fixed priors (0.15 negative, 0.08 late, strength 5) so thin histories do not look extreme.
**Accepted.** Priors are constants chosen from domain knowledge, not fitted; sellers with no history get the prior.

## D7 - Where features are built: database or Python
**Context.** The brief requires features in the database (view or procedure), under 500 ms.
**Options.** Python/pandas merge; SQL views only; SQL with materialised roll-ups and a refresh procedure.
**Decision.** SQL. Row-local roll-ups and the two history streams are materialised; `v_features` joins them and applies the shrinkage. `sp_build_feature_snapshot()` refreshes everything and stores a snapshot table.
**Accepted.** Features are stale until the procedure runs; a plain all-in-one view measured 1.1-5.7 s during development (not re-run), the materialised design measures about 0.3 s server-side.

## D8 - View or snapshot table for the model layer
**Context.** Training must be reproducible and fast; serving must always use the same logic.
**Decision.** Both: `v_features` is the single definition of the features; the model layer reads the snapshot table created from it by the procedure, so one run uses one frozen feature set recorded in the manifest. The prediction path rebuilds the same objects in a scratch schema.
**Accepted.** Two representations of the same data; the snapshot must be rebuilt after new data arrives.

## D9 - Decision rule and headline metric
**Context.** Support can handle 10% of orders; a probability threshold would drift with the base rate.
**Decision.** Rank and flag exactly ceil(10% x orders). Headline metric: share of all negative reviews inside the top 10%; also reported at 5% and 20%, per month, and for the last month.
**Accepted.** Precision at the quota depends on the base rate and is reported alongside.

## D10 - Model families and selection
**Options.** Logistic regression, small and deeper gradient boosting (scikit-learn only).
**Decision.** Select on a validation block taken from the *end of the training period* (with its own embargo), then refit the winner on all training data; the evaluation period is touched once. Winner: deeper gradient boosting (validation capture 24.2% vs 23.8% logistic); the differences between families are small.
**Accepted.** Little hyper-parameter search; the ceiling is the information in the audited columns, not the learner.

## D11 - External signal
**Context.** Need a free source without access limits that links by date or place.
**Decision.** Nager.Date public holidays for Brazil; feature = number of holidays between approval and the estimated delivery date. Policy: live API -> disk cache -> bundled snapshot; the manifest records the source per year; one failed year disables live calls for the others.
**Result.** Ablation shows no measurable gain (difference of capture rate -0.5 to +0.5 points, 95% interval). Reported as a negative result.
**Accepted.** Dates only (no regional holidays); the bundled snapshot was written by hand and could differ from the API on edge cases.

## D12 - Prediction path
**Decision.** One entry point accepts an orders CSV or a folder of Olist CSVs, validates headers and rows with the same contract, loads fresh rows into a scratch copy of the source schema (production untouched, dropped afterwards), runs the same feature SQL and model, and prints a ranked list. History features use the production tables as of each order's own approval time, so scoring an order that is already in the database gives the identical score (checked: maximum difference 0.0).
**Accepted.** About 16 s per call (measured on 2,551 orders) because the feature layer is rebuilt in the scratch schema.

## D13 - How leakage is caught
**Decision.** Two layers. (1) A machine-readable registry (`audit.py`): every model column must be listed, flagged available at approval, and aggregates must name their time control; the feature view's columns must equal the registry. (2) Perturbation tests on a synthetic database: changing post-approval columns, or any event after an order's decision time, must not change that order's features, while changing the past must. Mutation checks (answer time replaced by creation time; query rows sorted after events) turn the tests red.
**Accepted.** Tests prove invariance on a toy dataset, not on production data.

## D14 - Reuse of stages 1-3
**Context.** The brief asks for the stage 1-3 package, database and discipline to be reused.
**Decision.** The stage 1-3 code base was not available while this system was built, so the ingestion layer (contract, quarantine, manifest), the PostgreSQL schema/views/procedure and the notebook discipline were re-implemented here with the same concepts.
**Accepted.** Stage 1-3 tests are not part of this repository and could not be run; merge the packages before submitting if a literal reuse is required.
