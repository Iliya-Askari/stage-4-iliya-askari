# Report - why it was solved this way

All numbers come from `notebooks/02_system.ipynb` (section in brackets) unless marked 01. The system was run twice from an
empty database and produced identical results (same metrics, same follow-up list).

## 1. Understanding the problem
Every approved order must be ranked by the probability of a 1-2 star review, using only what exists at approval time; the
support team can follow 10% of orders, so the output is a quota-based list and the headline metric is the share of all
negative reviews captured in that 10%.

Assumptions the brief left open: the first *answered* review defines the label (D1); never-approved orders are outside the
system's population (D2); orders without a review are excluded, not treated as satisfied (D3); the survey *answer* time, not
its creation time, is when a review becomes known (section 4); the estimated delivery date is known at purchase (the
estimate is shown at checkout and never precedes it, checks E1-E2 in notebook 01).

## 2. Key decisions (details in `decisions.md`)
| # | Decision | Options | Evidence | Lost |
|---|---|---|---|---|
| 1 | Temporal split at 2018-04-01 | Feb / Apr / Jun | Test share 47% / 33% / 19% (01 s6); 54,753 train, 32,618 test (02 s5) | Test period has a lower negative rate (11.5% vs 14.8%) |
| 2 | 46-day embargo + purge | none / 30 d / 46 d | p99 of approval->answer = 45.5 days (01 s5); 11,050 embargoed, 97 purged | 11k recent labelled orders unused |
| 3 | Label = first answered review | first / last / lowest | 547 multi-review orders, 96 disagree (01 s3) | 96 contradictory labels |
| 4 | Unapproved and unreviewed orders excluded | impute / drop | 160 and 763 orders (02 s5) | Selection bias on silent customers |
| 5 | History as event-stream windows, strictly before decision | GROUP BY / quadratic join / windows | Perturbation tests and mutation checks (tests/test_anti_leak.py) | Constant priors, not fitted |
| 6 | Features in PostgreSQL with materialised roll-ups + refresh procedure | pure view / Python | Serving query 223-330 ms vs 500 ms budget; refresh about 10 s (02 s2, s4); a pure view was 1.1-5.7 s in development (not re-run) | Features are stale until refreshed |
| 7 | Quota (top 10%), not a probability threshold | threshold / quota | Capture at 5/10/20% = 15.6 / 23.7 / 37.1% (02 s6) | Precision depends on the base rate |
| 8 | Select model on end-of-training validation, test touched once | CV on all / inner time block | Deeper boosting 24.2% vs logistic 23.8% validation (02 s6) | Small search, differences between families are small |
| 9 | Holiday signal from Nager.Date with cache and bundled fallback | other APIs / none | Ablation: difference -0.5 to +0.5 points (02 s11) | No measurable gain; dates only |
| 10 | Prediction path in a scratch schema | write to production / Python-only | Score difference vs pipeline 0.0 (02 s13); 16 s per call | Per-call overhead |
| 11 | Two-layer leakage control (registry + perturbation tests) | review only / registry only | 8 anti-leak tests, 2 mutation checks fail as expected | Proven on toy data only |

## 3. What failed
* **Random split.** It looked better (capture 24.3%, AUC 0.667) than the temporal split (23.7%, 0.638) (01 s9.1). The gain is
  moderate here, but it is not real: the model sees the future of sellers and seasons. Abandoned.
* **Pure view for features.** One plain view with all windows and joins took seconds, far above the 500 ms budget; the design moved
  to materialised roll-ups.
* **A correlated sub-query for holidays** (99k index probes) added about 1 s; replaced by a cumulative-calendar join.
* **Holiday feature.** The hypothesis was tested and found useless (interval contains zero); it stays only because it is harmless.
* **Rule baselines.** Distance and promised lead time were weaker than seller reputation on the training period (01 s7).

## 4. Temporal leakage (dedicated section)
**Audit.** `audit.py` holds the table the brief asks for: for each of the 35 model columns, when it exists, whether it exists at
approval, and the source of certainty; `02_system` section 4 prints it, and notebook 01 section 4 prints the empirical checks
(for example 0 orders approved before purchase, 0 items with a shipping limit before purchase, 76.5% negative rate for cancelled
orders vs 12.8% for delivered ones, and only 8.6% of review surveys created before delivery).

**Tempting columns set aside.** `order_status` (final status, decided after approval), the delivered carrier / customer dates
(and their NULLs, which encode status), review score and comments, `review_creation_date`, and the naive all-time seller rate
(contains future reviews and the order's own review).

**Cheat model (same learner, rows and split; only the columns differ), at 10% capacity:**
| model | share of negatives caught | ROC AUC |
|---|---|---|
| honest | 23.7% | 0.638 |
| + status, delivery delay, handover | 42.1% | 0.755 |
| + naive all-time seller rate | 29.8% | 0.718 |
| + review comment length | 40.4% | 0.792 |
| all leaky columns (cheat) | 54.2% | 0.881 |

Leakage would have faked a gain of 30.6 capture points and 24.3 AUC points.

**Aggregate features.** Every history feature (customer orders and negative rate; seller orders, reviews, negative rate,
deliveries, late rate) counts only events with a timestamp strictly before the order's decision time; a review counts from its
answer timestamp, a delivery from the delivery date. Tests: the features are identical when post-approval columns are changed
or when anything after an order's decision time is flipped, and they do change when the past is flipped (so the test is sensitive).

## 5. Results (evaluation period Apr-Aug 2018, 32,618 labelled orders, base rate 11.5%)
* Baseline rule (sellers with the worst reputation): 16.8% of negatives caught at 10% capacity; random would give 10%.
* Model: **23.7%** (95% interval 22.5-24.8%); precision 27.2%, F1 0.253, ROC AUC 0.638, PR AUC 0.231.
  Confusion matrix at the quota: 888 true positives, 2,374 false positives, 2,865 false negatives, 26,491 true negatives.
* Months: capture 24.3 / 24.1 / 23.9 / 21.6 / 21.0% for Apr-Aug; last month (Aug, 6,594 orders) 21.0%, AUC 0.596.
* Time: the whole chain runs in about 80 s from an empty database (02 s2); installing the pinned packages into a brand-new
  virtual environment is network-bound (more than 8 minutes in the build sandbox, not stopwatch-timed); both fit the 45-minute budget.

## 6. Drift and what breaks first
Largest shifts between the periods (PSI, 02 s9): seller late rate 0.79, shipping-limit slack 0.41, promised lead time 0.30 (the
platform shortened its estimates by 3 days at the median), freight 0.29; the negative rate fell from 14.8% to 11.5%. The slow
decline in monthly capture (24.3% to 21.0%) and AUC (0.655 to 0.596) is consistent with these shifts, because the history and
lead-time features are among the ones that moved; this is an association, not a proven cause.
If the system ran for six months unchanged, the seller-history features would go first (sellers churn, delivery promises and
carrier performance change) and the negative-rate base would move. It would be noticed by (a) the monthly capture rate and PSI of
the same features, (b) the weekly share of reviewed orders that are negative, and (c) the share of sellers with no history;
labels arrive about 45 days late, so the capture rate is always a month and a half behind and the PSI is the early signal.

## 7. Limits and risks
Breaks if: the review/approval timestamps change meaning; the schema changes (rejected with an error); a new seller or category
dominates (priors only); the holiday API returns different content (the bundled fallback is a hand-written snapshot). The product
and geography tables are static snapshots, so a catalogue edit after approval would leak slightly. Absolute performance is modest
(AUC 0.64): the information available at approval limits the ranking.

## 8. With two more days
Build the Stage 1-3 merge properly (see D14), calibrate the priors by sellers' age, add regional weather as a second external
signal, test a monthly retraining schedule on the same split, and write the monitoring job that computes PSI and capture by month.

## AI usage

An AI assistant (Claude) drafted the code, SQL queries, tests, notebooks, and project documentation through an iterative development process. The assistant was used as an expert-level technical consultant and development partner throughout the project. The student provided project requirements, defined objectives, guided the overall workflow, discussed technical approaches, and made decisions at different stages of development. Claude was used to develop and refine the implementation, troubleshoot technical issues, optimize queries, review results, and improve the project documentation. The student actively guided the development process, evaluated the generated outputs, requested modifications where necessary, and worked through multiple rounds of consultation and refinement until the project was completed. The final implementation was therefore achieved through continuous collaboration, technical guidance, review, and iterative improvement rather than simply accepting the initial AI-generated draft. Any part of the project may be discussed or questioned during the defence.
