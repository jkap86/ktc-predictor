# Production-conditioned market-movement evaluation

Date: 2026-10-25. Status: design for review; no implementation or evaluation is authorized by this document alone. Base: `c4270804379c26bcb340f32a07e507153b4c7342` (master after PR #8).

## Purpose and limits

Answer this question: **On subsequent historical seasons, does season production add predictive information about a player's Backchannel market-value movement beyond starting value, age and position?** Evaluate all primary models against one MFL-inclusive Backchannel target, with a no-production ablation and no-change baseline. KTC is an optional secondary benchmark, never a prerequisite for usefulness or a promotion gate.

The product is a conditional scenario tool: given start value, age, position and an assumed season points total, estimate end value. Historical tests supply realized points. A successful test supports the conditional production-to-value relationship; it does not validate a preseason points forecast, a combined preseason forecast, or a causal effect of scoring more points. The full model also remains subject to unobserved injuries, roles, expectations and market changes.

This is retrospective development on existing 2020–2025 evidence. Every fold is temporally out of sample for its fitted estimator, but the architecture and data choices were informed by earlier historical experiments. In particular, **2025 is reused development evidence**, even if withheld from a particular fold's fit. Neither five chronological folds nor new report names turn these seasons into an untouched confirmatory test.

The design evaluates predictability of one generated market measurement, not whether that measurement is objective exchange value. Backchannel combines Sleeper and MFL trades, priors, a valuation algorithm and reference-scale choices; it is neither MFL-only nor raw transaction prices. Independent trade/league validation of the valuation model and uncertainty in measured endpoint values remain separate research.

## Approaches considered

1. Extend `evaluate.py` or adapt a sealed confirmation/replacement runner. These have useful metrics and identity checks, but modifying bound sources invalidates sealed identities, and the replacement runner answers a different question using source-specific targets. Rejected.
2. Add an isolated historical movement evaluator with one shared target, matched ablation, fixed chronological folds and explicit measurement/coverage limits. Chosen within the user-approved separate-evaluator scope. It answers the incremental-production question without changing either sealed protocol or runtime.
3. First build a new independently held-out raw-trade target or a full preseason points forecaster. These would address broader product and measurement questions, but require new data, definitions and validation beyond this historical-only task. Deferred; absence of that work limits claims, not the ability to run the scoped experiment.

The constant-points ablation, fixed expanding window, metric conventions and output contract below are design decisions proposed for this evaluator, not claims that the user selected particular numeric thresholds.

## Protected boundary

Implementation must add only isolated evaluator modules, tests and its own future development outputs. Do not edit any existing file bound by either sealed manifest, including `model.py`, `evaluate.py`, `labels.py`, `data.py`, `__main__.py`, `__init__.py`, `label_export.py`, SQL recipes, confirmation/replacement sources, training data, provenance, mapping snapshots, starts, manifests or registries. Preserve existing artifacts and documentation of historical outcomes. Do not regenerate or repair a seal to accommodate this work.

No database queries, data exports, network data fetches, 2026 season outcomes, real registrations, claims, attestations, registry transitions, sealed protocol execution, promotion decisions, model persistence, runtime configuration, API, UI or deployment changes. Reading existing protocol metadata for an integrity comparison is allowed; importing their lifecycle runners or opening their outcome paths is unnecessary and prohibited for this evaluator.

The eventual implementation PR must prove that the protected file set (the union of both manifests' identities and all directly bound inputs/metadata) is byte-identical to the base. That is a read-only hash check, not a seal operation. The present design commit contains this document only.

## Primary target and exact data identity

The target identifier is `backchannel-mfl-v1-reference-same-season-log-movement-2020-2025-v1`.

| Input | Exact repository path | SHA-256 |
| --- | --- | --- |
| Labels | `backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv` | `3e57be0b6e6ccb8f626a8b4ee1e726eaad75f635174d147514c3aa8ccf3c8e89` |
| Label provenance | `backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json` | `5258f1247b982b371e96125aa122d58f8afae3738b3e24d4fa39cacc05ee709a` |
| Stats / optional historical KTC | `backend/data/training-data.zip`, member exactly `training-data.json` | archive `b67e7b03e600ae9ad83b5b179b416f273b7e169c8d47eff7345ef257cb034a86` |

The checked-in label sidecar identifies `baseline-hl28-v1`, `model_version_source: export`, export time `2026-10-24T11:39:31Z`, 3,560 joined rows and raw export hash `ea66510a8608eedf838e599bb3bbbc91e5bf45f483a2fb8ad8382e4db6e046b2`. The raw export is not an additional required local input. The implementation verifies the sidecar's label/training hashes and records its generations; it must not substitute another file with a similar name or a v3 target. Hash the selected ZIP member as well and record its computed identity in the report, alongside the pinned archive identity. Require exactly that member and the expected schema.

Use dynasty/superflex reference values on the **same season's board**, columns `start_value` and `end_value`. Do not use `end_value_next_board` or a daily top-asset normalization. The SQL recipe documents these dates and the sidecar documents these generations:

| Season | Start date | End date | Start / end generation (revision) |
| --- | --- | --- | --- |
| 2020 | 2020-09-10 | 2021-01-03 | 9381 / 9496 (3 / 3) |
| 2021 | 2021-09-09 | 2022-01-09 | 9719 / 9841 (3 / 3) |
| 2022 | 2022-09-08 | 2023-01-08 | 10026 / 10148 (3 / 3) |
| 2023 | 2023-09-07 | 2024-01-07 | 10350 / 10472 (3 / 3) |
| 2024 | 2024-09-05 | 2025-01-05 | 10665 / 10787 (3 / 3) |
| 2025 | 2025-09-04 | 2026-01-04 | 10983 / 11105 (3 / 3) |

Season identity governs the boundary: the January 2026 endpoint of the **2025 season** is permitted historical evidence; outcomes for **season 2026** are not. The label filename mentions 2019, but the joined labels and training archive contain only seasons 2020–2025. Provenance coverage metadata for 2019 is not an evaluable season.

For each eligible player-season, define:

```text
S = common Backchannel start_value; E = common Backchannel end_value
r = log(E / S)                     # primary realized movement
d = E - S                          # secondary absolute movement
r_hat = log(E_hat / S); d_hat = E_hat - S
```

Every primary model uses the same S, E and row keys. Primary loss is mean absolute log-change error with equal weight per player-season. Report percentages as `100 * expm1(r)` where an individual change needs interpretation; do not label mean absolute log error as an arithmetic percentage error. Absolute-change MAE equals end-value MAE on common starts and is secondary, not an independent corroborating result.

## Input validation, eligibility and coverage

The new loader reads only the three pinned historical inputs. Check the exact byte identities before parsing; do not provide configurable data paths or seasons. Unexpected identities, duplicate player-season keys before filtering, malformed schemas, infinities, negative market values/trade counts, nonintegral season or trade counts, unknown position codes, or rows outside 2020–2025 are integrity failures. Report them without fitting or writing a success report. Missing fields used by the contract are schema errors; present null observations are accounted-for exclusions as below. Player IDs are strings; joins must be one-to-one on `(player_id, season)` and output order is season ascending then player ID lexicographically.

Stats come from the archive, not duplicated CSV features: position, season age, full-PPR points (4-point passing TD), games and experience. Validate matching CSV position/age/games/points against the archive for all joined rows, using exact integer/string comparisons and absolute tolerance `1e-9` for floats. A conflict is an integrity error, not a silent choice. Finite negative fantasy points are legitimate and present in this archive; keep them. The existing model uses them during fit and clamps the prediction grid to zero for negative input, which must remain unchanged and be counted in reporting. Do not use archive auxiliary features such as team KTC, weekly KTC, point changes or position rank.

Define the following ledger populations for every season and position:

1. `archive_population`: QB/RB/TE/WR archive player-seasons in 2020–2025, including rows later excluded. Count pre-NFL rows using the existing convention `years_exp == 0 and games == 0`; retain rookies/veterans with genuine zero points or zero games otherwise. Unknown experience follows the archive's `-1` convention and is counted as unknown, not assumed pre-NFL. This is a retrospective eligibility convention, not proof of preseason NFL eligibility.
2. `available_label_population`: every row in the pinned label CSV, joined to archive keys with an audited one-to-one join. Record archive-only and label-only keys/counts separately.
3. `available_start_eligible`: among available label rows, known NFL-eligible stats, supported position, finite age and strictly positive S, and `start_trades_window >= 5`. Missing age/start/start-liquidity values are explicit reasons for exclusion. Age and position are archive attributes with the as-of limitation below. Points/end values/end liquidity do not decide this provisional starting group.
4. `primary_scored`: available-start-eligible rows with present finite season points, strictly positive E and `end_trades_window >= 5`. Zero points are valid. Missing points, missing/zero end value, and low/missing endpoint liquidity have distinct reasons. Use the same rule for training and scored test rows of both learned models.

Use a fixed first-failure reason order matching the list above for mutually exclusive attrition totals, plus independent reason flags so overlapping conditions are visible. Publish row counts, distinct players, position coverage, common cohort key hashes, start/end evidence and trade-count distributions. Publish scored/start-eligible coverage within the available-label population, training/test counts per fold, and optional KTC intersection attrition. No minimum test cohort may be silently dropped to improve scores.

The committed CSV was already built from a start/end board intersection and a stats join; its sidecar supplies aggregate upstream counts, not missing endpoint rows or a complete preseason roster. Therefore `full_preseason_population_count` and `full_preseason_coverage` are **null with reason `not_recoverable_from_historical_export`**. Archive rows absent from the CSV cannot be called endpoint dropouts: their start observability is unknown. Preserve the upstream aggregate counts/warnings as provenance, clearly separate from the new ledger. Do not claim a deployable coverage rate or full attrition audit from these files.

The positive endpoint rule also omits observed zero-value collapses from log scoring. Count them, show their absolute-value/no-change errors in a separately named observed-zero-end diagnostic when prediction inputs exist, and never replace zeros with an outcome-chosen epsilon. Missing end values cannot be scored. The `end_trades_window >= 5` restriction selects for future observability: the main result applies to liquid-at-both-endpoints observed player-seasons, not automatically to every preseason user query. Thin endpoint rows with positive values may be reported in a separate `thin_endpoint` diagnostic (0–4 trades), never pooled into the primary result.

## Models and matched comparison

Use the existing `WhatIfModel` as a read-only imported dependency, with its source SHA recorded and verified. The evaluator adds no runtime model and never saves a fitted joblib file. Model keys:

- `production`: four features in existing order `[log_start, age, points, position_code]`; positions `[QB, RB, TE, WR]`; target `log(E/S)`; scale cap 10,000; `HistGradientBoostingRegressor` parameters `loss=absolute_error`, `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `l2_regularization=1.0`, `random_state=0`; monotonic constraints `[0,0,1,0]`; existing one-point running-maximum correction at prediction. Preserve the effective pinned scikit-learn defaults and record them (including effective early stopping: auto is inactive for these small training folds).
- `no_production`: instantiate exactly the same model and pass a copied frame with `points=0.0` in **both fit and prediction**, preserving all other columns, rows, target, hyperparameters, cap and correction. The constant feature carries no production information; the points grid degenerates to zero. This matched ablation asks what points add to this architecture. It does not compare against a different tuned model, remove low-production players, condition training rows on point magnitude, shuffle future points, or use average test-season points.
- `no_change`: E_hat=S, r_hat=0 and d_hat=0, on the identical cohort.

Both learned models train on exactly the same ordered keys and predict exactly the same ordered held-out keys. Any failed or invalid prediction fails the fold/run; never salvage a comparison by dropping rows for one model. Count endpoints at the scale cap as `at_cap_count` (output saturation, not proof of a strict clipping event). A prediction must be strictly positive and finite to enter log metrics; no hidden metric floor is allowed. Catch numerical warnings/errors from prediction rather than silently accepting an overflow that a later cap could hide. Unclipped log predictions are optional diagnostics only if available without changing protected model code; they must not be reverse-engineered from clipped values.

The primary contrast is `MAE_log(no_production) - MAE_log(production)`, positive when production helps. Also report `1 - MAE_log(production)/MAE_log(no_production)` when its denominator is positive. No-change skill remains the separate absolute-usefulness comparator. Same architecture is necessary but not proof of a causal production effect.

## Chronological evaluation and leakage controls

The fixed v1 expanding folds are:

| Test season | Training seasons |
| --- | --- |
| 2021 | 2020 |
| 2022 | 2020–2021 |
| 2023 | 2020–2022 |
| 2024 | 2020–2023 |
| 2025 | 2020–2024 |

Require nonempty eligible training rows in every listed season and nonempty primary test rows in every fold. Missing required coverage fails with an explicit reason; never shorten windows or omit a difficult season. Tag the one-season 2021 training fold as `short_history`. Empty individual position/slice cells remain present with null metrics. Repeated players across seasons are allowed because the product forecasts returning players; this is temporal generalization, not a test on unseen player identities.

No hyperparameter/window/model selection occurs in v1. The fold fits and all preprocessing receive only seasons earlier than the test season; test endpoints are unavailable to fitting, feature construction and threshold fitting. All models use the same train-derived value-tier boundaries and reporting masks. The fixed rules do not inspect fold scores to change thresholds, target, cohort or model. Future tuning requires a new development version with inner rolling-origin validation wholly inside each outer training window; never reuse outer-season scores as a selection signal while calling them untouched.

Controls and honest limits:

- Read realized production only as the expressly conditioned-on scenario input. Games/experience affect the documented eligibility convention or descriptive coverage, never additional regression features.
- End values and end liquidity may determine label observability and retrospective slices, never prediction features, start-value tiers, optional cross-source calibration, model selection or preprocessing.
- Compute value tiers only from eligible training S within position: 1/3 and 2/3 empirical quantiles, NumPy `method="linear"`. Apply `S <= q1`, `q1 < S <= q2`, `S > q2` to test starts; preserve collapsed/empty tiers. No test quantiles.
- Preserve fixed reference values; do not normalize a season by its end board, ranks or future top asset. Publish start/end medians, median log movement and liquidity distributions per season to expose scale drift. They are target diagnostics, not corrections.
- Sidecar hashes and generation revisions establish identity, **not historical availability**. Retrospectively regenerated valuations may use revised mappings, priors, eligibility or anchors; archive position/age conventions and alignment of full-season points with final-Sunday timestamps also require provenance beyond these files. Do not assert these were audited when they were not. Report `historical_asof_validity: unverified` with specific limits. A later independent upstream audit must establish trade/priors/normalization cutoffs before claiming realistic as-of predictive validity.
- Endpoint board date and full-season points may refer to different intraday availability. No intraday timestamp evidence is present here. Record that timing limit; do not fabricate midnight/as-of precision.
- The historical loader never reads current starts, current market boards, unregistered input files, registries' outcome paths, or any season-2026 values. It has no database/client credentials or outcome discovery mechanism.

## Metrics and uncertainty

Report identical metric definitions for all three primary models on all matched cohorts. Primary headline is pooled equal-player-season log MAE and its paired improvement over no-production, plus positive skill against no change. Also publish each fold and equal-season macro summaries so a larger recent season cannot hide a poor regime. Pooled skill is a ratio of pooled MAEs, never an average of row-level skills; macro mean of fold skills is separately named and counts undefined folds.

| Family | Exact reporting convention |
| --- | --- |
| Change error / skill | MAE, RMSE, median and p90 absolute error on r; `skill_log=1-MAE_log/mean(abs(r))`; value-unit MAE/RMSE/median/p90 on d and `skill_value=1-MAE_value/mean(abs(d))`. Baseline denominator zero yields null with `zero_baseline_error`, not zero or infinity. Quantiles use linear interpolation. |
| Direction | Flat when `abs(r) <= log(1.10)`, up above the band, down below it, using the same band for predictions. This is a symmetric multiplicative convention (down boundary approximately -9.09%), proposed for reporting, not a validated economic threshold. Report 3x3 counts, overall accuracy, class recalls, balanced accuracy over actual supported classes with support count, and accuracy on actual non-flat movers (flat predictions are incorrect). Absent classes yield null recall. Compare both with no-change and a train-only majority-class classifier (ties: flat, down, up). |
| Bias / calibration | Mean signed `(r_hat-r)` and `(d_hat-d)`, mean predicted and actual movement, and OLS `r = intercept + slope*r_hat` as a held-out descriptive diagnostic. Slope is null for constant predictions; no recalibration is fitted to test outcomes. Fixed bins in log space have internal edges `[-log(2), -log(1.25), -log(1.10), 0, log(1.10), log(1.25), log(2)]`, outer unbounded bins, lower inclusive/upper exclusive. Record n, mean prediction, mean actual and their difference in every bin, including empty bins. |
| Movement ranking | Spearman between r_hat and r, with average ranks for ties, overall and within season-position; constant vectors / fewer than two rows yield null. Do not substitute end-value rank correlation. Within each season-position, `k=max(1,ceil(0.10*n))`; report predicted/actual top-k and bottom-k overlap divided by k and mean actual movement of each predicted selection. Break selection ties by player ID; report tie-at-boundary counts and abstain/null on a completely constant prediction vector. |
| Large moves / tails | Actual `abs(r) >= log(1.50)` defines a retrospective large-move slice; report its coverage, change error, bias and direction. Also report positive/negative large-mover recall, precision (no predicted positives => null), top/bottom metrics above, all-row p90 and RMSE, and the 20 largest absolute log errors per model sorted by error descending then season/player ID. These outcome-defined slices are diagnostic, never eligibility rules. |
| Coverage / robustness | Overall, season, position, season-position, start-value tier within position, position x tier, and start liquidity 5–9 / 10–24 / 25+; endpoint liquidity in the same bands separately as retrospective diagnostics. Include counts, distinct players, exclusions, age/games/points distributions and prediction clipping counts. Report n<30 cells with `low_support:true`; 30 is a descriptive support flag, not a power claim or a promotion threshold. |

No-change direction is always flat and its mover hit rate is zero where movers exist; its majority-class advantage must not be mistaken for movement skill. The training-majority classifier is direction-only, not another end-value regressor.

For paired comparisons `production - no_production` and `production - no_change`, retain aligned row predictions and recompute statistics on each common resample. Report paired MAE/RMSE differences (negative means production lower error), skill gains (positive better), direction-accuracy differences and within-season-position movement-Spearman differences. For a rank aggregate, use an equal-weight arithmetic mean of correlations over cells defined for both compared models, and report the number and identities of included cells; this is separately named from pooled-row Spearman. Per-position rank aggregates use that position's season cells. No-change has constant predictions, so its rank contrast is null. Keep the positive-improvement headline explicitly named to avoid sign ambiguity.

Use 2,000 deterministic resamples, RNG seed 20261025, percentile 2.5/97.5 intervals and sorted keys. For per-fold intervals, resample distinct player IDs with replacement and carry their rows together; pooled intervals cluster by player ID across all seasons to preserve repeated-player dependence. These intervals condition on observed seasons and do not capture common market shocks. Separately resample the five entire test-season blocks for pooled error/skill/accuracy contrasts, show all five fold estimates and leave-one-test-season-out aggregate contrasts. The season bootstrap has only five blocks and is explicitly exploratory, not a reliable new-season guarantee. Do not present either one-way bootstrap as accounting simultaneously for players, teams, seasons and leagues. Missing team/league identifiers preclude those clusters. Use identical draws for paired models; do not refit models during this uncertainty calculation.

Report the effective valid resample count and null reason for undefined statistics (especially rank/direction cells); suppress an interval if fewer than 90% of draws define it. Compute uncertainty for headline metrics overall, each fold and each position; detailed intersections/tail lists may remain point estimates with support counts. Family-wide multiplicity is not controlled: slices are exploratory, and no significant slice may be selected to override the primary contrast. Point-model interval coverage or calibrated predictive intervals are not claimed; that needs a separate uncertainty model.

## Optional KTC diagnostic

Default `--ktc-diagnostic off`. Optional `archive` mode uses `start_ktc` and `end_ktc` from the same pinned training archive, positive finite values at most 9,999, under identical temporal folds. Existing `weekly_ktc` fields contain dated summaries, but the archive does not supply a verified raw observation/export/mapping chain comparable to the sealed future raw-target recipe. Do not query historical or current databases to fill this gap. A provenance-equivalent raw KTC target is deferred and does not block the main evaluator.

Archive mode reports `provenance_limited:true` and `status:diagnostic_only`. Fit the same points-conditioned architecture on a **training intersection** of primary Backchannel-eligible rows and valid archive KTC labels in the same seasons. For a controlled diagnostic, refit Backchannel production and no-production models on that same training intersection; score them, KTC and no-change on one common test intersection. These refits are separately named and cannot replace the main models or primary scores. Report every additional row loss. If an intersection leaves a required train season/test fold empty, mark that optional fold unavailable and continue the complete primary report.

The KTC model uses its own source S and cap 9,999, then its predicted ratio `E_hat_ktc/S_ktc` is applied to the shared Backchannel S. Score on Backchannel E with the same formulas; mark the ratio transplant as an uncalibrated cross-market comparability assumption. Do not cap the transplanted Backchannel endpoint a second time or silently learn a test-season scale mapping. An optional cross-target matrix may also score the same ratios against archive KTC endpoints on the identical test keys, but every cell must state its target/start/baseline. Cross-target errors and skills must never be subtracted as if they measured the same outcome. KTC superiority is neither necessary nor sufficient for the production-information conclusion, and KTC unavailability has no effect on primary interpretation.

## Interpretation and acceptance

There is no `promote`, `replace`, `retain_ktc`, `winner` or automatic acceptance action. The report always contains `evidence_class: historical_development`, `promotion_authorized:false`, `independent_confirmation_complete:false`, `2025_status:reused_development`, and the explicit conditional-on-realized-points and target-measurement limitations.

The evaluator reports mechanical evidence flags, not a product-release verdict:

- `incremental_production_point_estimate`: improved / tied / worse from the sign of pooled log-MAE reduction against the matched ablation.
- `no_change_point_estimate`: improved / tied / worse / undefined from shared-target no-change log MAE.
- Paired player/season intervals, every fold and position contrast, and counts of improved folds/positions, without inventing a universal p-value or margin.
- Coverage, timing/provenance, zero-collapse omissions, bias/calibration, direction and tail-risk warnings, each with supporting values.

Positive incremental improvement and positive no-change skill are the minimum evidence for the intended claim. A full model beating no change while losing to the no-production ablation does **not** establish added production signal. Mixed or small improvements, intervals spanning zero, season reversals, poor calibration, tail deterioration, low coverage or unresolved as-of validity call for qualified/inconclusive development findings. RMSE/direction/calibration/ranking are separate dimensions; do not hide their regressions behind one MAE number. Statistical intervals from reused historical evidence cannot certify an untouched result.

No arbitrary numeric practical-effect, calibration, noninferiority or coverage threshold is inferred from KTC or borrowed from either sealed protocol. Practical tolerances require a product-utility decision and future confirmatory design before viewing its outcomes. The evaluator can be implemented and reviewed now without that future release policy. Neither good scores nor a previously passing historical gate authorize shipping changes.

## Isolated implementation surface and CLI

Add a new package only:

```text
backend/predictor_v5/movement_evaluation/
  __init__.py          # no imports with side effects
  __main__.py          # dedicated historical CLI
  contract.py          # pinned inputs, folds, metric conventions
  data.py              # strict historical reader + ledger
  models.py            # read-only WhatIfModel adapter + constant-points ablation
  metrics.py           # matched change metrics + uncertainty
  report.py            # canonical serialization/publication
backend/predictor_v5/tests/test_movement_*.py
```

Do not add a subcommand to protected `predictor_v5/__main__.py`. Proposed command from `backend`:

```bash
OMP_NUM_THREADS=1 python -m predictor_v5.movement_evaluation
OMP_NUM_THREADS=1 python -m predictor_v5.movement_evaluation --ktc-diagnostic archive
```

Only `--ktc-diagnostic {off,archive}` and `--help` are public v1 arguments. No season/data-path/model/threshold/output-path overrides, registry flags, 2026 mode, training/save/promotion action, network access or runtime side effects. Pin dependency versions to the repository requirements and record Python, NumPy, pandas, sklearn and scipy versions plus thread configuration; differences make distinct run identities. Private pure functions accept in-memory synthetic frames for tests, never arbitrary public filesystem paths.

The runner validates input identities and contract, builds folds, fits ephemeral models, scores and publishes one deterministic JSON development report. Exit 0 means the evaluator produced a valid report (including optional diagnostic unavailability), not positive model quality. Exit 2 means contract/input/cohort refusal; exit 1 means unexpected fitting/numerical/I/O failure. No partial success report on failure. Historical reruns are allowed and do not consume a holdout or create claims.

## Report identity and schema

Output path: `backend/predictor_v5/artifacts/market_movement/development-2020-2025-<run-sha256>.json`, a new namespace. No writes under `confirmations/`, `replacements/`, `data/`, existing artifacts or runtime directories. The runner may create its new regular output directory; reject symlinks, nonregular destinations and any resolved path outside that namespace.

`run-sha256` is SHA-256 of canonical UTF-8 JSON containing protocol `market-movement-development`, schema 1, input identities, package/dependency source identities, exact folds/models/metrics/seed/diagnostic mode and runtime-library versions/thread configuration. Use sorted object keys, compact separators, `allow_nan=False`, one final LF, sorted arrays wherever order is not semantic, and full-precision finite numeric metrics. Record source file hashes rather than a working branch name or wall-clock time. Git base/provenance commit is fixed contract metadata; do not include a changing current HEAD or timestamp that makes an otherwise identical rerun differ. Store the complete run-identity payload so its hash can be recomputed.

The report has these required top-level fields:

```text
schema_version, protocol, run_identity, run_sha256
evidence {class, historical_exposure, 2025_status, conditional_points,
          historical_asof_validity, measurement_limits,
          promotion_authorized, independent_confirmation_complete}
target {id, formula, scale, columns, dates, generations, input_identities}
contract {models, folds, eligibility, metrics, bins, bootstrap, ktc_mode}
coverage {upstream_provenance, ledger, reasons, key_hashes,
          full_preseason_population_count, full_preseason_coverage}
folds [{test_season, train_seasons, train_key_hash, test_key_hash,
        n_train, n_test, model_parameters, clipping, slices, contrasts}]
pooled {models, contrasts, macro_by_season, uncertainty,
        leave_one_season_out, slices}
predictions [{season, player_id, position, S, E, r, points, age,
              start_trades, end_trades, value_tier,
              production_E_hat, no_production_E_hat, no_change_E_hat}]
optional_ktc {status, provenance_limits, cohorts, fits, scores, predictions}
interpretation {evidence_flags, caveats}
```

Metric cells contain `n`, `n_players`, relevant supports and nullable values with explicit reasons, not NaN/Infinity or omitted empty slices. Prediction rows expose the common endpoints to enable independent recomputation; full raw archive records and player names are unnecessary. Include fixed-bin definitions and train-derived tier edges per fold. `predictions` are held-out only, once per model/key; optional diagnostic predictions are separately keyed. No trained estimators or raw future evidence are serialized.

Publish without replacing existing content. If the exact final bytes already exist at the identity path, verify equality and exit successfully; different bytes at the same identity are a determinism/integrity error. Use a staged regular file and exclusive final publication; remove only the writer's temporary file on failure. Do not mutate the old report or create success evidence before completion. Tests compare byte-identical reports under a fixed supported environment. Across dependency/hardware differences, source and runtime metadata delimit the reproducibility claim; unexpected same-identity numerical differences fail loudly.

## Verification plan for eventual implementation

Use synthetic fixtures and the permitted historical inputs only. Do not run sealed evaluators as a regression-test shortcut. Required checks:

1. Strict loading: pinned hashes/member/schema; duplicate/unknown/out-of-range years fail before fitting; present missing observations produce ledger exclusions; negative/infinite/malformed observations refuse; zero points remain; pre-NFL, true zero endpoints, missing labels and endpoint-liquidity cases are distinguished; archive/CSV feature conflicts refuse. Fixtures with 2026 keys refuse without processing them into features.
2. Temporal isolation: a fit spy proves each model sees only its exact listed train keys; changing a test endpoint cannot change fitted model/predictions, tier edges or majority classifier; changing test liquidity can change the scored mask but not training. A future historical fold's rows cannot affect an earlier fold's outputs. Whole report identity may change in synthetic input tests without confusing this invariance.
3. Ablation: exact shared ordered train/test keys and estimator parameters; points are constant during both ablation fit and predict; changing only held-out points never changes ablation/no-change predictions; production uses the existing monotone-grid behavior. Target, age/position/start and all nonproduction features remain matched. Do not demand production improve on arbitrary real data to make tests pass.
4. Metrics: hand-computed movements/skills and paired signs; identical delta/end-value errors; zero baseline denominator; empty/constant rank/calibration; flat-band boundaries and supported-class balanced accuracy; ties/top-k; train-only tiers; large moves; clipping; multi-season pooling versus macro means; null serialization and reason codes.
5. Uncertainty: deterministic seeds/draws, paired model sampling, repeated-player clusters and whole-season block sampling, undefined-draw handling, no refits, and correct counts. Synthetic data tests statistical bookkeeping, not guaranteed nominal coverage under five market seasons.
6. Optional KTC: off mode never accesses KTC fields; absence/limited intersection cannot change the primary result; diagnostic refits share train/test keys; ratio transfer uses source starts and common target endpoints with no second cap; differing targets are labeled and never compared by raw cross-target subtraction.
7. Isolation/output: CLI rejects overrides and 2026; no database/network/outcome discovery; spy filesystem access allowlists the pinned inputs, source identity files and new output namespace. Assert no registry/claim/model/runtime writes. Verify both existing sealed protected-byte sets before and after, without invoking lifecycle methods. Byte-stable report reruns, output-collision refusal, symlink refusal and atomic failure cleanup use temporary directories.
8. One historical smoke evaluation after implementation is separately authorized: verify all five folds, complete ledger/slices and recompute primary metrics independently from emitted rows. Report findings honestly without a required favorable score. No real evaluation is part of this design-only commit.

## Preserving a later independent season

2026 outcomes remain unopened by this work. The two existing sealed protocols retain their original hypotheses, source hashes, decisions and PR #8 runbook; this evaluator does not replace, amend or reinterpret them. Their later coordinated review/registration/execution order and no-retry rules remain intact.

The new historical CLI cannot accept 2026. Testing the **new production-ablation hypothesis** on 2026 later requires a separate reviewed prospective contract, with its model/cohort/target/metrics and practical tolerances frozen before outcome exposure, and an honest account of all interim 2026 information already observed by humans or other tooling. Being designed midseason is not preseason preregistration. A separate registration and consent are required then; this document creates neither.

If a future new contract and both old sealed protocols will share 2026 evidence, their independent review must coordinate exposure **before any of them opens outcomes**, without changing existing contracts. The frozen models/analysis for the new question must not be chosen after seeing either sealed result. Reading a shared outcome for any protocol consumes its untouched status for later adaptive development. If 2026 information has already influenced this new hypothesis/configuration, or its exposure history cannot support independence, label it development and reserve a genuinely later untouched season instead. Do not promise 2026 independence based solely on software hashes or this design date.

## Remaining questions and deferred decisions

No unresolved product choice blocks implementing this development evaluator as specified. Independent historical as-of provenance, a full preseason population/endpoint attrition dataset, raw comparable historical KTC evidence, a business-meaningful movement/no-change band and practical release tolerances, predictive intervals, and a preseason points-forecast pipeline are explicitly unresolved/deferred. They limit interpretation or future promotion; they are not excuses to edit protected files, fetch 2026 outcomes or invent unavailable evidence.
