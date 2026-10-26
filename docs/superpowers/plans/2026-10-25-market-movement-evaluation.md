# Production-Conditioned Market-Movement Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, deterministic historical evaluator that measures whether realized season production improves prediction of same-season Backchannel market movement beyond start value, age, and position.

**Architecture:** A new `predictor_v5.movement_evaluation` package reads only three pinned 2020–2025 historical inputs, constructs one audited MFL-inclusive Backchannel cohort, runs five fixed expanding folds through the existing read-only `WhatIfModel`, and scores production, constant-points ablation, and no-change predictions on identical rows. Pure data/model/metric functions feed a canonical report builder and no-replace publisher; a dedicated CLI orchestrates the pipeline without touching runtime, sealed protocols, or 2026 outcomes.

**Tech Stack:** Python 3.12, pandas 2.2.0, NumPy 1.26.3, scikit-learn 1.4.0, SciPy as installed through the pinned scikit-learn environment, pytest, Ruff, SHA-256, canonical JSON.

**Spec:** `docs/superpowers/specs/2026-10-25-market-movement-evaluation-design.md`

## Global Constraints

- Treat `c4270804379c26bcb340f32a07e507153b4c7342` as the fixed provenance/base commit and `00bf51b0e8a2f564cf673f53df697ae772c27a0c` as the clean design-only starting HEAD; never put the changing implementation HEAD or wall-clock time into run identity.
- Add only `backend/predictor_v5/movement_evaluation/*.py`, `backend/predictor_v5/tests/test_movement_*.py`, and the one authorized report under `backend/predictor_v5/artifacts/market_movement/`; do not modify any existing source, data, manifest, registry, artifact, SQL recipe, runtime configuration, API, UI, deployment, or season behavior.
- The protected set is the exact path list in `/private/tmp/ktc-movement-baseline.md`, including both sealed protocols' bound sources and direct inputs. Prove every protected path byte-equal to the fixed base before and after the historical run; do not regenerate a seal or import/invoke either lifecycle runner.
- Read no season-2026 outcome, current board, starts snapshot, registry outcome path, database, network service, or unregistered file. A January 2026 endpoint belonging to season 2025 is permitted; a row whose `season` is 2026 is an integrity refusal before feature construction.
- Use target `backchannel-mfl-v1-reference-same-season-log-movement-2020-2025-v1` from `backchannel_labels_2019_2025_mfl.csv`; do not substitute the v3 target, `end_value_next_board`, KTC, or a normalized/ranked target.
- Pin and verify: labels SHA-256 `3e57be0b6e6ccb8f626a8b4ee1e726eaad75f635174d147514c3aa8ccf3c8e89`; provenance SHA-256 `5258f1247b982b371e96125aa122d58f8afae3738b3e24d4fa39cacc05ee709a`; archive SHA-256 `b67e7b03e600ae9ad83b5b179b416f273b7e169c8d47eff7345ef257cb034a86`; sole member `training-data.json`; member SHA-256 `6dfa1e3eaed486d5a37880217c74fb3b139f19504aedb58b39a38013466a5388`; protected `model.py` SHA-256 `01674fa50df69a8e27f221f46eafa2e55b459df9e74c2c2357d6e03e733cade1`.
- Derive position, season age, full-PPR/4-point-passing-TD points, games, and experience from the archive. Compare the CSV copies on every joined row, retain finite negative points and genuine zero-point/zero-game NFL seasons, and exclude only the documented `years_exp == 0 and games == 0` pre-NFL convention.
- Use fixed folds only: test 2021/train 2020; 2022/2020–2021; 2023/2020–2022; 2024/2020–2023; 2025/2020–2024. Require every train season and test fold to be nonempty and tag 2021 `short_history`; 2025 remains reused development evidence and 2026 remains reserved.
- Instantiate the existing `WhatIfModel` read-only with scale 10,000 for both learned models. Production uses realized points. No-production uses a copied frame with `points=0.0` for fit and predict, with identical keys, target, parameters, cap, and monotone correction. No-change predicts `E_hat=S` on those same test keys.
- End values and end liquidity may determine retrospective observability/slices only; they must never affect features, tier edges, majority direction, model selection, preprocessing, or calibration. Value tiers use only position-specific training starts and NumPy linear 1/3 and 2/3 quantiles.
- Primary output is pooled equal-player-season log-change MAE and the positive-improvement contrast `MAE_log(no_production) - MAE_log(production)`. Also report no-change, fold/macro results, all specified metrics/slices, paired uncertainty, honest coverage, zero-end/thin-endpoint diagnostics, and provenance/timing limits without adding an acceptance threshold or release verdict.
- KTC is optional diagnostic-only behavior. Default `--ktc-diagnostic off` must not access KTC fields; archive mode uses only archive `start_ktc`/`end_ktc`, refits all learned comparisons on the same KTC intersection, reports every row loss, and cannot alter or gate the primary result.
- Serialize sorted canonical UTF-8 JSON with compact separators, `allow_nan=False`, full finite precision, and one final LF. Publish only in the new namespace, never replace differing content, accept byte-identical reruns, reject symlinks/nonregular paths/escapes, and remove only the writer-owned temporary file on failure.
- Always report `evidence_class: historical_development`, `2025_status: reused_development`, `historical_asof_validity: unverified`, `promotion_authorized: false`, `independent_confirmation_complete: false`, conditional-on-realized-points semantics, incomplete preseason coverage, measurement/provenance/timing limits, and exploratory/multiplicity caveats.
- The design authorizes exactly one historical smoke evaluation after implementation and review (Verification Plan item 8). Run the default KTC-off evaluator once, verify it independently, and commit its deterministic JSON. Do not run either sealed evaluator and do not require favorable results.

## Review Focus

- Malformed, duplicate, nonfinite, negative-market/trade-count, unknown-position, out-of-range, or season-2026 rows must refuse before fitting or report publication; Task 2 pins this with `test_invalid_rows_refuse_before_feature_construction` and related parametrized loader tests.
- Cohort drift between production, no-production, no-change, or optional KTC models must fail rather than silently drop rows; Tasks 4 and 6 pin ordered train/test key equality and invalid-prediction refusal.
- Test-season endpoints/liquidity, later folds, and held-out points for the ablation must not leak into fitting, tier thresholds, or train-majority direction; Task 4 uses fit spies and mutation invariance tests.
- Empty/constant cells, zero baselines, undefined correlations/precision, and bootstrap draws with undefined statistics must emit explicit null reasons and never NaN/Infinity; Tasks 3 and 5 cover every case and Task 7 rejects noncanonical values.
- Protected bytes and output durability are the highest operational risk: Task 8 compares protected files to the fixed base and enforces the read/write allowlists; Task 7 exercises symlink, collision, idempotence, and atomic-cleanup behavior.

## File Map

- Create `backend/predictor_v5/movement_evaluation/__init__.py`: side-effect-free package marker only.
- Create `backend/predictor_v5/movement_evaluation/contract.py`: immutable protocol constants, exact inputs/hashes, folds, dates/generations, model/metric/bootstrap settings, typed refusal, and deterministic key hashing.
- Create `backend/predictor_v5/movement_evaluation/data.py`: pinned byte loading, strict archive/CSV/provenance validation, one-to-one reconciliation, eligibility ledger, common primary cohort, and zero/thin endpoint diagnostics.
- Create `backend/predictor_v5/movement_evaluation/metrics.py`: metric primitives, direction/calibration/rank/tail/slice summaries, matched contrasts, macro aggregation, clustered bootstrap, and leave-one-season-out estimates.
- Create `backend/predictor_v5/movement_evaluation/models.py`: read-only `WhatIfModel` adapter, fold preparation, constant-points ablation, no-change predictions, train-derived tiers/majority class, and optional KTC refits/ratio transplant.
- Create `backend/predictor_v5/movement_evaluation/report.py`: report assembly, runtime/source identities, canonical serialization, run hash, safe output-path derivation, and atomic idempotent publication.
- Create `backend/predictor_v5/movement_evaluation/__main__.py`: exact two-mode CLI, staged orchestration, exit-code mapping, and no other public options.
- Create `backend/predictor_v5/tests/test_movement_contract.py`: contract, folds, identity, key hashing, and protected-base invariants.
- Create `backend/predictor_v5/tests/test_movement_data.py`: strict parsing, duplicate/schema/value conflicts, eligibility, ledger, coverage, and 2026 refusal.
- Create `backend/predictor_v5/tests/test_movement_metrics.py`: every point metric, slice, null-reason, ranking, direction, calibration, and contrast convention.
- Create `backend/predictor_v5/tests/test_movement_models.py`: common-row model behavior, ablation, fixed folds, prediction validation, tier/majority leakage tests, and diagnostics.
- Create `backend/predictor_v5/tests/test_movement_uncertainty.py`: deterministic paired player and season-block resampling, valid-draw suppression, and leave-one-season-out checks.
- Create `backend/predictor_v5/tests/test_movement_ktc.py`: off-mode isolation and archive-mode common-intersection/refit/ratio-transfer behavior.
- Create `backend/predictor_v5/tests/test_movement_report.py`: schema, deterministic bytes, run identity, idempotent no-replace publication, and filesystem safety.
- Create `backend/predictor_v5/tests/test_movement_cli.py`: CLI grammar, orchestration/exit codes, access allowlist, protected-file preservation, and no partial success.
- Create once after all code is committed and verified: `backend/predictor_v5/artifacts/market_movement/development-2020-2025-<run-sha256>.json`.

---

### Task 1: Freeze the standalone protocol contract

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/__init__.py`
- Create: `backend/predictor_v5/movement_evaluation/contract.py`
- Create: `backend/predictor_v5/tests/test_movement_contract.py`

**Interfaces:**
- Consumes: design constants and fixed base `c4270804379c26bcb340f32a07e507153b4c7342`.
- Produces: `ContractRefusal(ValueError)`; `KtcMode = Literal["off", "archive"]`; immutable `FoldSpec(test_season: int, train_seasons: tuple[int, ...], short_history: bool)`; `FOLDS: tuple[FoldSpec, ...]`; exact input/model identities; `contract_payload(mode: KtcMode) -> dict[str, object]`; `validate_ktc_mode(value: str) -> KtcMode`; `sorted_keys(frame: pd.DataFrame) -> tuple[tuple[int, str], ...]`; `key_sha256(keys: Iterable[tuple[int, str]]) -> str`.

- [ ] **Step 1: Write failing contract tests**

Add tests asserting:

```python
assert [(f.test_season, f.train_seasons) for f in FOLDS] == [
    (2021, (2020,)),
    (2022, (2020, 2021)),
    (2023, (2020, 2021, 2022)),
    (2024, (2020, 2021, 2022, 2023)),
    (2025, (2020, 2021, 2022, 2023, 2024)),
]
assert FOLDS[0].short_history is True
assert all(not fold.short_history for fold in FOLDS[1:])
assert contract_payload("off")["target"]["id"] == TARGET_ID
assert contract_payload("off")["bootstrap"] == {
    "seed": 20261025, "resamples": 2000, "percentiles": [2.5, 97.5]
}
```

Also assert exact dates/generations, positions `("QB", "RB", "TE", "WR")`, scale 10,000, flat/large-move bands, fixed calibration-bin edges, model parameters/features/monotonic constraints, the five input/source hashes from Global Constraints, and that `validate_ktc_mode` rejects every value except `off`/`archive` (including booleans and `2026`). Test that differently ordered identical keys yield the same hash, while duplicate keys raise `ContractRefusal`.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run from `backend`:

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_contract.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'predictor_v5.movement_evaluation'`.

- [ ] **Step 3: Implement the minimal immutable contract**

Keep `__init__.py` empty except for a docstring. In `contract.py`, expose only immutable tuples/mappings or newly built payload dictionaries. `key_sha256` must hash canonical sorted `[season, player_id]` pairs, must not accept season 2026, and must never infer folds from data. Include the exact model defaults in the payload, including effective `early_stopping="auto"` and the statement that it is inactive for these small folds.

- [ ] **Step 4: Run contract tests and lint**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_contract.py -q
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m ruff check --no-cache predictor_v5/movement_evaluation predictor_v5/tests/test_movement_contract.py
```

Expected: all contract tests PASS; Ruff reports no errors.

- [ ] **Step 5: Commit the protocol skeleton**

```bash
git add backend/predictor_v5/movement_evaluation/__init__.py backend/predictor_v5/movement_evaluation/contract.py backend/predictor_v5/tests/test_movement_contract.py
git commit -m "feat: freeze movement evaluation contract"
```

### Task 2: Build the strict historical loader and eligibility ledger

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/data.py`
- Create: `backend/predictor_v5/tests/test_movement_data.py`

**Interfaces:**
- Consumes: `ContractRefusal`, exact paths/hashes/seasons/positions from Task 1.
- Produces: `HistoricalBundle(archive: pd.DataFrame, labels: pd.DataFrame, provenance: dict[str, object], identities: dict[str, str])`; `EvaluationData(joined: pd.DataFrame, primary_scored: pd.DataFrame, observed_zero_end: pd.DataFrame, thin_endpoint: pd.DataFrame, coverage: dict[str, object])`; `load_pinned_bundle(repo_root: Path, *, ktc_mode: KtcMode) -> HistoricalBundle`; `parse_archive_bytes(raw: bytes, *, include_ktc: bool) -> tuple[pd.DataFrame, str]`; `parse_label_bytes(raw: bytes) -> pd.DataFrame`; `prepare_evaluation_data(bundle: HistoricalBundle) -> EvaluationData`.

- [ ] **Step 1: Write synthetic parsing and integrity tests**

Create in-memory archive/CSV/provenance fixtures with all six permitted seasons and tests that assert:

- exact ZIP bytes, sole member name, member hash, CSV/provenance hashes, and sidecar cross-hashes are checked before parsing;
- duplicate JSON object keys, duplicate archive player-season keys, duplicate CSV player-season keys, missing/extra physical CSV columns, wrong member/envelope, label-only keys, and archive-only keys are handled exactly as designed;
- malformed/nonintegral season or trade counts, negative market values/trade counts, infinities, unknown positions, and years outside 2020–2025 refuse before `prepare_evaluation_data` can call any feature/model spy;
- finite negative points and zero points survive; `years_exp == 0 and games == 0` is pre-NFL, `years_exp == -1` is unknown, and rookies/veterans with zero games remain eligible;
- CSV position/games match archive exactly and CSV age/points match with absolute tolerance `1e-9`; every mismatch refuses even if the row would later be excluded;
- `parse_archive_bytes(..., include_ktc=False)` does not access `start_ktc`, `end_ktc`, `weekly_ktc`, team values, ranks, or auxiliary features.

- [ ] **Step 2: Run loader tests and verify they fail**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_data.py -q
```

Expected: FAIL because `movement_evaluation.data` and its dataclasses/functions do not exist.

- [ ] **Step 3: Implement captured-byte validation and strict parsers**

Read only the three fixed repository paths through `load_pinned_bundle`; verify `lstat`/regular-file/no-symlink and SHA-256 before parsing captured bytes. Parse JSON with duplicate-key rejection, require the `players` envelope and the exact `training-data.json` ZIP membership, normalize `player_id` to nonempty strings, and sort every frame by season then lexicographic player ID. Off mode extracts only core fields; archive mode additionally validates finite `start_ktc`/`end_ktc` observations with `0 < value <= 9999` for diagnostic eligibility without making them primary fields.

- [ ] **Step 4: Write failing ledger/cohort tests**

Use overlapping failures to assert both fixed first-failure totals and independent flags. Pin the stage order:

```text
not_joined_to_archive -> pre_nfl -> missing_age -> missing_start
-> nonpositive_start -> missing_start_liquidity -> low_start_liquidity
-> missing_points -> missing_end -> zero_end -> missing_end_liquidity
-> low_end_liquidity
```

Unknown experience is counted/described independently and is never assumed pre-NFL; it is not a first-failure exclusion by itself. Unsupported/unknown position codes and negative end/start values are integrity errors, not attrition reasons. Add `test_common_primary_cohort_is_sorted_and_one_to_one`, `test_missing_nonfinite_duplicate_and_2026_rows_refuse`, `test_zero_end_and_thin_endpoint_are_diagnostics_only`, and `test_full_preseason_coverage_is_explicitly_unrecoverable` asserting:

```python
assert result.coverage["full_preseason_population_count"] == {
    "value": None, "reason": "not_recoverable_from_historical_export"
}
assert result.coverage["full_preseason_coverage"] == {
    "value": None, "reason": "not_recoverable_from_historical_export"
}
assert list(result.primary_scored[["season", "player_id"]].itertuples(index=False, name=None)) == expected_keys
```

Require coverage by season and position for archive, available labels, start-eligible, and primary-scored populations; distinct players; archive-only/label-only key hashes and counts; start/end evidence and trade-count distributions; scored/start-eligible rates only within available labels; source attrition flags; negative-point count; observed-zero-end absolute/no-change inputs; thin positive endpoints kept separate. Preserve sidecar aggregate coverage/warnings under `upstream_provenance`, never reinterpret absent CSV rows as endpoint dropouts.

- [ ] **Step 5: Implement `prepare_evaluation_data` and run tests**

Build an audited one-to-one outer reconciliation, then the fixed eligibility ledger and common `primary_scored` frame. Do not impute missing outcome fields, floor endpoints, use next-board values, or add endpoint-dependent features. Return diagnostic frames separately and retain enough start-eligible columns for later zero/thin-endpoint prediction diagnostics.

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_data.py predictor_v5/tests/test_movement_contract.py -q
```

Expected: PASS, including 2026-before-feature and duplicate-before-filter assertions.

- [ ] **Step 6: Commit the loader and ledger**

```bash
git add backend/predictor_v5/movement_evaluation/data.py backend/predictor_v5/tests/test_movement_data.py
git commit -m "feat: add strict movement cohort ledger"
```

### Task 3: Implement exact point metrics and exploratory slices

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/metrics.py`
- Create: `backend/predictor_v5/tests/test_movement_metrics.py`

**Interfaces:**
- Consumes: sorted matched prediction frames with `S`, `E`, `r`, `d`, and per-model `<model>_E_hat`, `<model>_r_hat`, `<model>_d_hat` columns.
- Produces: `direction(values: ArrayLike) -> np.ndarray`; `training_majority_class(actual_r: ArrayLike) -> str`; `score_direction_baselines(frame: pd.DataFrame, training_majority: str) -> dict[str, object]`; `score_model(frame: pd.DataFrame, model: str) -> dict[str, object]`; `score_contrast(frame: pd.DataFrame, left: str, right: str) -> dict[str, object]`; `score_slices(frame: pd.DataFrame, model_names: tuple[str, ...]) -> dict[str, object]`; `target_diagnostics(frame: pd.DataFrame) -> dict[str, object]`; `macro_by_season(fold_scores: Sequence[Mapping[str, object]]) -> dict[str, object]`; nullable metric objects shaped as `{"value": float | None, "reason": str | None}`.

- [ ] **Step 1: Write hand-computed movement/error/skill tests**

Use two or three exact rows to assert `r=log(E/S)`, `d=E-S`, predicted equivalents, log/value MAE, RMSE, median and linear p90 absolute errors, signed bias, mean actual/predicted movement, and:

```python
assert score["skill_log"]["value"] == 1 - mae_log / np.mean(np.abs(actual_r))
assert score["skill_value"]["value"] == 1 - mae_value / np.mean(np.abs(actual_d))
assert contrast["mae_log_difference"]["value"] == production_mae - ablation_mae
assert contrast["positive_mae_log_improvement"]["value"] == ablation_mae - production_mae
```

Assert value-change error equals endpoint error because starts are common. Pin zero denominators to `value=None, reason="zero_baseline_error"`, empty cohorts to `empty_cohort`, and any mismatched key/order or nonfinite prediction to `ContractRefusal` rather than row dropping.

- [ ] **Step 2: Write direction, calibration, rank, and tail tests**

Cover every metric definition from the design:

- exact flat boundaries `abs(r) <= log(1.10)`, 3x3 matrix, per-class recalls with absent-class nulls, supported-class balanced accuracy, mover accuracy, and train-majority tie precedence `flat`, then `down`, then `up`; no-change is always flat and has zero mover accuracy whenever actual movers exist, while the train-majority comparator remains direction-only;
- OLS `r = intercept + slope*r_hat`, constant-prediction slope null reason, mean signed bias, and all eight lower-inclusive/upper-exclusive fixed bins with empty bins present;
- movement Spearman using average ties, null for fewer than two/constant vectors, within-season-position cells, equal-cell correlation aggregate with included cell identities, and no-change rank null;
- `k=max(1, ceil(0.10*n))`, player-ID tie-breaking, boundary tie counts, top/bottom overlap and selected actual movement, with completely constant predicted vectors abstaining;
- actual `abs(r) >= log(1.50)` large-move slice, positive/negative recall and precision, no-predicted-positive null, and exactly 20 or fewer largest errors sorted by error descending then season/player ID;
- overall/season/position/season-position/value-tier/position-tier/start-liquidity/endpoint-liquidity slices, all empty cells, `low_support: n < 30`, descriptive age/games/points/liquidity summaries, and endpoint slices labeled retrospective;
- per-season target diagnostics with start/end medians, median realized log movement, and start/end liquidity distributions, labeled as scale-drift observations rather than normalization or correction.

- [ ] **Step 3: Run metrics tests and verify they fail**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_metrics.py -q
```

Expected: FAIL because metric functions are absent.

- [ ] **Step 4: Implement finite-only metric primitives and deterministic slices**

Use NumPy linear quantiles, SciPy Spearman or an equivalent average-rank computation, and explicit support counts. Never round stored values, never turn mean absolute log error into a percentage, never average row-level skills, never rank end values, and never omit an empty requested cell. `score_contrast` must always describe sign conventions in its keys/metadata.

- [ ] **Step 5: Run metrics tests and commit**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_metrics.py -q
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m ruff check --no-cache predictor_v5/movement_evaluation/metrics.py predictor_v5/tests/test_movement_metrics.py
git add backend/predictor_v5/movement_evaluation/metrics.py backend/predictor_v5/tests/test_movement_metrics.py
git commit -m "feat: add movement evaluation metrics"
```

Expected: all point-metric, zero-denominator, boundary, empty-cell, and deterministic-order tests PASS.

### Task 4: Run fixed folds with a matched constant-points ablation

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/models.py`
- Create: `backend/predictor_v5/tests/test_movement_models.py`

**Interfaces:**
- Consumes: `EvaluationData.primary_scored`, `FOLDS`, Task 3 direction helpers, and protected `predictor_v5.model.WhatIfModel`.
- Produces: `FoldResult(spec: FoldSpec, train_keys: tuple[tuple[int, str], ...], test_keys: tuple[tuple[int, str], ...], predictions: pd.DataFrame, tier_edges: dict[str, tuple[float, float] | None], training_majority: str, model_parameters: dict[str, object], clipping: dict[str, int])`; `run_primary_folds(rows: pd.DataFrame, *, model_factory: Callable[..., WhatIfModel] = WhatIfModel) -> tuple[FoldResult, ...]`; `attach_start_observable_diagnostics(data: EvaluationData, folds: Sequence[FoldResult], *, model_factory=WhatIfModel) -> dict[str, object]`.

- [ ] **Step 1: Write failing fixed-fold and common-cohort tests**

Build six-season synthetic data and a spy model. Assert exactly ten fit calls (production and no-production for five folds), fixed train seasons, nonempty rows from every required train season, five nonempty test seasons, sorted identical fit/predict keys across learned models, and identical held-out keys for no-change. Assert a missing train season/test cohort refuses instead of shortening/omitting the fold.

- [ ] **Step 2: Write ablation and prediction-contract tests**

Pin the adapter behavior:

```python
assert production_fit["points"].tolist() == original_train_points
assert production_predict["points"].tolist() == original_test_points
assert set(no_production_fit["points"]) == {0.0}
assert set(no_production_predict["points"]) == {0.0}
assert no_change_E_hat.tolist() == test["start_value"].tolist()
assert production_model_kwargs == no_production_model_kwargs == {
    "label_source": TARGET_ID, "scale_max": 10000.0
}
```

Assert all non-points columns and target values are identical, changing only held-out points cannot change no-production/no-change predictions, production preserves protected model monotone-grid behavior, negative points reach production fit unchanged and are counted while its prediction grid clamps to zero, and cap endpoints are counted as `at_cap_count` rather than called strict clipping.

After each real synthetic fit, assert both estimators expose the same effective scikit-learn parameters as the contract and `do_early_stopping_ is False`; this verifies the documented effective default rather than assuming it from the training-row count.

- [ ] **Step 3: Write leakage mutation tests**

Add `test_fold_boundaries_and_test_outcomes_do_not_leak`, `test_future_fold_cannot_change_earlier_fold`, and `test_test_liquidity_changes_mask_not_training`. Mutate one test endpoint, a later season, or test end-liquidity and assert earlier fit frames, predictions, tier edges, and train-majority class remain byte/equality identical as appropriate. Assert tiers come from training `S` within position at linear thirds, apply `S <= q1`, `q1 < S <= q2`, `S > q2`, and preserve collapsed/empty tiers.

- [ ] **Step 4: Run model tests and verify they fail**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_models.py -q
```

Expected: FAIL because `FoldResult` and fold runner do not exist.

- [ ] **Step 5: Implement the minimal read-only adapter**

Copy frames before zeroing points. Feed the protected model only `player_id`, `season`, `position`, `age`, `points`, `start_value`, and `end_value`; games/experience remain descriptive only. Wrap fit/predict in warnings-as-errors and NumPy overflow/invalid checks. Refuse any nonfinite or nonpositive learned prediction and any key/length/order mismatch; do not salvage rows. Add `S/E/r/d` and all three model prediction columns to one common held-out frame.

For start-observable zero/thin diagnostics, fit only the same fold models and predict rows whose inputs are available; keep them outside primary metrics. Never epsilon-adjust zero endpoints and never score missing endpoints.

- [ ] **Step 6: Run model, data, and metric tests; commit**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_models.py predictor_v5/tests/test_movement_data.py predictor_v5/tests/test_movement_metrics.py -q
git add backend/predictor_v5/movement_evaluation/models.py backend/predictor_v5/tests/test_movement_models.py
git commit -m "feat: add matched movement fold models"
```

Expected: PASS; no test asserts that production must improve on synthetic or historical data.

### Task 5: Add deterministic paired uncertainty and fold aggregation

**Files:**
- Modify: `backend/predictor_v5/movement_evaluation/metrics.py`
- Create: `backend/predictor_v5/tests/test_movement_uncertainty.py`

**Interfaces:**
- Consumes: Task 4 held-out prediction frames and Task 3 point scorers.
- Produces: `bootstrap_uncertainty(frame: pd.DataFrame, *, seed: int = 20261025, resamples: int = 2000, cluster: Literal["player", "season"]) -> dict[str, object]`; `leave_one_season_out(frame: pd.DataFrame) -> list[dict[str, object]]`; `summarize_primary(folds: Sequence[FoldResult]) -> tuple[list[dict[str, object]], dict[str, object]]`.

- [ ] **Step 1: Write deterministic clustered-bootstrap tests**

Use repeated player IDs across seasons to assert a sampled player carries all of that player's rows, per-fold clustering samples distinct player IDs within that fold, season-block sampling carries entire season frames, and production/no-production/no-change statistics use the identical draw indices without refitting. Monkeypatch `model_factory.fit` to raise if uncertainty calls it.

Assert same seed/sorted input produces byte-equal results after input row shuffling; a different seed changes synthetic intervals; exactly 2,000 draws and 2.5/97.5 linear percentiles are recorded.

- [ ] **Step 2: Write undefined-draw, macro, and leave-one-season tests**

Assert each interval records requested/effective/undefined draw counts and is `null` with `insufficient_valid_resamples` when fewer than 1,800/2,000 draws define the statistic. Cover paired MAE/RMSE differences, skill gains, direction-accuracy differences, positive-improvement headline, and equal-cell season-position Spearman differences overall and per position. No-change rank contrast must remain null.

Hand-compute a deliberately unbalanced two-season fixture to show pooled MAE differs from equal-season macro MAE, pooled skill is a ratio of pooled MAEs, macro skill is a mean of defined fold skills with undefined count, and each leave-one-season-out result omits exactly one named season. Pin the five individual fold estimates and exploratory five-block bootstrap metadata/caveat.

- [ ] **Step 3: Run uncertainty tests and verify failure**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_uncertainty.py -q
```

Expected: FAIL because uncertainty/aggregation interfaces are absent.

- [ ] **Step 4: Implement resampling over retained prediction rows**

Generate all indices from sorted keys using `np.random.default_rng(20261025)`. Recompute each paired statistic on the same resample; never refit. Produce player-cluster intervals overall/per fold/per position and a separately labeled exploratory season-block interval. State that neither one-way bootstrap captures simultaneous player/team/season/league uncertainty and that family-wide multiplicity is uncontrolled.

- [ ] **Step 5: Run tests and commit**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_metrics.py predictor_v5/tests/test_movement_uncertainty.py predictor_v5/tests/test_movement_models.py -q
git add backend/predictor_v5/movement_evaluation/metrics.py backend/predictor_v5/tests/test_movement_uncertainty.py
git commit -m "feat: add clustered movement uncertainty"
```

### Task 6: Add optional archive-KTC diagnostics without affecting primary results

**Files:**
- Modify: `backend/predictor_v5/movement_evaluation/models.py`
- Create: `backend/predictor_v5/tests/test_movement_ktc.py`

**Interfaces:**
- Consumes: validated archive KTC fields only when mode is `archive`, common primary eligibility, fixed folds, and Task 3 metrics.
- Produces: `run_ktc_diagnostic(data: EvaluationData, *, model_factory: Callable[..., WhatIfModel] = WhatIfModel) -> dict[str, object]`; off-mode literal payload `{"status": "off", "predictions": []}`; archive mode fold records with common intersection keys and separately named Backchannel refits.

- [ ] **Step 1: Write off-mode isolation tests**

Use a mapping/frame wrapper that raises on any `start_ktc`, `end_ktc`, or `weekly_ktc` access. Assert the complete primary folds/pooled/prediction bytes are identical with KTC absent versus ignored and the optional payload is exactly off. Assert no KTC absence can fail the primary run.

- [ ] **Step 2: Write archive-intersection/refit tests**

Assert each available optional fold:

- intersects valid positive finite `<=9999` KTC labels with primary-eligible train/test keys;
- refits KTC production plus separately named Backchannel production/no-production models on exactly the same ordered training intersection;
- scores all diagnostic models and no-change on exactly one ordered test intersection;
- reports train/test attrition keys/counts and marks an empty required-season/fold `unavailable` without changing the complete primary report;
- computes `ratio = KTC_E_hat / KTC_S`, then `transplanted_Backchannel_E_hat = ratio * Backchannel_S` with no second cap;
- labels `provenance_limited: true`, `status: diagnostic_only`, and the transplant as uncalibrated cross-market comparability;
- never subtracts errors/skills across different target/start/baseline definitions.

- [ ] **Step 3: Run diagnostic tests and verify failure**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_ktc.py -q
```

Expected: FAIL because `run_ktc_diagnostic` is absent.

- [ ] **Step 4: Implement optional diagnostic and verify**

Reuse fold/model/metric primitives; do not introduce database, mapping, weekly-KTC, or sealed replacement imports. Keep optional predictions separately keyed and never let optional availability or scores enter `interpretation.evidence_flags`.

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_ktc.py predictor_v5/tests/test_movement_models.py -q
git add backend/predictor_v5/movement_evaluation/models.py backend/predictor_v5/tests/test_movement_ktc.py
git commit -m "feat: add optional KTC movement diagnostic"
```

### Task 7: Assemble, identify, and safely publish deterministic reports

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/report.py`
- Create: `backend/predictor_v5/tests/test_movement_report.py`

**Interfaces:**
- Consumes: bundle/data/folds/scores/optional diagnostic plus source/runtime metadata.
- Produces: `runtime_identity() -> dict[str, object]`; `source_identities(package_root: Path) -> dict[str, str]`; `build_run_identity(*, inputs: Mapping[str, str], sources: Mapping[str, str], mode: KtcMode, runtime: Mapping[str, object]) -> tuple[dict[str, object], str]`; `build_report(...) -> dict[str, object]`; `canonical_bytes(value: object) -> bytes`; `report_path(repo_root: Path, run_sha256: str) -> Path`; `publish_report(report: Mapping[str, object], repo_root: Path) -> Path`.

- [ ] **Step 1: Write report-schema and run-identity tests**

Assert exact top-level keys and nested required sections from the design. Pin evidence values, target formula/columns/dates/generations, fixed contract, complete coverage, five folds, pooled scores, held-out predictions sorted once per key, optional payload, and mechanical interpretation flags only.

Assert run identity contains protocol/schema, input/member/source hashes, base provenance commit, exact folds/models/metrics/bootstrap/mode, Python/NumPy/pandas/scikit-learn/SciPy versions and thread settings. Assert it omits current HEAD, branch, timestamp, hostname, temp paths, and scores. Recompute `run_sha256 == sha256(canonical_bytes(run_identity)).hexdigest()`.

- [ ] **Step 2: Write canonical serialization and null-safety tests**

Assert sorted keys, compact separators, exactly one final LF, stable sorted nonsemantic arrays, full precision, row-order independence, and byte equality across repeated construction. `float("nan")`, infinities, sets, non-string object keys, and a metric `None` without a reason must refuse. Empty cells must serialize with support counts and explicit reasons.

- [ ] **Step 3: Write output durability tests**

Using a temporary repository root, assert:

- output is exactly `artifacts/market_movement/development-2020-2025-<run-sha256>.json` beneath `predictor_v5`;
- safe creation of the new regular directory only, with symlink/non-directory/escape refusal;
- first publish is exclusive and atomic; same destination+same bytes succeeds idempotently; same destination+different bytes refuses without replacement;
- temporary-name collision is preserved, partial/interrupted write and file/directory `fsync` failures remove only writer-owned temporary/final inodes, and no failure leaves a success report;
- existing artifacts, confirmations, replacements, data, model files, and runtime directories never change.

- [ ] **Step 4: Run report tests and verify failure**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_report.py -q
```

Expected: FAIL because report interfaces are absent.

- [ ] **Step 5: Implement report assembly/publication**

Hash all seven new package source files plus protected `model.py`; source hashes, not source content, enter identity. Canonicalize the run identity first, derive the destination, then assemble the report with matching `run_identity`/`run_sha256`. Prediction rows must contain only the specified common fields and predictions; no names, estimators, raw archive rows, or future evidence. Keep `full_preseason_*` null with the required reason and list every as-of/measurement/timing/bootstrap/multiplicity caveat.

- [ ] **Step 6: Run report tests and commit**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_report.py predictor_v5/tests/test_movement_metrics.py predictor_v5/tests/test_movement_uncertainty.py -q
git add backend/predictor_v5/movement_evaluation/report.py backend/predictor_v5/tests/test_movement_report.py
git commit -m "feat: add deterministic movement reports"
```

### Task 8: Add the isolated CLI and enforce the protected boundary

**Files:**
- Create: `backend/predictor_v5/movement_evaluation/__main__.py`
- Create: `backend/predictor_v5/tests/test_movement_cli.py`
- Modify: `backend/predictor_v5/tests/test_movement_contract.py`

**Interfaces:**
- Consumes: all prior task APIs.
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`; internal `run(*, ktc_mode: KtcMode, repo_root: Path) -> Path`; public `main(argv: Sequence[str] | None = None) -> int`. Only `--ktc-diagnostic {off,archive}` and `--help` are accepted.

- [ ] **Step 1: Write CLI grammar and staged-orchestration tests**

Assert default mode `off`, explicit `archive`, and help. Assert `--season`, `--data-path`, `--output`, `--model`, `--threshold`, registry/promotion/save flags, positional arguments, unknown values, and `--ktc-diagnostic 2026` exit 2 before any loader call or write.

With stage spies, pin order: contract/runtime/source identity -> pinned byte load/validation -> ledger -> primary folds -> primary metrics/uncertainty -> optional diagnostic -> complete report -> publication. Contract/input/cohort refusals exit 2, unexpected fit/numerical/I/O failures exit 1, and valid report publication exits 0 regardless of evidence flags or optional unavailability. Every failure before publication leaves no report.

- [ ] **Step 2: Write runtime access-isolation tests**

Instrument reads/writes/imports and assert runtime reads are limited to the three pinned inputs, `model.py`, the new package sources, and installed-version metadata; writes are limited to the new report directory/temp/final file. Explicitly fail on paths containing current starts, current boards, registrations, claims, holdout registries, outcome paths, or season-2026 data. Assert no `asyncpg`, HTTP/network client, database service, sealed lifecycle, model `save`, API/router, or runtime registry is imported/called.

- [ ] **Step 3: Add the protected-byte preservation test**

In `test_movement_contract.py`, define the exact protected path tuple from `/private/tmp/ktc-movement-baseline.md`. For each path compare current bytes with `git show c4270804379c26bcb340f32a07e507153b4c7342:<path>` and assert equality. Also assert `git diff --name-only <base>...HEAD` contains no protected path. This is a read-only comparison only; do not run seal/lifecycle code or open any registered outcome.

- [ ] **Step 4: Run CLI tests and verify failure**

```bash
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_cli.py predictor_v5/tests/test_movement_contract.py -q
```

Expected: FAIL because `movement_evaluation.__main__` is absent.

- [ ] **Step 5: Implement the minimal CLI**

Map only `ContractRefusal`/validated cohort refusals to exit 2; allow `SystemExit(2)` from argparse; map all unexpected stage exceptions to exit 1 with no partial JSON. Do not add this command to protected `predictor_v5/__main__.py`. The internal `repo_root` parameter exists for tests only and is not exposed as a CLI option.

- [ ] **Step 6: Run all new tests, the existing suite, and Ruff**

From `backend`:

```bash
env -u DATABASE_URL -u DB_HOST -u DB_PORT -u DB_NAME -u DB_USER -u DB_PASSWORD /Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests/test_movement_*.py -q
env -u DATABASE_URL -u DB_HOST -u DB_PORT -u DB_NAME -u DB_USER -u DB_PASSWORD /Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m pytest predictor_v5/tests -q
/Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python -m ruff check --no-cache predictor_v5
```

Expected: all new tests PASS; the baseline suite remains at least `776 passed, 9 skipped` plus the new passes with no new failures; Ruff passes. Investigate any changed existing skip/pass count rather than blindly updating the expectation.

- [ ] **Step 7: Commit the standalone CLI**

```bash
git add backend/predictor_v5/movement_evaluation/__main__.py backend/predictor_v5/tests/test_movement_cli.py backend/predictor_v5/tests/test_movement_contract.py
git commit -m "feat: add isolated movement evaluation CLI"
```

### Task 9: Run and commit the authorized deterministic historical evaluation

**Files:**
- Create: `backend/predictor_v5/artifacts/market_movement/development-2020-2025-<run-sha256>.json`
- Do not modify any source, test, input, protected file, or existing artifact in this task.

**Interfaces:**
- Consumes: the fully committed default-mode evaluator and the three pinned historical inputs.
- Produces: one immutable KTC-off historical-development JSON report whose filename hash recomputes from its stored run identity.

- [ ] **Step 1: Establish a clean committed implementation and snapshot protected hashes**

From repository root:

```bash
git status --short
git diff --exit-code c4270804379c26bcb340f32a07e507153b4c7342 -- \
  backend/data/training-data.zip \
  backend/predictor_v5/__init__.py backend/predictor_v5/__main__.py \
  backend/predictor_v5/confirmation.py backend/predictor_v5/data.py \
  backend/predictor_v5/data/backchannel_labels_2019_2025.csv \
  backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv \
  backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json \
  backend/predictor_v5/data/ktc_replacement_2026_holdout_registry.json \
  backend/predictor_v5/data/ktc_replacement_2026_manifest.json \
  backend/predictor_v5/data/ktc_replacement_mapping_snapshot.csv \
  backend/predictor_v5/data/label_manifest.json \
  backend/predictor_v5/data/model_2026_confirmation_manifest.json \
  backend/predictor_v5/data/model_2026_holdout_registry.json \
  backend/predictor_v5/data/start_values_2026.csv \
  backend/predictor_v5/evaluate.py backend/predictor_v5/ktc_replacement_2026.py \
  backend/predictor_v5/label_export.py backend/predictor_v5/labels.py \
  backend/predictor_v5/model.py backend/predictor_v5/replacement_cohort.py \
  backend/predictor_v5/replacement_contract.py backend/predictor_v5/replacement_ktc.py \
  backend/predictor_v5/replacement_lifecycle.py backend/predictor_v5/replacement_metrics.py \
  backend/predictor_v5/replacement_mfl.py backend/predictor_v5/replacement_report.py \
  backend/predictor_v5/replacement_seal.py backend/predictor_v5/replacement_starts.py \
  backend/predictor_v5/sql/backchannel_labels.sql backend/predictor_v5/sql/ktc_replacement_end.sql
```

Expected: clean tree and zero diff for every protected path. Stop if either condition fails.

- [ ] **Step 2: Run exactly one authorized primary historical evaluation**

From `backend`:

```bash
OMP_NUM_THREADS=1 \
  /Users/jaykapoor/.codex/worktrees/mfl-hardened-refresh/ktc-predictor/.venv/bin/python \
  -m predictor_v5.movement_evaluation
```

Expected: exit 0 and exactly one new regular JSON file under `predictor_v5/artifacts/market_movement/`. Do not run `--ktc-diagnostic archive` on real data in this task; KTC remains optional synthetic-tested diagnostics only.

- [ ] **Step 3: Independently verify the emitted report**

Using a read-only Python check independent of `build_report`, assert:

- filename suffix equals SHA-256 of stored canonical `run_identity`;
- stored `run_sha256` matches both; canonical reserialization equals file bytes and has one LF;
- protocol/schema/evidence flags and fixed input/member/model hashes are exact;
- folds are exactly 2021–2025 with the fixed earlier train seasons, nonzero required train/test counts, matching train/test key hashes, and 2021 `short_history`;
- held-out prediction keys are unique/sorted and exactly equal the union of fold test keys; all three models use every row; all predictions are finite/positive and no-change equals `S`;
- recomputed `r`, `d`, each model's log/value errors, pooled/fold MAE/RMSE, no-change skill, production-vs-ablation headline, direction counts, and macro summaries equal the stored values at full precision;
- coverage ledgers reconcile, zero-end and thin-endpoint rows are excluded from primary scores, `full_preseason_*` is null for `not_recoverable_from_historical_export`, and all empty/undefined metrics have reasons;
- `optional_ktc.status == "off"`, no 2026 season keys exist, and findings are reported regardless of sign.

Expected: the independent check exits 0. Record actual headline, fold, position, coverage, saturation, tail, and caveat values in the commit body/review notes without editing the report or inventing a favorable threshold.

- [ ] **Step 4: Verify byte stability and no-replace idempotence**

Run the exact Step 2 command once more only as a deterministic rerun of the already-published identity. Expected: exit 0, no second file, unchanged SHA-256/bytes/inode-independent content. This rerun does not create a second evaluation or a new report identity.

- [ ] **Step 5: Re-run protected/full verification after publication**

Repeat Task 8 Step 6 and Task 9 Step 1's protected diff. Expected: tests/Ruff pass, only the new report is untracked, and every protected path remains byte-identical.

- [ ] **Step 6: Commit only the deterministic report**

```bash
git add backend/predictor_v5/artifacts/market_movement/development-2020-2025-*.json
git diff --cached --name-only
git commit -m "data: add historical movement evaluation report"
```

Expected staged path list: exactly one JSON file in the new namespace. Do not amend any sealed result, make a promotion decision, or push.

## Final Verification Checklist

- [ ] `git diff --name-only c4270804379c26bcb340f32a07e507153b4c7342..HEAD` contains only the design, this plan, new movement package/tests, and one new movement report.
- [ ] The protected-base test and explicit protected diff both pass.
- [ ] The complete predictor_v5 suite and Ruff pass in the pinned environment with database variables unset.
- [ ] The default report independently recomputes from its held-out rows and is byte-identical on rerun.
- [ ] No 2026 outcome, sealed runner, database, network, registration, claim, model persistence, API/UI/deployment/runtime surface, or existing artifact was read for outcomes or changed.
- [ ] Review notes explicitly call the result historical development conditioned on realized points, not a preseason forecast, causal estimate, target-objectivity validation, untouched confirmation, or shipping authorization.
