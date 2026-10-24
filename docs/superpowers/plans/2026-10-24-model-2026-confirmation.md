# 2026 Model Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sealed, one-shot `python -m predictor_v5 confirm-2026` protocol that verifies a reviewed untouched registration before fitting the frozen old-v1 baseline and refreshed-v1 candidate on 2021–2025 and writing one canonical atomic 2026 confirmation report.

**Architecture:** Keep the existing season/model/API behavior untouched. Add a focused `backend/predictor_v5/confirmation.py` core that owns canonical artifact validation, preflight ordering, fixed fits/scoring/diagnostics, payload hashing, and atomic no-replace output; add only a no-override CLI subcommand in `__main__.py`. Commit the pre-outcome manifest and sealed registry as repository data, while tests construct temporary roots and inject them through the core API to exercise post-season registration and all refusal paths.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn `HistGradientBoostingRegressor`, existing `WhatIfModel`/label-export helpers, pytest, canonical UTF-8 JSON, SHA-256, POSIX fsync/atomic-create primitives.

**Spec:** `docs/superpowers/specs/2026-10-24-model-2026-confirmation-design.md`

## Global Constraints

- The only user invocation is `python -m predictor_v5 confirm-2026`; it accepts no global or confirmation configuration inputs.
- Register post-season 2026 `target_stats: {path, sha256}` separately from the unchanged frozen training archive; hash-check it before any target rows and verify all derived target stats against it.
- Manifest, registry, and input registration use schema version 2; the strict confirmation sidecar uses version 1 and is read directly from its registered path.
- Train both models on exactly `[2021, 2022, 2023, 2024, 2025]`; the sole target is canonical refreshed-v1 2026 data.
- Require canonical strict JSON (sorted keys, compact separators, UTF-8 newline), repository-relative paths only, and reject traversal, unknown/missing fields, duplicate keys, noncanonical bytes, and hash/provenance drift.
- Require one matching reviewed `untouched` registry/input registration before opening target labels; sealed, reused, missing, malformed, duplicate, or mismatched records exit 2 and produce no report.
- Verify live `FEATURES`, `MONOTONIC`, `HGB_PARAMS`, `POINTS_GRID_STEP`, filters, gate, recipe source hashes, and v1 provenance before fitting; no diagnostics may influence configuration, promotion, or selection.
- Sort every row by `(season, player_id)`, reject raw and joined duplicate keys, reject all raw 2026/future training leakage before filtering, require complete positions/seasons, and use full-precision gate comparisons.
- `promotion_authorized` is always `false`; a successful run may recommend promotion only and never changes deployment/runtime/API/UI/season behavior.
- Output is `backend/predictor_v5/artifacts/confirmations/2026-<manifest-sha256>.json`, payload-only SHA-256, temporary-file flush/fsync, atomic create without replacement, directory fsync, and cleanup on failure.
- The manifest hash is SHA-256 of the manifest's canonical bytes with no self-hash field; the wrapper hash is not included in the payload, and `payload_sha256` hashes only canonical payload bytes.
- Live source identities are `backend/predictor_v5/model.py`, `backend/predictor_v5/sql/backchannel_labels.sql`, and `backend/predictor_v5/label_export.py`; source drift errors name path, expected hash, actual hash, and the first differing contract.

## Review Focus

- Target labels must never be opened before a matching untouched registration: test with an instrumented loader and a sealed/mismatched registry.
- Canonical JSON and path validation must reject semantically equivalent but noncanonical bytes, unknown fields, missing fields, absolute paths, and `..` traversal: pin in artifact-validation tests.
- A duplicate raw key must fail before filtering/fitting, and a duplicate after stats join must fail with source/count/sample: pin in preflight tests.
- A valid but empty/insufficient cohort, zero no-change MAE, missing position skill, or incomplete position must atomically retain rather than recommend: pin in gate tests.
- Re-running against an existing final path or an injected atomic-write failure must preserve the prior state and produce no replacement/partial report: pin in atomic-output tests.

### Task 1: Define committed protocol artifacts and canonical validation

**Files:**
- Create: `backend/predictor_v5/data/model_2026_confirmation_manifest.json`
- Create: `backend/predictor_v5/data/model_2026_holdout_registry.json`
- Create: `backend/predictor_v5/confirmation.py`
- Test: `backend/predictor_v5/tests/test_confirmation_protocol.py`

**Interfaces:**
- Produces `canonical_json_bytes(value) -> bytes`, `sha256_bytes(data) -> str`, `file_sha256(path: Path) -> str`, `validate_repo_relative_path(value) -> Path`, `load_canonical_json(path, schema) -> tuple[dict, str]`, and manifest/registry/input schema validators for later tasks. `load_canonical_json` must parse with an `object_pairs_hook` that raises on duplicate keys, compare file bytes to the canonical bytes, and return the hash of those canonical bytes.
- The manifest records `source_identities` for `backend/predictor_v5/model.py`, `backend/predictor_v5/sql/backchannel_labels.sql`, and `backend/predictor_v5/label_export.py`; validation recomputes each file SHA-256 and checks the live model constants `FEATURES`, `MONOTONIC`, `HGB_PARAMS`, `POINTS_GRID_STEP`, the SQL/build recipe paths, and `label_export` provenance helper contract before accepting the manifest.
- The manifest must encode the exact spec values, including train seasons, positions, `min_trades: 5`, `baseline-hl28-v1`, gate `0.02/30/0.05`, walk-forward folds, bootstrap seed `20261024`/`10000`/`[2.5,97.5]`, registry/input paths, and output template.

- [ ] **Step 1: Write failing schema/path tests** for canonical bytes/hash, every required manifest field/value, unknown/missing fields, absolute/traversal paths, duplicate registry entries, and sealed initial lifecycle.
- [ ] **Step 2: Run** `cd backend && python -m pytest predictor_v5/tests/test_confirmation_protocol.py -q`; **Expected:** FAIL because the confirmation module/artifacts do not exist.
- [ ] **Step 3: Implement** canonical serialization, strict schema/path checks, SHA-256 helpers, and the two committed artifacts; compute the manifest hash from its exact canonical bytes and make the registry key `(2026, manifest_sha256)` sealed.
- [ ] **Step 4: Run** the focused test command; **Expected:** PASS.
- [ ] **Step 5: Commit** `git add backend/predictor_v5/confirmation.py backend/predictor_v5/data/model_2026_confirmation_manifest.json backend/predictor_v5/data/model_2026_holdout_registry.json backend/predictor_v5/tests/test_confirmation_protocol.py && git commit -m "feat: seal 2026 confirmation protocol artifacts"`.

### Task 2: Implement preflight and safe registration gating

**Files:**
- Modify: `backend/predictor_v5/confirmation.py`
- Test: `backend/predictor_v5/tests/test_confirmation_preflight.py`

**Interfaces:**
- Consumes Task 1 validators and `label_export.file_sha256`, `label_identity`, `validate_market_export`, plus injected `repo_root`, `manifest_path`, `registry_path`, `input_registration_path`, `target_loader`, and `output_dir`.
- The input-registration schema is exact: `schema_version: 2`, `season: 2026`, `manifest_sha256`, `target_csv`, `target_sidecar`, `target_stats`, `provenance`, `target_contract`, `recipe`, and `holdout_attestation`. All three target artifacts have exactly `{path, sha256}` with canonical repository-relative paths and lowercase SHA-256. The existing provenance/target/recipe field contracts remain fixed, with `provenance.identity: "sidecar"` and `holdout_attestation.transition_source: "sealed"`. See the amended design for the complete sidecar contract.
- The raw target remains the 14-column build-labels CSV. The ten market columns are validated as raw SQL output; derived `position`, `age`, `games`, and `points` must match a complete key join to the separately registered 2026 stats archive. The frozen 2020--2025 training archive is never used for the target join.
- The registered confirmation sidecar has exactly `schema_version: 1`, `target_csv: {path, sha256}`, `target_stats: {path, sha256}`, `model_version: "baseline-hl28-v1"`, `verified: true`, `problems: []`, `source: "export"`, and `identity: "sidecar"`. Both artifact identities and provenance identity must equal registration. Read this exact hash-bound path directly; do not use target `label_identity` or consult adjacent unregistered files.
- Registry sealing metadata must match the manifest. Nonempty string commits/review metadata, canonical seconds-plus-offset timestamps, review at or after sealing, and exactly `sealed -> untouched` or `sealed -> reused` are required. All review fields must equal the registration attestation.
- `PreflightContext` has concrete fields `manifest: dict`, `manifest_sha256: str`, `input_registration: dict`, `input_registration_sha256: str`, `holdout_registry: dict`, `holdout_registry_sha256: str`, `baseline_csv: Path`, `candidate_csv: Path`, `training_archive: Path`, `target_csv: Path`, `target_sidecar: Path`, `target_stats: Path`, `output_path: Path`, `train_seasons: tuple[int, ...]`, `positions: tuple[str, ...]`, `label_filter: dict`, `target_recipe: dict`, and `target_loader: Callable[[Path, Path], pd.DataFrame]`.
- Produces `preflight(config: ConfirmationConfig) -> PreflightContext` and `run_confirmation(config: ConfirmationConfig) -> dict`; `ConfirmationConfig` contains only injected roots/paths/loaders for tests, never user-selectable model/filter/season values. Preflight does not call `target_loader` until registration, source hashes, target CSV/sidecar hashes, v1 provenance, columns, and recipe identities all pass.

- [ ] **Step 1: Write failing tests** for missing/noncanonical/mismatched artifacts, duplicate JSON object keys, unknown wrapper fields, sealed/reused status, registry attestation fields/status/reviewer/evidence requirements, wrong manifest/season/provenance/recipe hashes, missing-input registration and missing target sidecar, target-loader call ordering, raw/join duplicate keys, 2026 training leakage, incomplete 2021–2025 coverage, and exact exit classification (2 for preflight, 1 only unexpected runtime/write errors).
- [ ] **Step 2: Run** `cd backend && python -m pytest predictor_v5/tests/test_confirmation_preflight.py -q`; **Expected:** FAIL on missing preflight API.
- [ ] **Step 3: Implement** ordered preflight: canonical manifest/hash/output-exists check; exactly one matching untouched registry/input attestation; source/archive/SQL/build/live-code hash and provenance checks; target hash/recipe checks; then target load, duplicate checks, sorting, filters, coverage, and no leakage.
- [ ] **Step 4: Run** the focused tests; **Expected:** PASS, including an assertion that target loading remains uncalled for every registration/refusal case.
- [ ] **Step 5: Commit** `git add backend/predictor_v5/confirmation.py backend/predictor_v5/tests/test_confirmation_preflight.py && git commit -m "feat: fail closed on 2026 confirmation preflight"`.

### Task 3: Add fixed model fits, common-target cohort, gate, and diagnostics

**Files:**
- Modify: `backend/predictor_v5/confirmation.py`
- Test: `backend/predictor_v5/tests/test_confirmation_scoring.py`

**Interfaces:**
- Consumes `PreflightContext` and `WhatIfModel`, `build_dataset`, `summarize_exact`, and `score_on_target`-compatible semantics. Baseline and candidate label tables are normalized to columns `player_id:str, season:int, position:str, age:float, points:float, start_value:float, end_value:float`; target rows use the same columns and are additionally required to be refreshed-v1/provenance-verified.
- Produces `fit_and_score(context: PreflightContext) -> dict` with `fits`, `cohort`, `gate`, `diagnostics`, and `decision`. `score_predictions(predictions: pd.DataFrame, target: pd.DataFrame, keys: list[tuple[str,int]]) -> pd.DataFrame` returns exactly `player_id, season, position, start_value, end_value, pred_value, abs_error, no_change_abs_error, residual`; prediction relative change is `pred_value / model_start_value` and scored value is `target_start_value * relative_change`, with `SCALE_MAX = 10000.0` applied by `WhatIfModel`.
- Diagnostic APIs are `summarize_metrics(scored: pd.DataFrame) -> dict` (n, mae, median_absolute_error, p90_absolute_error, signed_residual_mean, residual_summary), `coverage_report(raw, filtered, joined, trainable, one_sided, matched, missing) -> dict` grouped by season/position, and `bootstrap_delta(scored, seed=20261024, resamples=10000, percentile=(2.5,97.5)) -> dict` reporting mean/median delta, win/tie/loss counts, and interval. Every diagnostic object carries both `used_for_configuration: false` and `used_for_promotion: false`.

- [ ] **Step 1: Write failing tests** for exact train seasons and shared target start/end scaling, fixed folds `2022:[2021]`, `2023:[2021,2022]`, `2024:[2021,2022,2023]`, `2025:[2021,2022,2023,2024]`, sorted-row determinism, gate boundaries at gain `0.02`/drop `0.05`/count `30`, zero-MAE/missing-skill/empty-overlap retain behavior, median/p90/bias/residual/coverage/bootstrap shapes, and diagnostics tagged `used_for_configuration: false` and `used_for_promotion: false`.
- [ ] **Step 2: Run** `cd backend && python -m pytest predictor_v5/tests/test_confirmation_scoring.py -q`; **Expected:** FAIL because scoring is not implemented.
- [ ] **Step 3: Implement** fixed fits only after preflight, apply each predicted relative change to the same target start value, intersect baseline/candidate 2026 keys, compute all required diagnostics, and bootstrap paired candidate-minus-baseline absolute-error deltas with seed `20261024` and `10000` resamples.
- [ ] **Step 4: Run** the focused tests; **Expected:** PASS and byte-stable results for shuffled input rows.
- [ ] **Step 5: Commit** `git add backend/predictor_v5/confirmation.py backend/predictor_v5/tests/test_confirmation_scoring.py && git commit -m "feat: score fixed 2026 confirmation cohort"`.

### Task 4: Add payload hashing, one-shot atomic report writing, and CLI

**Files:**
- Modify: `backend/predictor_v5/confirmation.py`
- Modify: `backend/predictor_v5/__main__.py`
- Create: `backend/predictor_v5/tests/test_confirmation_cli.py`

**Interfaces:**
- Consumes `run_confirmation` from Tasks 2–3.
- Produces `write_report(payload: dict, final_path: Path) -> dict` and a CLI `confirm-2026` handler with no options; success prints `payload_sha256=<64 hex>` and exits 0, preflight errors print concise stderr/exit 2, unexpected/write errors print stderr/exit 1. Because the current parser defines global `--labels`, `--min-trades`, and `--end-column` before subcommands, `main(argv=None)` must inspect raw `sys.argv[1:]` (or use a dedicated confirmation parser) before normal argparse processing: if *any* token exactly equal to `--labels`, `--min-trades`, or `--end-column`, or an attached form beginning `--labels=`, `--min-trades=`, or `--end-column=`, occurs anywhere before or after the `confirm-2026` token, reject with exit 2—even when the supplied value equals the default. It must not pass these values into confirmation config.

- [ ] **Step 1: Write failing CLI/output tests** for payload-only hash (not wrapper self-hash), wrapper unknown-field rejection, canonical wrapper bytes (`json.dumps(wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\\n"`), environment versions and nullable Git revision, missing input-registration refusal, raw-market versus joined-column validation, existing-output refusal, and every forbidden argv spelling/position (`--labels`, `--labels=ktc`, `--min-trades`, `--min-trades=5`, `--end-column`, `--end-column=end_value` both before and after `confirm-2026`) returning exit 2 with no report, plus atomic temp cleanup/no-replace behavior and exact stdout/stderr/exit streams.
- [ ] **Step 2: Run** `cd backend && python -m pytest predictor_v5/tests/test_confirmation_cli.py -q`; **Expected:** FAIL because the command and writer are absent.
- [ ] **Step 3: Implement** the strict `confirm-2026` parser entry, report wrapper, fingerprint output, and temp-file flush/fsync/atomic-create/directory-fsync cleanup; never replace an existing final path and never write on preflight failure.
- [ ] **Step 4: Run** `cd backend && python -m pytest predictor_v5/tests/test_confirmation_cli.py -q`; **Expected:** PASS.
- [ ] **Step 5: Commit** `git add backend/predictor_v5/confirmation.py backend/predictor_v5/__main__.py backend/predictor_v5/tests/test_confirmation_cli.py && git commit -m "feat: add one-shot confirm-2026 command"`.

### Task 5: Full regression and protocol acceptance

**Files:**
- Modify: `backend/predictor_v5/tests/test_confirmation_protocol.py`
- Modify: `backend/predictor_v5/tests/test_confirmation_preflight.py`
- Modify: `backend/predictor_v5/tests/test_confirmation_scoring.py`
- Modify: `backend/predictor_v5/tests/test_confirmation_cli.py`

- [ ] **Step 1: Add acceptance tests** for stable runtime/model contracts and that every successful report has `promotion_authorized: false`. Verify branch scope against the recorded base during review/reporting, never using a permanent runtime test dependency on `origin/master`.
- [ ] **Step 2: Run** `cd backend && python -m pytest predictor_v5/tests -q`; **Expected:** all existing and new predictor tests pass.
- [ ] **Step 3: Run** `cd backend && ruff check predictor_v5`; **Expected:** no lint errors.
- [ ] **Step 4: Run** `git diff --check`; **Expected:** no whitespace errors, and inspect `git diff --stat` to confirm only protocol artifacts/module/CLI/tests changed.
- [ ] **Step 5: Commit** `git add backend/predictor_v5/tests && git commit -m "test: cover 2026 confirmation acceptance contract"`.

## Execution Handoff

Execute this plan with `superpowers:subagent-driven-development`: use a fresh implementer/reviewer cycle for each task, preserve the task commits, then perform a whole-branch review and the full verification commands in Task 5 before claiming completion.

## Final review amendment (single fix wave)

- [x] Register and hash-bind a separate post-season 2026 stats ZIP; retain the frozen training archive unchanged. Bump manifest/registry/registration to schema 2 and reseal registry to the revised manifest hash.
- [x] Use real canonical fixtures and fixed repository recipe paths. Prove successful actual manifest validation and exactly one default target load with registered stats.
- [x] Reject raw baseline/candidate duplicates and 2026/future leakage before filtering; reject joined duplicates; require every trainable season/position and every exact fold before any fit. Cache validated training frames for scoring.
- [x] Directly validate the exact registered sidecar and its CSV/stats identities, model version, verified flag, source, problems, and registration identity. Refuse before target loading on every malformed or mismatched provenance case.
- [x] Allow valid data filtered to empty or insufficient target cohorts; skip empty prediction, atomically retain, and prove deterministic real CLI reports and no replacement.
- [x] Validate and bind all sealing and review lifecycle metadata; reject arbitrary transition sources and noncanonical timestamps.
- [x] Remove the permanent local-origin branch comparison test. Perform branch-scope review in the final report.
- [x] Run exactly one full predictor suite, full Ruff check, and whitespace check after focused verification; self-review and commit implementation plus amended design/plan. Evidence: 266 passed, 9 optional SQL tests skipped; Ruff and whitespace checks clean. See the final-fix report.
