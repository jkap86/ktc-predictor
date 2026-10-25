# 2026 MFL-versus-KTC Replacement Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and pre-outcome seal an independent, one-shot, matched-cohort 2026 replacement evaluator without changing the existing confirmation.

**Architecture:** Separate narrowly scoped modules own contracts, starts, target verification, scoring, lifecycle, and reporting; the standalone module orchestrates them. Reuse the existing model and training helpers without modifying them, freezing every imported helper and its transitive source identities in the new manifest. Inputs, review, claim, and output are independent of confirmation.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest, standard-library JSON/ZIP/OS/Git; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-24-mfl-ktc-replacement-evaluation-design.md`

## Global Constraints

- Train exactly 2021–2025; positions QB/RB/TE/WR; KTC scale 9999, MFL scale 10000; MFL model `baseline-hl28-v1`.
- Start 2026-09-09, end 2027-01-10 America/New_York; KTC end window 2026-12-28 through 2027-01-10 inclusive.
- Positive KTC values strictly below 9999; positive MFL values with at least five trades at each end. No synthetic imputation.
- No 2026 input registration, outcome reads, one-shot evaluation, runtime/API/UI/deployment or `season.py` changes, or existing confirmation artifact changes in this plan.
- Do not modify the eight sources hash-bound by confirmation: `backend/predictor_v5/__main__.py`, `backend/predictor_v5/confirmation.py`, `backend/predictor_v5/data.py`, `backend/predictor_v5/evaluate.py`, `backend/predictor_v5/label_export.py`, `backend/predictor_v5/labels.py`, `backend/predictor_v5/model.py`, and `backend/predictor_v5/sql/backchannel_labels.sql`.
- The protected confirmation baseline is PR #6 merge commit `1157678526d3408850780c7ec7808a730d8c2dbc`, sealed artifact commit `e0a278ed1100471b8f9fa127755d374bc51609f8`, and confirmation manifest SHA-256 `c376651de489d0ae5c6478c001d565bb635d6968b46df4a16e99875d9da7430e`; preserve its manifest, registry, and all eight bound source bytes.
- Run tests from `backend` with `python -m pytest`; an existing compatible local virtual environment may be activated as a local example, never committed or assumed as a dependency.

## Review Focus

- Canonical-but-unsafe paths and duplicate JSON keys refuse: Task 1 `test_contract_rejects_ambiguous_json_and_paths`.
- A start key absent versus conflicting has different meaning: Task 2 `test_start_missing_is_coverage_but_conflict_refuses`.
- Earlier conflicting KTC observations cannot hide behind a valid latest row: Task 3 `test_conflict_anywhere_in_window_refuses`.
- Different stats archives with one conflicting common fact refuse: Task 4 `test_common_stats_disagreement_refuses`.
- Crash after durable claim never permits a retry: Task 7 `test_crash_after_claim_consumes_without_report`.

---

## File map and execution convention

Create `replacement_contract.py` (strict identities and schemas), `replacement_starts.py` (frozen starts), `replacement_ktc.py` and `sql/ktc_replacement_end.sql` (raw KTC contract), `replacement_mfl.py` (MFL verification), `replacement_cohort.py` (stats, training and matching), `replacement_metrics.py` (scores and diagnostics), `replacement_lifecycle.py` (preflight and claim), `replacement_report.py` (atomic report), `replacement_seal.py` (reproducible pre-outcome artifact generation), and `ktc_replacement_2026.py` (orchestration/CLI), all under `backend/predictor_v5/`. Tests live in `backend/predictor_v5/tests/test_replacement_*.py`. Each task owns its files; later tasks import the signatures below. Use synthetic fixtures only. An identity is exactly `{"path": <repo-relative POSIX path>, "sha256": <lowercase 64 hex>}`. Frames are pandas DataFrames keyed by `(season, player_id)`; integrity errors raise `ValueError`. Commit steps are for implementers after approval, not for the plan author.

### Task 1: Canonical contracts

**Files:** Create `backend/predictor_v5/replacement_contract.py`; Test `backend/predictor_v5/tests/test_replacement_contract.py`.

**Interfaces:** Produces `canonical_bytes(value: object) -> bytes`, `load_canonical(path: Path, required: set[str]) -> tuple[dict, str]`, `validate_identity(value: dict) -> None`, `validate_manifest(value: dict) -> None`, `validate_joint_attestation(value: dict) -> None`, `validate_registry(value: dict) -> None`, `validate_registration(value: dict) -> None`.

- [ ] Write `test_contract_rejects_ambiguous_json_and_paths`: assert duplicate/unknown/missing keys, noncanonical serialization, uppercase digest, absolute paths, empty components, `..`, backslashes and symlink escape all raise; assert sorted compact UTF-8 plus newline round-trips.
- [ ] Write `test_manifest_and_joint_attestation_exact_schemas`: assert missing and unknown fields in either schema refuse. Write `test_registry_and_registration_cross_references`: assert duplicate keys, sealed/reused/used state, mismatched manifest, reviewer, evidence reference/commit, review time or transition source refuse; only one matching untouched entry and matching joint-review hash are accepted. Validate exact schemas, timestamps and lifecycle `sealed -> untouched -> used` or `sealed -> reused`.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_contract.py -v`; expect FAIL on missing module.
- [ ] Implement signatures and recursive exact-field validation for manifest, registry, registration and joint attestation; resolve paths beneath repository root without opening outcome files. Joint artifact names both sealed hashes, reviewer, evidence and timestamp; registry and registration bind its path and hash.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_contract.py backend/predictor_v5/tests/test_replacement_contract.py && git commit -m 'feat: define replacement protocol contracts'`.

### Task 2: Frozen starts

**Files:** Create `backend/predictor_v5/replacement_starts.py`; Test `backend/predictor_v5/tests/test_replacement_starts.py`.

**Interfaces:** Produces `load_starts(path: Path, root: Path) -> pd.DataFrame` and `check_target_starts(starts: pd.DataFrame, target: pd.DataFrame, source: str) -> dict`.

- [ ] Write `test_frozen_identity_and_dates`: assert SHA `6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb`, Git blob `ead24b508102ca2a1071d07f3f665e9ea920aa96`, introduction `c415177`, v1 model, BC date at/before cutoff and KTC date within the documented 14-day lookback; reject duplicates and future observations. Assert limitation says raw start provenance is not independently verified.
- [ ] Write `test_start_missing_is_coverage_but_conflict_refuses`: absent keys count as exclusions, but differing shared `ktc_value`/`bc_value` or MFL start trades raises.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_starts.py -v`; expect FAIL missing module.
- [ ] Implement frozen content/Git identity checks and source-specific shared-key comparison, sorted by season/player ID; do not infer raw provenance.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_starts.py backend/predictor_v5/tests/test_replacement_starts.py && git commit -m 'feat: verify frozen replacement starts'`.

### Task 3: KTC raw-end and normalized targets

**Files:** Create `backend/predictor_v5/replacement_ktc.py`, `backend/predictor_v5/sql/ktc_replacement_end.sql`; Test `backend/predictor_v5/tests/test_replacement_ktc.py`.

**Interfaces:** Consumes Task 2 starts; produces `select_end_observations(raw: pd.DataFrame) -> pd.DataFrame`, `load_ktc_target(registration: dict, starts: pd.DataFrame, stats: pd.DataFrame, root: Path) -> pd.DataFrame`.

- [ ] Write `test_raw_join_and_snapshot`: assert SQL joins `ktc_value_history h JOIN ktc_values k USING (format, ktc_id)`, dynasty, nonnull Sleeper ID, positive `h.sf_value`, and exports all candidates with exact ordered columns `format,ktc_id,sleeper_id,date,sf_value`; composite source identity is `(format, ktc_id, date)`. Both window endpoints are included and latest date selected. Assert identical same-day duplicates are accepted, invalid/sentinel/nonfinite values rejected, dates and mapping verified, no network/database access during evaluator load.
- [ ] Write `test_conflict_anywhere_in_window_refuses`: same player/date differing values on an older date raise even when a newer row exists. Write `test_normalized_ktc_contract_and_provenance`: exact ordered columns `player_id,season,start_value,end_value,position,age,games,points`, season 2026, valid numeric types, raw end/date match, registered sidecar and raw/SQL/extractor/mapping identities and coverage; stat and shared-start disagreement raise.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_ktc.py -v`; expect FAIL missing module.
- [ ] Implement the offline SQL recipe and verifier; bind exporter/mapping source and mapping-data identities in manifest and sidecar, verify registered bytes only after claim. Never use `DISTINCT ON` to discard candidates.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_ktc.py backend/predictor_v5/sql/ktc_replacement_end.sql backend/predictor_v5/tests/test_replacement_ktc.py && git commit -m 'feat: verify KTC end snapshots'`.

### Task 4: MFL targets and shared statistics

**Files:** Create `backend/predictor_v5/replacement_mfl.py`, `backend/predictor_v5/replacement_cohort.py`; Test `backend/predictor_v5/tests/test_replacement_targets.py`.

**Interfaces:** Produces `load_stats(path: Path) -> pd.DataFrame`, `load_mfl_target(registration: dict, starts: pd.DataFrame, stats: pd.DataFrame, root: Path) -> pd.DataFrame`, `verify_common_stats(ktc: pd.DataFrame, mfl: pd.DataFrame) -> None`.

- [ ] Write `test_stats_zip_contract`: exactly one `training-data.json` member with players envelope and only 2026 seasons; duplicate keys, nonfinite age/points, negative/noninteger games, position or stat mismatch raise.
- [ ] Write `test_mfl_registration_is_independent`: registered CSV, sidecar, stats, SQL/build hashes and v1 provenance must all match even when CSV bytes equal confirmation's; verify exact ordered columns `player_id,season,start_value,end_value,position,age,games,points,start_evidence,end_evidence,start_trades_window,end_trades_window,end_value_next_board,end_evidence_next_board`, dates, start/end trades >=5, positive finite values and frozen shared starts. Missing or adjacent-only sidecar refuses.
- [ ] Write `test_common_stats_disagreement_refuses`: nonidentical ZIP bytes are allowed only if all common player/season position, age, games and points match.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_targets.py -v`; expect FAIL missing functions.
- [ ] Implement strict loaders and one-to-one stats joins; freeze existing `label_export.py`, `sql/backchannel_labels.sql`, `data.py`, and `labels.py` identities when reused.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_mfl.py backend/predictor_v5/replacement_cohort.py backend/predictor_v5/tests/test_replacement_targets.py && git commit -m 'feat: verify MFL targets and shared stats'`.

### Task 5: Training and exact cohort

**Files:** Modify `backend/predictor_v5/replacement_cohort.py`; Test `backend/predictor_v5/tests/test_replacement_cohort.py`.

**Interfaces:** Produces `prepare_training(root: Path, manifest: dict) -> dict[str, pd.DataFrame]` and `matched_cohort(starts: pd.DataFrame, ktc: pd.DataFrame, mfl: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]`.

- [ ] Write `test_training_coverage_and_leakage`: frozen archive and verified MFL CSV, five exact training seasons and all four positions required; duplicates, nonfinite values, future leakage, missing season/position fail before claim, rows sorted. Assert scale/filter contracts match each source.
- [ ] Write `test_exact_intersection_and_relative_predictions`: intersection of starts, both target filters, stats and both predictions is sorted and identical for both sources; `pred_end_s = frozen_start_s * predicted_end_own_s / fitted_start_own_s`; absent keys appear in coverage, no source substitutions.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_cohort.py -v`; expect FAIL missing functions.
- [ ] Implement using existing `KtcLabels`, `CsvLabels`, `build_dataset`, and `WhatIfModel` without changing their code; validate stricter contracts before their permissive loaders and ensure model fits occur only after all outcome validation.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_cohort.py backend/predictor_v5/tests/test_replacement_cohort.py && git commit -m 'feat: construct matched replacement cohort'`.

### Task 6: Metrics, rank and gate

**Files:** Create `backend/predictor_v5/replacement_metrics.py`; Test `backend/predictor_v5/tests/test_replacement_metrics.py`.

**Interfaces:** Produces `score(cohort: pd.DataFrame) -> dict`, `gate(metrics: dict) -> dict`, `diagnostics(cohort: pd.DataFrame, seed: int = 20261024, resamples: int = 10000) -> dict`.

- [ ] Write `test_source_specific_baselines_and_targets`: KTC and MFL each use own actual and start for MAE and `1 - MAE/baseline`; hand-computed unequal scales expose cross-source swaps.
- [ ] Write `test_gate_exact_boundaries`: overall -0.01 and each positional -0.05 pass, -0.010000001 and -0.050000001 fail, 30 passes/29 fails; empty cohort, zero/nonfinite baseline, missing skill/position produce stable retain reasons and `promotion_authorized == False`.
- [ ] Write `test_rank_ties_and_diagnostics_never_gate`: average descending ranks, raw mean absolute rank error, undefined Spearman null; bootstrap is deterministic and diagnostics tagged `used_for_configuration: false`, `used_for_promotion: false` cannot alter decision or reasons. Include MAE, median, p90, bias, residual, paired errors and coverage.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_metrics.py -v`; expect FAIL missing module.
- [ ] Implement full-precision metrics and exact inclusive gate, deterministic paired bootstrap with percentile [2.5,97.5]; compute Spearman using pandas average ranks and numpy correlation, not scipy or any new dependency. Pass means eligible MFL primary with KTC fallback only, not promotion.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_metrics.py backend/predictor_v5/tests/test_replacement_metrics.py && git commit -m 'feat: score replacement gate and diagnostics'`.

### Task 7: Claim-first lifecycle

**Files:** Create `backend/predictor_v5/replacement_lifecycle.py`; Test `backend/predictor_v5/tests/test_replacement_lifecycle.py`.

**Interfaces:** Produces `preflight(root: Path) -> dict` and `claim(context: dict) -> Path`; context binds manifest/registry/registration/joint hashes, training frames and output/claim paths, without reading outcomes.

- [ ] Write `test_preflight_never_opens_outcomes`: spies assert invalid identity, joint review, registration, existing claim/report, training leakage or missing coverage invoke no outcome loader, fit, claim or report. Require both identified sealed manifest hashes and matching review evidence; refuse evidence already exposed without joint review.
- [ ] Write `test_concurrent_claim_is_unique`: two processes race; exactly one exclusive create succeeds, contents bind manifest, registration and joint hash, file and directory are fsynced before outcome reads.
- [ ] Write `test_crash_after_claim_consumes_without_report`: injected exceptions after claim, during validation, fit and write leave claim intact, no partial report, retry refuses; preclaim failures remain retryable without outcome reads.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_lifecycle.py -v`; expect FAIL missing module.
- [ ] Implement metadata-only preflight then `O_CREAT|O_EXCL` claim and directory fsync; never delete claim. Reviewed later used transition carries claim/report identity; do not update confirmation registry.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_lifecycle.py backend/predictor_v5/tests/test_replacement_lifecycle.py && git commit -m 'feat: claim replacement holdout before outcomes'`.

### Task 8: Atomic report and standalone CLI

**Files:** Create `backend/predictor_v5/replacement_report.py`, `backend/predictor_v5/ktc_replacement_2026.py`; Test `backend/predictor_v5/tests/test_replacement_cli.py`.

**Interfaces:** Produces `write_report(payload: dict, path: Path) -> dict`, `run(root: Path, dependencies: dict | None = None) -> dict`, `main() -> int`; CLI accepts no overrides.

- [ ] Write `test_atomic_canonical_report`: wrapper exactly `{schema_version,payload,payload_sha256}`, digest canonical payload bytes, sorted deterministic inputs; temp file fsync, atomic no-replace publish and directory fsync; injected failures leave no partial destination and existing report is unchanged.
- [ ] Write `test_cli_exit_and_order`: preflight/identity failures exit 2 before claim; postclaim invalid outcomes exit 2 without fit/report; unexpected fit/IO/fsync errors exit 1 retaining claim; insufficient cohort writes retain report and exits 0. Assert order preflight, durable claim, hash/verify all outcomes, fit, report; `python -m predictor_v5.ktc_replacement_2026 --anything` refuses.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_cli.py -v`; expect FAIL missing modules.
- [ ] Implement orchestration and report containing protocol/input/registry/code/archive/model/raw/sidecar/mapping hashes, actual dates, filters, fits, exact cohort, coverage, metrics, diagnostics, gate and decision. No CLI data/path/model/threshold overrides.
- [ ] Run the same command; expect PASS.
- [ ] Commit: `git add backend/predictor_v5/replacement_report.py backend/predictor_v5/ktc_replacement_2026.py backend/predictor_v5/tests/test_replacement_cli.py && git commit -m 'feat: add standalone atomic replacement evaluator'`.

### Task 9: Runbook, reviewed source anchor and artifact-only seal

**Files:** Modify `backend/predictor_v5/README.md`; create `backend/predictor_v5/replacement_seal.py`, `backend/predictor_v5/data/ktc_replacement_2026_manifest.json`, `backend/predictor_v5/data/ktc_replacement_2026_holdout_registry.json`; Test `backend/predictor_v5/tests/test_replacement_seal.py`.

**Interfaces:** Consumes all prior interfaces; produces `build_manifest(root: Path, source_anchor: str) -> dict`, `build_registry(manifest: dict, sealed_at: str) -> dict`, `write_seal(root: Path, source_anchor: str, sealed_at: str) -> None`, and CLI `python -m predictor_v5.replacement_seal --source-anchor <commit> --sealed-at <timestamp>`. Manifest binds all new executable sources including the generator, SQL, reused source identities, training archive/member and verified MFL CSV, exact model/filters/cutoffs/recipes, deterministic diagnostics, output and registry/registration paths. Registry starts with one `sealed` entry keyed by final canonical manifest SHA. Neither artifact contains future target hashes.

- [ ] Write `test_generator_deterministic_and_refuses_unreviewed_identity`: using synthetic committed Git fixtures, two runs with identical anchor and timestamp produce identical canonical bytes; hashes equal actual anchor blobs, registry key equals manifest SHA, and no future target hashes appear. Dirty tracked files, a non-HEAD/non-commit anchor, missing required identities, or existing output refuse without writing either artifact.
- [ ] Write `test_source_anchor_and_artifact_only_seal`: compare every manifest source hash to the reviewed anchor commit and working tree; assert the seal commit parent is that anchor and its diff contains only the two new JSON artifacts. Verify canonical bytes, manifest hash/registry key, Git identity and fixed settings. Compare byte identities of all eight named confirmation sources and existing manifest/registry to the protected baseline; assert `season.py`, API, frontend and deployment files unchanged.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_seal.py -v`; expect FAIL missing generator and seal artifacts.
- [ ] Implement `replacement_seal.py`: require a clean tracked worktree and exact committed HEAD anchor, resolve every required source/data identity from that commit, reject missing or mismatched working bytes and existing output, then write only the two canonical JSON artifacts without replacement. Derive registry manifest SHA from canonical manifest bytes; use the explicitly supplied reviewed timestamp, never the wall clock. Fixture tests exercise generation before the actual seal; the real-seal test may skip until artifacts exist, then must pass after the artifact commit.
- [ ] Document pre-outcome sealing, later independent joint review and registration, start-provenance limitation, one-shot failure recovery and manual promotion/fallback. Explicitly prohibit deleting a claim or retrying after it; no real registration or outcomes now.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_contract.py predictor_v5/tests/test_replacement_starts.py predictor_v5/tests/test_replacement_ktc.py predictor_v5/tests/test_replacement_targets.py predictor_v5/tests/test_replacement_cohort.py predictor_v5/tests/test_replacement_metrics.py predictor_v5/tests/test_replacement_lifecycle.py predictor_v5/tests/test_replacement_cli.py predictor_v5/tests/test_replacement_seal.py -v` and `cd backend && python -m pytest predictor_v5/tests -v`; expect PASS (only the real-seal assertion may skip before generation). Review all source and tests.
- [ ] Commit reviewed source anchor: `git add backend/predictor_v5/replacement_seal.py backend/predictor_v5/README.md backend/predictor_v5/tests/test_replacement_seal.py && git commit -m 'feat: add reviewed replacement seal generator'`. Record this commit as `sealing_commit`; no source changes after this point.
- [ ] From `backend`, run `python -m predictor_v5.replacement_seal --source-anchor <reviewed-anchor-sha> --sealed-at <reviewed-utc-timestamp>`, substituting the recorded commit and reviewed ISO-8601 timestamp. Inspect both canonical artifacts and commit only them: `git add backend/predictor_v5/data/ktc_replacement_2026_manifest.json backend/predictor_v5/data/ktc_replacement_2026_holdout_registry.json && git commit -m 'data: seal replacement evaluation protocol'`.
- [ ] Run `cd backend && python -m pytest predictor_v5/tests/test_replacement_seal.py -v` and `git diff --name-only HEAD^ HEAD`; expect PASS and exactly the two artifact paths. Verify source identities against anchor and sealed confirmation unchanged. If verification fails, stop for review; do not amend a source into the artifact-only seal.
