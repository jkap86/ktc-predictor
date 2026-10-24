# 2026 Model Confirmation Design

## Intent

Confirm one refreshed Backchannel candidate on 2026 without allowing 2026
outcomes to tune it. This is a two-stage protocol:

1. A pre-outcome committed manifest freezes every selectable input through
   2025 and the canonical recipe for producing the 2026 target. It cannot and
   does not hash future outcome bytes.
2. After the season, independently reviewed input registration records the
   actual canonical 2026 target identities and holdout attestation. Only an
   `untouched` registration for that exact manifest unlocks one run.

Git is audit evidence, not proof of human non-observation. `untouched`
means outcomes did not influence the frozen protocol.

## Goals

- Freeze through-2025 baseline/candidate labels, training archive, model
  contract, training window, target recipe, diagnostics, and gate before
  outcome evaluation.
- Add only `python -m predictor_v5 confirm-2026`, with no
  configuration-changing CLI inputs.
- Fit old-v1 baseline and refreshed-v1 candidate on exactly 2021--2025 and
  score both against one canonical refreshed-v1 2026 target.
- Fail closed before reading target labels without reviewed untouched
  registration for the same manifest.
- Write one deterministic atomic report; never deploy or alter runtime config.

## Non-goals and anti-goals

- No change to `predictor_v5/season.py`, API, frontend, start-value exports,
  model artifacts, or deployment state.
- No future old/pre-MFL 2026 target and no cross-target comparisons.
- No score while sealed, unknown, reused, missing, or mismatched; refusal
  preserves the one shot.
- Walk-forward and diagnostics cannot choose features, labels, parameters,
  thresholds, or a recommendation.
- A passing result is never promotion authorization.

## Protocol artifacts and lifecycle

All paths are repository-relative. Absolute paths and traversal are rejected.
Every JSON artifact is strict, canonical UTF-8 JSON (sorted keys, compact
separators, newline); unknown or missing fields and noncanonical bytes fail.

### Sealed pre-outcome manifest

`backend/predictor_v5/data/model_2026_confirmation_manifest.json` is
committed before outcome evaluation. Its SHA-256 is `manifest_sha256` and it
is authoritative field-by-field:

Sealing uses two commits to avoid a self-referential Git identity. The
manifest's `sealing_commit` is the immutable source anchor whose bytes are
hashed by `source_identities`; it is not the commit containing the finalized
manifest. `SEALED_ARTIFACT_SHA` names the later Git commit that contains the
canonical manifest, matching registry, and this documentation. The handoff
records `SEALED_ARTIFACT_SHA` outside the protocol artifacts after that commit
exists. `manifest_sha256` is independently the SHA-256 of the canonical
manifest bytes. Neither the future artifact commit SHA nor a Git tag is
self-embedded in the manifest or registry.

| Field | Required value/meaning |
| --- | --- |
| `schema_version`, `protocol` | Version 2, `model-2026-confirmation` identity |
| `baseline` | Old v1 label path, SHA-256, required label columns/filter/provenance |
| `candidate` | Refreshed v1 labels-through-2025 path, SHA-256, required label columns/filter/provenance |
| `training_data` | Archive path and SHA-256 |
| `market_model_version` | Verified `baseline-hl28-v1`, never assertion-only |
| `train_seasons` | Exactly `[2021, 2022, 2023, 2024, 2025]` for both fits |
| `positions`, `label_filter` | QB/RB/TE/WR; exact positive-value and `min_trades: 5` columns/semantics |
| `model` | Exact `FEATURES`, monotonic vector, `HGB_PARAMS`, scale, points-grid step |
| `target_recipe` | Canonical refreshed-v1 2026 SQL/export/build recipe, repo-relative SQL/build paths and SHA-256 identities, output columns, registered-sidecar provenance contract, and separately registered 2026 stats contract |
| `gate` | Full-precision gain `0.02`, per-position count `30`, maximum drop `0.05` |
| `walk_forward` | Exact folds/no-shortening policy below |
| `bootstrap` | Seed `20261024`, `10000` resamples, percentile `[2.5, 97.5]` |
| `registry_path` | `backend/predictor_v5/data/model_2026_holdout_registry.json` |
| `input_registration_path` | `backend/predictor_v5/data/model_2026_confirmation_inputs.json` |
| `output_template` | `backend/predictor_v5/artifacts/confirmations/2026-<manifest-sha256>.json` |

The recipe is reviewable before the season without pretending the target's
future CSV bytes can be frozen.

### Holdout registry

`backend/predictor_v5/data/model_2026_holdout_registry.json` is
schema-versioned (version 2) and keyed by `(season, manifest_sha256)`. Its initial
reviewed 2026 entry is `sealed`, with manifest hash, sealing commit, and
timestamp. A later independent review transitions it from `sealed` to
`untouched` or `reused` and records exact season/hash, reviewer, evidence
reference and commit, review timestamp, and transition source.

Only `sealed -> untouched` authorizes confirmation. Missing, duplicate,
malformed, mismatched, sealed, or reused entries are preflight failures. The
registry must state that this is attestation backed by review/Git evidence,
not proof of non-observation. Sealing commits, reviewers, evidence commits, and
evidence references must be nonempty strings. All sealing/review timestamps
must use canonical ISO-8601 seconds and an explicit UTC offset (for example,
`2027-02-01T12:00:00+00:00`). Review cannot predate sealing. Registry sealing
commit/timestamp must exactly match the manifest. Reviewed entries have
`transition_source: "sealed"`; only `sealed -> untouched` or
`sealed -> reused` is a valid transition. Every review field must match the
input attestation.

### Post-season input registration

`backend/predictor_v5/data/model_2026_confirmation_inputs.json` is committed
later after independent review. It records schema version `2`, season `2026`,
exact `manifest_sha256`, canonical refreshed-v1 target CSV and sidecar
repo-relative paths/SHA-256 values, a separate `target_stats: {path, sha256}`
identity, verified v1 provenance identity, target
columns/filter, reverified target-recipe code identities, and reviewed
holdout-attestation metadata (registry key, reviewer, evidence commit,
timestamp, `untouched`).

It is not a second target. Before opening target labels, the command requires
one matching, reviewed `untouched` registration and registry entry. Missing,
noncanonical, non-v1, unverified, or mismatched registration refuses and
writes no report.

### Registered target stats and provenance sidecar

The pre-outcome training archive remains frozen at its original 2020--2025
bytes. Post-season `target_stats` is a separate archive using the existing
training-players ZIP format, read by `read_season_stats`, and containing 2026
stats only. It is never used for training. Its path is strict repository-relative,
with no absolute, empty, dot, traversal, or backslash components; its SHA-256
is checked before any target CSV rows are read. The default target loader
joins against this archive, requires every target key to join, and verifies
the CSV's `position`, `age`, `games`, and `points` against the joined stats.

The independently reviewed confirmation sidecar is canonical JSON with exactly:

```json
{
  "schema_version": 1,
  "target_csv": {"path": "repo-relative.csv", "sha256": "<64-lowercase-hex>"},
  "target_stats": {"path": "repo-relative.zip", "sha256": "<64-lowercase-hex>"},
  "model_version": "baseline-hl28-v1",
  "verified": true,
  "problems": [],
  "source": "export",
  "identity": "sidecar"
}
```

Both artifact identities must exactly equal registration. The identity must
equal registration's `provenance.identity`. Validation reads precisely the
registered, hash-checked sidecar; it does not call `label_identity` for target
provenance and cannot consult an adjacent, unregistered provenance file.
This confirmation sidecar is a distinct reviewed protocol artifact derived
from the build-labels export provenance; an unchanged build-labels coverage
sidecar is not a substitute.

The post-season operator must produce the canonical target using the frozen
recipe and separately supplied 2026 stats, independently review its export
provenance, register CSV/stats/confirmation-sidecar paths and hashes, and
record matching untouched review metadata in registry and input registration.
The extra stats artifact and version-2 manifest/registry hash change are an
accepted protocol cost; future outcome bytes are never committed now.

## One common target

The baseline fits old/pre-MFL v1 labels and the candidate fits refreshed-v1
labels, both for 2021--2025. Each model's predicted relative change is applied
to the same canonical refreshed-v1 2026 start value and scored against its
same end value on the same matched cohort.

This is a model-only comparison. It avoids depending on a future old/pre-MFL
Backchannel target and avoids target-scale/definition confounding; both
models share target, filter, keys, and no-change MAE.

## Command and preflight

The only invocation is:

```bash
python -m predictor_v5 confirm-2026
```

It accepts no global or confirmation overrides. Success prints the payload
fingerprint to stdout and exits 0. Manifest/registry/input/data/config
preflight failures write concise stderr and exit 2. Unexpected runtime or
atomic-write failures write stderr and exit 1. A core API may inject roots
and output directories for tests; the CLI cannot.

```text
sealed manifest --> registry + reviewed input registration
       |                         |
       +--strict preflight-------+-- untouched, same manifest?
                                           | no: exit 2; do not read target
                                           v yes
through-2025 baseline/candidate + archive --> fixed 2021--25 fits
canonical refreshed-v1 2026 target --------> matched cohort -> gate/diagnostics/report
```

Preflight order:

1. Canonically validate manifest, hash it, derive the sole output path, and
   fail if that final path already exists.
2. Canonically validate exactly one registry entry and input registration for
   `(2026, manifest_sha256)`; require matching reviewed `untouched`
   attestation before opening target labels.
3. Verify baseline/candidate/archive hashes and verified v1 provenance.
   Verify SQL/build source hashes and all fixed model/filter/gate/diagnostic
   values against live code/constants.
4. Verify target CSV/sidecar/stats hashes, exact registered-sidecar v1 provenance,
   target columns and recipe
   identities; only then read raw target rows.
5. Reject duplicate raw `(player_id, season)` keys in every label file
   before filtering, then post-join duplicates. Sort all rows by
   `(season, player_id)` before fitting, matching, resampling, or reporting.
6. Reject raw 2026 or future training rows before any filter. Apply exact
   filters and joins; require at least one finite trainable row for every
   2021--2025 season and QB/RB/TE/WR combination in both fits. Validate every
   walk-forward train/test window before any fit; never shorten a fold.
   Duplicate, leakage, and coverage errors identify source, count, and sample.
   Require canonical 2026 targets.
7. Fit only after preflight. There is no selection stage.

## Decision and report schema

The test cohort is the fixed-filter intersection of baseline and candidate
2026 prediction keys, scored only on the canonical target. The report's
`decision` has:

| Field | Semantics |
| --- | --- |
| `status` | `recommend_promotion` only when gate passes; otherwise `retain` |
| `gate_evaluated` | True only after untouched preflight and valid target |
| `gate_passed` | All single-target guards pass at full precision |
| `promotion_authorized` | Always `false` |
| `reason` | Stable machine-readable outcome/reason(s) |
| protocol hashes | `manifest_sha256`, `input_registration_sha256`, `holdout_registry_sha256` |

The gate passes only when both skills exist, target no-change MAE is nonzero,
gain is at least 0.02, every position has at least 30 matched rows and defined
skills, no position loses more than 0.05, and all cohorts are complete. Zero
no-change MAE, missing skill/position, incomplete position cohort, or empty
overlap is insufficient: after untouched preflight it produces atomic
`retain`, never a recommendation. Valid target data filtered to zero rows
passes preflight; the scorer skips model prediction for empty frames. A pass recommends promotion but retains
`promotion_authorized: false`; any production change is manual and reviewed.

The final fixed-path report is canonical:

```json
{
  "schema_version": 1,
  "payload": { "...": "protocol, inputs, fits, cohort, gate, decision, diagnostics" },
  "payload_sha256": "SHA-256 of canonical UTF-8 payload bytes only"
}
```

There is no self-hash claim: `payload_sha256` hashes payload bytes only.
Callers may externally hash final wrapper bytes. The payload records library
and environment versions informationally; Git revision is optional and
`null` is valid.

Write a temporary file in the final directory, flush and fsync it, atomically
create the final path without replacement, fsync its directory, and remove
the temporary file on error. Existing output is an error, never a cache hit.

## Fixed non-promotional diagnostics

Every diagnostic is tagged `used_for_configuration: false` and
`used_for_promotion: false`.

- Walk-forward folds are exactly `2022:[2021]`, `2023:[2021,2022]`,
  `2024:[2021,2022,2023]`, and `2025:[2021,2022,2023,2024]`. Missing
  required history is insufficient; never silently shorten.
- Report overall/per-position MAE, median absolute error, p90 absolute error,
  signed residual mean/bias, residual summaries, and baseline/candidate delta.
- Report raw, filtered, joined, trainable, one-sided, matched, and missing
  coverage by season/position.
- Report paired candidate-minus-baseline absolute-error delta mean/median,
  win/tie/loss, and deterministic bootstrap interval using manifest settings.

## Failure modes and acceptance

| Condition | Behavior |
| --- | --- |
| Manifest/registry/input missing, noncanonical, duplicate, unknown-field, or mismatched | stderr, exit 2; do not read target; no report |
| Registry sealed/reused/not untouched | stderr, exit 2; preserve one shot |
| Target hash/provenance/recipe/version mismatch | stderr, exit 2; no fit/report |
| Raw/joined duplicate, model/filter/training drift, or 2026 leakage | stderr, exit 2; name source/count/sample; no fit |
| Valid but insufficient confirmation cohort | atomic retain report; never recommend |
| Existing final path | stderr, exit 2; unchanged |
| Serialization/fsync/atomic-create failure | stderr, exit 1; no partial final artifact |

Tests must cover strict schemas/paths/source hashes; every registry lifecycle
and registration mismatch; proof target loading cannot occur before matching
untouched attestation; exact CLI/exit/stream behavior; all hash/provenance,
duplicate, coverage, leakage, and full-precision boundary guards; one-target
cohort semantics; zero-MAE/missing-skill cases; exact walk-forward folds;
diagnostic non-influence; fixed-bootstrap and row-sort determinism;
environment/Git-null reporting; payload-only hashing; one-shot atomic writes;
and no season/API/UI/deployment change.

Acceptance requires a reviewed pre-outcome manifest and sealed registry entry,
then a later reviewed untouched registration for the same manifest. A
successful run may recommend promotion only; deployment remains a separate
manual reviewed change.
