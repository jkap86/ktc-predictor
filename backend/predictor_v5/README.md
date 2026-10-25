# predictor_v5: season points → end-of-season value

A what-if model. You give it a player's start-of-season dynasty SF value,
age, position and the season fantasy points you expect them to score; it
returns the predicted end-of-season value. The horizon is week 1 to the end
of the regular season: KTC start values are from early September and end
values from about the last regular-season week (Dec 30 to early Jan);
Backchannel values are at week-1 kickoff and the last regular-season Sunday. There is no points forecaster:
the points are your input.

```bash
cd backend
python -m predictor_v5 evaluate      # season folds vs "value doesn't change"
python -m predictor_v5 train         # fit on every season, writes artifacts/ktc.joblib
python -m predictor_v5 predict --position WR --age 24 --start 5200 --sweep 100,175,250,325
python -m pytest predictor_v5/tests
```

On small containers, `OMP_NUM_THREADS=1` makes training and tests much faster.

## Data

- **Stats** (position, age, season points) come from `data/training-data.zip`.
  Points are full-PPR season totals (4 pt passing TD).
- **Labels** (`start_value`, `end_value`) come from a label source. Rows where
  either value is missing or zero are dropped. KTC gives 1,727 labelled
  player-seasons for 2020–2025.
- **College seasons are dropped.** KTC values devy players, so the zip has
  0-point seasons before the draft, such as Caleb Williams 2023. In the zip a
  rookie NFL season has `years_exp` 1, so a season with `years_exp` 0 and no
  games is college (169 labelled rows). Inactive rookies and injured or
  backup seasons, such as Jordan Love 2020, stay.

## Swapping labels

`labels.py` defines the contract: one row per `(player_id, season)` with
`start_value` and `end_value`, keyed by Sleeper player id.

- `--labels ktc` (default): KTC start/end values from the training zip.
- `--labels backchannel:path.csv` (or `csv:path.csv`): any CSV with
  `player_id, season, start_value, end_value`. For Backchannel, export
  `market_history_values.value` at each season's start and end cutoffs. This
  is the stored reference-scale value; the published value is renormalized to
  that day's top asset.

Outputs are clipped to the source's scale (9,999 for KTC, 10,000 for CSV).

The label refresh is reproducible from the local thelab database without
editing any SQL. `sql/backchannel_labels.sql` takes the market model version
as a psql variable (default `baseline-hl28-v1`, the version the committed
labels and the season tool use):

```bash
# v1 (default) and v3 exports
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f predictor_v5/sql/backchannel_labels.sql --csv -o /tmp/raw_v1.csv
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -v model_version=baseline-hl28-v3 \
  -f predictor_v5/sql/backchannel_labels.sql --csv -o /tmp/raw_v3.csv

python -m predictor_v5 build-labels --values /tmp/raw_v1.csv \
  --out predictor_v5/data/backchannel_labels_2019_2025_mfl.csv --model-version baseline-hl28-v1
python -m predictor_v5 build-labels --values /tmp/raw_v3.csv \
  --out predictor_v5/data/backchannel_labels_2019_2025_mfl_v3.csv --model-version baseline-hl28-v3

python -m predictor_v5 compare-labels   # defaults: old v1 control, MFL v1 candidate, MFL v3 for the model-version experiment
```

The SQL reads the newest revision of the chosen model's generation at each
cutoff. It stops with an error naming the season if a start or end
generation is missing, has no dynasty superflex board, or shares its newest
revision with another generation; the next-season board is optional (v3 has
none on 2026-01-04). Each row also carries the model version, generation ids
and revisions, board sizes and the export time.

`build-labels` joins the exported values to `training-data.zip`, applies the
same pre-NFL exclusion as model training, and writes the established
14-column label contract. It deliberately keeps players with thin trade
evidence; `--min-trades 5` applies the evaluation filter later. Before
joining it fails on integrity errors: duplicate player-seasons, missing or
negative required values, an expected season (default 2019–2025) with no
rows, mixed or unexpected model versions, or more than one start or end
generation in a season. An export carrying any provenance column must carry
all of them, filled in and consistent: one start, end and next generation id
and revision per season (next either set for every row of a season or for
none), positive whole board sizes that agree within a season, and a single
ISO 8601 export timestamp. Ordinary exclusions are counted, not hidden, in
`<label file>.provenance.json`: per season, the market rows, labelled rows,
pre-NFL seasons, players without QB/RB/WR/TE stats (all of 2019, which the
archive lacks), rows below 5 trades, missing next-board values and labelled
rows by position, plus warnings when a season with stats joins nothing or a
position is empty. A build where no market row joins at all (for example, ids
that don't match the archive) is an error, reported with those counts, and
nothing is written; partial coverage and the 2019 gap are fine. The same
applies when every joined row has a zero start or end value, because training
keeps only rows with both values above zero; that error gives the zero-start,
zero-end and both-zero counts. Zero-valued rows stay in the file whenever at
least one row is trainable.
The same file records the model version, generation ids
and revisions per season, export time, and SHA-256 of the raw export, label
file and training data. Both files are built and checked in a staging
directory, then moved into place together; if either move fails the previous
CSV and provenance are put back, so a failed build never leaves a mismatched
pair.

**Duplicate keys.** `compare-labels` refuses any baseline, candidate or
model-effect file with a repeated (player_id, season), before the min-trades
filter and before loading stats or fitting, naming the file, the number of
repeated keys and a sample. It checks the joined frames again, so a repeat
from the stats join is caught too. Nothing is deduplicated.

**Which model version a label file is.** `compare-labels` never takes a
version on trust. It accepts a file's version from its `.provenance.json`
when the SHA-256 recorded there matches the file and the version came from
the export's own `model_version` column, or from `data/label_manifest.json`,
which lists the committed CSVs that predate sidecars by SHA-256 (file names
are ignored; editing a file drops it from the list). Sources that disagree,
stale or incomplete sidecars, and files in neither place are unverified.
`--baseline-model-version` and friends are assertions: a mismatch with a
verified version is an error, and on an unverified file they only label an
informational comparison. An unverified or unknown version can never make a
candidate eligible for promotion. An export without the provenance columns gets
`model_version: null` in its sidecar; a `build-labels --model-version` given
for it is stored as `asserted_model_version` and never counts as verified.
The unchanged old control CSV predates those provenance columns. Refreshed
MFL CSVs carry checked-in sidecars recording their actual generation ids;
the manifest retains the historical pre-refresh hashes without relabelling them.

### How `compare-labels` chooses and judges a candidate (schema 2)

Selection and final evaluation are separate stages:

1. **Selection** uses only seasons before the test season. Each candidate
   window length (3, 4 or 5 seasons; `--train-starts` keeps its old meaning)
   is trained on the seasons before a validation season (default 2024, so
   2021–23 or 2020–23) and scored there. A length is eligible only if every
   season of its window has data, so 5 seasons (which would need 2019) is
   ruled out rather than scored on a shorter window. Highest validation
   skill wins; lengths within 1 point of it go to the shortest. Test-season
   rows are removed before this stage runs.
2. **Freeze**: label file, model version, window, `min_trades`, features,
   model parameters and evaluation targets are recorded in the report.
3. **Final evaluation**: the frozen candidate (trained through 2024) and the
   deployed control (old labels, 2022–24) are scored on 2025.

Head-to-head numbers use one target at a time: each model's predicted
relative change (prediction / its own start value) is applied to the
target label set's start value and scored against that set's end value, on
the (player, season) keys both predictors have. Both predictors then share
the same no-change MAE, so MAE and skill compare directly. The gate (+2
points overall, ≥30 rows per position, no position more than 5 points down)
must pass on the control's target *and* the candidate's. Every threshold
(the gain, the position drop, and selection's 1-point tie margin) compares
full-precision skill; the report rounds skill to 0.001 for display and keeps
`skill_exact` beside it.

The `decision` block separates what passed from what may happen:

- `gate_passed`: every guard passed on every target.
- `promotion_eligible` (and `action: update`): the gate passed on two label
  files whose shared model version is verified. This is a recommendation.
- `evidence_sufficient_for_promotion`: also needs an untouched test season.
- `promotion_authorized` is always false: the tool never deploys; promotion
  is a manual change to `season.py`.

Test-season history comes from `KNOWN_HOLDOUTS` in `compare_labels.py`
(2025: reused, because PR #3 picked a window from its scores). That
registry is the only verified source: an entry gets there through a
reviewed commit with evidence someone can check. `--holdout-status` (with
`--holdout-evidence` for `untouched`) records a declaration as
`verified: false`, so a declared untouched season never makes
`evidence_sufficient_for_promotion` true; a season on record can't be
redeclared. Any other season is `unknown`. `requires_unseen_season_confirmation`
stays true unless the season is untouched *and* verified.

If the control and candidate share no test player-seasons, the final
evaluation is marked `insufficient_overlap`, the gate is not evaluated and
the decision is `retain`; informational experiments with no overlap, or
whose optional label set lacks the test season, are marked
`insufficient_overlap` / `insufficient_data` and the rest of the report
carries on. Reports are written with `allow_nan=False`. Every training
window, control included, must have rows in each of its seasons, or the run
stops with the label file and missing seasons named; `fits` records the rows
each model trained on.

Experiment A (`label_effect`) trains both label sets on the same
(player, season) keys, those both keep after the 5-trade filter, so only
the label values differ; `training_population` gives the shared and
one-sided row counts. `label_and_population_effect` is the operational view
in which each set trains on all its rows. Neither is used for selection.

The schema-1 reports (`artifacts/backchannel_mfl_comparison.json`,
`artifacts/backchannel_mfl_v3_comparison.json`) are kept as committed for
the record; they selected the window on 2025 and compared skills measured on
different targets. Schema-2 output goes to
`artifacts/backchannel_label_experiments_v2.json`.

## Historical MFL-inclusive labels (2026-10-21, schema 1)

This section records the pre-hardening experiment, not the current refreshed
CSV contents. Its schema-1 reports remain unchanged as historical evidence.

`data/backchannel_labels_2019_2025_mfl.csv` was built as above after
thelab's market history (`baseline-hl28-v1`) was revised with MFL dynasty
trades: 2019–2021 loaded in full (two 1,500-league slices each), 2022–2025 as
120-league samples. The old CSV is unchanged and stays the control.
`artifacts/backchannel_mfl_comparison.json` is the full, deterministic report
from the schema-1 `compare-labels` (superseded; see the schema-2 section
below).

Average dynasty superflex trades per fit, latest revision, before → after MFL:

| Season | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Before | 171 | 868 | 1,904 | 3,623 | 6,274 | 12,720 | 23,046 |
| After | 4,394 | 5,184 | 6,984 | 3,913 | 6,573 | 12,929 | 22,597 |

2025 went down because the local Sleeper data was re-fetched live after an
accidental truncation, and the re-fetch has fewer 2025 leagues: 44 of its
pre-MFL dates were revised, against none in 2019–2024. The before figures are
from that pre-truncation database; the 2025 figure the original plan quoted
(19,936) does not match them.

Test 2025, ≥5 trades. Skill on all rows / on the 361 players present under both
label sets:

| Labels | Train | Train rows | All rows | Matched | QB | RB | TE | WR |
|---|---|---|---|---|---|---|---|---|
| Old (control) | 2022–24 | 911 | 43.6% (361) | 43.6% | 43.6% | 39.1% | 34.8% | 49.3% |
| MFL | 2022–24 | 927 | 43.6% (363) | 43.2% | 44.4% | 35.8% | 35.3% | 49.7% |
| MFL | 2021–24 | 1,249 | 44.9% (363) | 44.5% | 47.2% | 39.9% | 33.1% | 49.2% |
| MFL | 2020–24 | 1,537 | 45.3% (363) | 44.9% | 51.0% | 40.1% | 31.5% | 47.9% |

Position columns are matched-cohort skill (QB 67, RB 101, TE 56, WR 137
rows). Rows surviving the filter per season, old / MFL: 2020 160 / 288, 2021
229 / 322, 2022 273 / 280, 2023 297 / 304, 2024 341 / 343, 2025 361 / 363.
There is no 2019 row in either training set: `training-data.zip` has no 2019
stats.

2021–24 is selected (within a point of 2020–24, and shorter). It beats the
control by 0.9 points, short of the 2-point bar; every position has ≥30 rows
and none loses more than 5 points (TE −1.7). **Decision: retain.** The season
tool still trains on `backchannel_labels_2019_2025.csv`, and
`data/start_values_2026.csv` was not re-exported. Re-running
`sql/start_values.sql` today does change it, but only because the newest
2026 generation is now `baseline-hl28-v3` (thelab's new default), not
because of MFL.

### The same refresh on `baseline-hl28-v3`

thelab's default model is now `baseline-hl28-v3`, so the refresh was repeated
on it: v3 history generated for the same 2019–2025 windows (same MFL corpus,
so the trades per fit match v1's exactly), labels exported with
`backchannel_labels.sql`'s `model_version` changed to `baseline-hl28-v3` (now
`-v model_version=baseline-hl28-v3`), into
`data/backchannel_labels_2019_2025_mfl_v3.csv`, and compared against the same
old v1 control (`artifacts/backchannel_mfl_v3_comparison.json`). v3 has no
2026 board on 2026-01-04, so its 2025 rows have no `end_value_next_board`.

| Labels | Train | All rows | Matched | QB | RB | TE | WR |
|---|---|---|---|---|---|---|---|
| Old v1 (control) | 2022–24 | 43.6% | 43.6% | 43.6% | 39.1% | 34.8% | 49.3% |
| MFL v3 | 2022–24 | 35.6% | 35.2% | 46.5% | 24.0% | 23.1% | 38.6% |
| MFL v3 | 2021–24 | 43.1% | 42.7% | 50.1% | 35.1% | 29.9% | 46.8% |
| MFL v3 | 2020–24 | 41.1% | 40.8% | 46.1% | 32.2% | 26.3% | 47.3% |

2021–24 is selected and is 0.9 points *below* the control; RB (−4.0) and TE
(−4.9) come close to the per-position limit. **Decision: retain.** v3's
values sit on a different scale, so its lower MAEs are not comparable; read
skill. This compares MFL + v3 against neither change, so it cannot separate
the two effects.

## Historical schema-2 re-evaluation (2026-10-22)

The numbers in this section describe the pre-remediation labels. The current
schema-2 report is regenerated by the hardened-corpus refresh below; the old
report remains recoverable in Git history.

`artifacts/backchannel_label_experiments_v2.json`, from `compare-labels`
with its defaults (two runs are byte-identical). Same label files as above;
nothing was re-exported.

All three label files verify through `data/label_manifest.json` (old and MFL
v1: `baseline-hl28-v1`; MFL v3: `baseline-hl28-v3`).

**Selection (2024 validation, MFL v1 labels):** 2020–23 scores 48.51%, 2021–23
46.30%; 5 seasons is ineligible (no 2019 stats). The 4-season window is
frozen, so the final candidate trains on 2021–24, the same window PR #3
picked from 2025.

**Final, 2025, 361 matched player-seasons:**

| Target | Control (old, 2022–24) | MFL v1 (2021–24) | Gain | No-change MAE |
|---|---|---|---|---|
| Old v1 labels | 43.6% (MAE 117.6) | 44.9% (MAE 114.9) | +1.26 | 208.5 |
| MFL v1 labels | 43.2% (MAE 117.7) | 44.5% (MAE 115.0) | +1.32 | 207.2 |

Gains are full precision. Short of the 2-point bar on both targets; every position has ≥30 rows and
none drops more than 5 points (TE −2.1 / −1.6). **Decision: retain**
(`gate_passed` false; 2025 is a reused holdout regardless).

**MFL label effect, same training rows** (Experiment A, same players, same
target): small and inconsistent. Old-label target skill, old vs MFL labels,
both trained on the rows both keep: 2022–24 43.6% vs 44.7%, 2021–24 45.5% vs
44.2%, 2020–24 46.6% vs 45.8% (MFL target: 43.2/44.4, 45.2/43.9, 46.3/45.3).
Old labels are a subset of the MFL rows; MFL adds 16 rows over 2022–24, 93
in 2021 and 128 in 2020. Letting each set train on all its rows
(`label_and_population_effect`) gives 43.6/43.7, 45.5/44.9, 46.6/45.3. Either
way, the gain the PR #3 report credited to MFL comes from the longer window,
and old labels gain at least as much from it. Selected on 2024, though, the
old labels would keep 3 seasons (41.6% vs 39.9% for 4), so the longer
old-label windows are not a validated choice either (Experiment B).

**v1 vs v3 (Experiment C, informational):** MFL v1 and MFL v3 trained on
2021–24 each win on their own target (v1 target: 44.9% vs 39.4%; v3 target:
43.1% vs 36.7%), so the target choice decides the result and the comparison
cannot rank the versions. v3 labels without MFL were never exported, so the
model-version effect without MFL is unresolved.

Limitations: 2025 has been looked at before (see above). Jay's local MFL rows
were loaded before backchannel-thelab #278 added its skip for leagues with
more than one copy of a player, so the MFL label files may include such
leagues; reload before any future MFL gate. 2022–2025 MFL data are
120-league samples. The thelab database is only on Jay's Mac, so the SQL was
tested against a disposable local Postgres, not re-run on real data.

## Hardened-corpus refresh (2026-10-24 EDT): numerical gate passes; no deployment

This refresh actually applied the hardened MFL semantics to local `thelab`,
then rebuilt both histories and both versioned label files. Backchannel's
durable evidence is in `mfl-remediation-2026-10-23/progress.md` on branch
`codex/mfl-hardened-remediation`; the narrow reversible quarantine tool is
commit `400f28ce132fdf429a3170b1f72a6b8a9d64bb12`.

The original 52,309 MFL trades became 47,314 stored trades after the reviewed
remediation excluded 133 league-seasons / 4,995 trades (52 multi-copy leagues
/ 2,788 trades; 81 uncertain leagues / 2,207 trades). Backups preserve the
original rows. Another 29 historical league-seasons / 968 trades were retained
through verified later-season vouches. The hardened reload populated keys and
known single-copy metadata for all remaining eligible trades.

Under the user's conservative ruling, exactly 13 still-unverified
league-seasons / 205 unkeyed trades were then **reversibly quarantined**, not
deleted: 12 unsupported future-pick classifications and one invalid-ID
fixture-like row. All 47,109 eligible MFL trades are now keyed, classifier 2,
and known single-copy; there are no source-key collisions or duplicate trade
shapes. Historical Sleeper populations are unchanged.

Source completeness is deliberately not claimed: 426 partial transaction
responses kept their stored data under the existing reconciliation guard.
Those failed outcomes remain recorded; no fabricated vouches or relaxed
empty-sided-trade parsing was used. The 2022–2025 MFL sample limitation remains.

| Season | MFL stored trades | Eligible SF leagues / trades | Latest average SF trades, v1 and v3 |
|---|---:|---:|---:|
| 2019 | 12,811 | 518 / 12,723 | 3,830.37712895 |
| 2020 | 14,176 | 702 / 14,111 | 4,729.57524272 |
| 2021 | 16,781 | 838 / 16,730 | 6,564.08515815 |
| 2022 | 1,045 | 42 / 1,045 | 3,909.94403893 |
| 2023 | 1,014 | 51 / 1,014 | 6,567.40145985 |
| 2024 | 715 | 44 / 714 | 12,923.25485437 |
| 2025 | 772 | 44 / 772 | 22,585.12895377 |

Each history has all 2,879 expected dates (411/year, 412 in 2020 and 2024).
The refresh appended 2,295 generations per version: v1 revision 3 and v3
revision 2 where inputs changed; unchanged dates keep their earlier highest
revision. All 8,166 pre-existing generation/fit/value row hashes, and all
revision-1 control dates/averages, were preserved. Both 2025 runs encountered
the generator's default heap limit after December 2; supported bounded-range
resumes completed, with that overlap unchanged and separate retained logs.
All fourteen endpoint dry-runs were unchanged afterward.

The pinned export SQL selected the highest revisions in explicit read-only
transactions. Both CSVs contain 3,560 unique player-seasons, the same
14-column schema, and 1,669 thin rows retained for later filtering. Rows by
season 2020–2025 are 457, 540, 587, 631, 634, 711; rows after the five-trade
evaluation filter are 281, 320, 280, 304, 343, 363. The 2019 market exists,
but the training archive still has no 2019 stats. Sidecars record all gaps,
generation ids/revisions, raw-export hashes and training-data identity.
Each CSV/sidecar pair and each comparison report reproduced byte-for-byte.

**v1 selection and gate:** the 2024 validation skills are 46.7279% for three
training years and 47.9621% for four. Five years is ineligible because its
validation window needs 2019 stats. The existing one-point/shortest-window
rule selects four years, freezing the final 2021–24 candidate before 2025
evaluation. All three requested final windows were also scored informationally:
candidate skills on old / refreshed common targets are 44.0431% / 43.5428%
(2022–24), 46.1142% / 45.6419% (2021–24), and 45.6223% / 45.0694% (2020–24).

| 2025 target, 361 matched rows | Old-label control, 2022–24 | Refreshed v1, 2021–24 | Gain, points |
|---|---:|---:|---:|
| Old labels | 43.605958% | 46.114244% | +2.508287 |
| Refreshed labels | 43.053240% | 45.641904% | +2.588664 |

Both +2-point guards pass at full precision. Position samples are QB 67,
RB 101, TE 56, WR 137; all exceed 30. The largest drop is TE on the old
target, −0.871829 points, within the five-point guard. The report therefore
records `gate_passed: true`, `promotion_eligible: true`, `action: update`.
**That is an experimental recommendation, not deployment authorization:**
2025 is a verified reused holdout, so `evidence_sufficient_for_promotion` and
`promotion_authorized` remain false, and unseen-season confirmation is still
required. `season.py`, model behavior, API and UI remain unchanged.

The refreshed v3 report is
`artifacts/backchannel_label_experiments_mfl_v3_v2.json`. It selects three
years under the tie rule (2024 validation 45.4536% vs 45.8491%) and scores
35.4784% on its own 363-row 2025 target. Its four-/five-year informational
skills are 41.7388% / 40.5919%. Comparison against the old v1 control is
cross-version, so the deployment gate is **not evaluated**, not a valid
same-target win or loss. Historical schema-1 reports remain unchanged.

SHA-256 identities:

- Old control CSV, unchanged: `eb8eab7365746b408d188a1fa2a78a20c6085a7650782a83abf512b7160491f5`.
- Refreshed v1 CSV: `3e57be0b6e6ccb8f626a8b4ee1e726eaad75f635174d147514c3aa8ccf3c8e89`.
- Refreshed v3 CSV: `0e2de95e23b0abd3a57ee9fbc0cd7035d10172a0dce41f496eef6cb5d1a39155`.
- v1 schema-2 report: `ee78f8decbfff7edfd6e9007d4f7794f1d5b7cfd0fc869874c59f1c093940f43`.
- v3 schema-2 report: `5bc7d7b9158e6486dc7be1dcefd8f7f56e20dc91912f162354ce44350b97d366`.

Canonical `sql/start_values.sql` has no semantic change from base commit
`d91bad38c01bc74612a7aae571e9d9d54ceb4cba`. Live player-name/team/membership
drift is not grounds for a rewrite, so `start_values_2026.csv` remains
`6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb`.
Verification: all 131 predictor tests passed, including marked scratch-DB
SQL tests; Ruff passed. The optional-PyArrow warning remains non-fatal.

## Independent 2026 MFL-versus-KTC replacement protocol

`predictor_v5.ktc_replacement_2026` is a separate, pre-outcome protocol.
KTC is the incumbent; verified `baseline-hl28-v1` MFL labels are the candidate.
Both models train on exactly 2021–2025 and score the same liquid matched
2026 player cohort, each against its own market outcomes and no-change
baseline. Start is 2026-09-09 and end is 2027-01-10, America/New_York;
KTC end observations must lie in the inclusive 2026-12-28–2027-01-10 window.
Positive KTC values must be below 9999; positive MFL values require at least
five trades at both endpoints. There is no imputation or source substitution.

The gate uses full-precision MFL-minus-KTC normalized skill: at least -0.01
overall, at least -0.05 in each position, and at least 30 matched players in
each of QB/RB/TE/WR. Equality passes. Missing/invalid scoring evidence retains
KTC. A pass recommends MFL primary only for eligible players, with KTC fallback
when MFL evidence is missing or insufficient. Every result has
`promotion_authorized: false`. Any promotion requires a later, separately
reviewed manual runtime change; removing KTC fallback requires separate
coverage evidence. Results do not establish performance for excluded players.

Bootstrap output is the **paired raw MFL-minus-KTC MAE difference**, with
10,000 resamples, seed 20261024 and percentile interval [2.5, 97.5]. It mixes
the sources' distinct raw scales and is a diagnostic, not normalized skill
or a promotion gate. Rank diagnostics use independent descending average
ranks, Spearman correlation and raw mean absolute rank error. Bootstrap,
rank, residual, and coverage diagnostics are non-gating and must not choose
configuration or promotion.

### Source review and pre-outcome sealing

Current work supplies the generator, runbook and synthetic tests only. It
does not register real 2026 inputs, inspect outcomes, run either protocol,
or authorize deployment. The existing confirmation's eight bound sources,
manifest and registry remain protected by PR #6 baseline
`1157678526d3408850780c7ec7808a730d8c2dbc`; its manifest SHA-256 is
`c376651de489d0ae5c6478c001d565bb635d6968b46df4a16e99875d9da7430e`.

The previous replacement seal at `8800b7c2937aa22d48a57789f5c97005d60f2f30`
was superseded by the final-review fix wave. Its two replacement JSON files
were explicitly removed; Git history preserves the earlier evidence. They
must not be restored as a valid seal over the changed source. A newly reviewed
source-anchor commit and its new artifact-only child are required before
registration or evaluation. No new replacement artifacts are supplied in
this Phase A fix.

An independent reviewer must approve the committed source anchor, including
the seal generator and its tests, before actual sealing. The anchor must be
the exact full committed HEAD SHA and the tracked worktree/index must be
clean. The generator checks required regular-file identities against Git
blobs and working bytes, training archive member, verified MFL v1 provenance,
frozen starts and the actual canonical mapping snapshot. Untracked review notes are permitted;
either existing output (including a dangling symlink) prevents sealing.
Neither artifact contains future target hashes.

After source approval, from `backend`, supply the approved anchor and explicit
reviewed UTC timestamp (canonical seconds with `+00:00`):

```bash
python -m predictor_v5.replacement_seal \
  --source-anchor <reviewed-full-commit-sha> \
  --sealed-at <reviewed-YYYY-MM-DDTHH:MM:SS+00:00>
```

The CLI has no path, data, model or gate overrides. `build_manifest(root,
source_anchor)` is a deterministic **untimestamped template**, not a final
valid manifest. `write_seal` adds the supplied `sealed_at`, validates the
complete manifest, and passes it to `build_registry`, which requires an
identical timestamp. Time is never inferred from Git or the wall clock.
The registry starts with one `sealed` entry keyed by the SHA-256 of the final
canonical manifest bytes. The output files are:

- `data/ktc_replacement_2026_manifest.json`
- `data/ktc_replacement_2026_holdout_registry.json`

Inspect both and commit **only those two new JSON files**, with the approved
source anchor as the commit's sole parent. Run
`python -m pytest predictor_v5/tests/test_replacement_seal.py -v` and check
`git diff --name-status HEAD^ HEAD`: exactly those two additions are allowed.
Then run the dedicated strict Phase B handoff check from `backend`:

```bash
python -c 'from predictor_v5.tests.test_replacement_seal import ROOT, assert_seal_handoff; assert_seal_handoff(ROOT)'
```

This handoff check requires current HEAD to be the artifact-only child. The
normal suite instead identifies the historical seal and verifies its exact
sole parent, two-artifact diff, committed/working bytes, and source hashes.
Unrelated later commits and merges are allowed; uncommitted substitute pairs
or changed sealed source bytes still refuse.
Before generation, only the real artifact assertion skips; all synthetic
generator and protected-baseline checks must pass. Do not amend source into
the artifact commit. If checks fail, stop for independent review.
Publication never replaces a file; caught failures remove only files the
call created. Two file publications cannot form one filesystem transaction:
a process crash may leave a partial seal. Preserve that evidence and stop
for review; do not overwrite it or silently resume generation.

The frozen start CSV has SHA-256
`6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb`,
Git blob `ead24b508102ca2a1071d07f3f665e9ea920aa96`, introduced at `c415177`.
These identities freeze the existing bytes. **Raw start provenance is not
independently verified**: they do not prove its historical export or mapping.
This exception never extends to future KTC end observations.
Reports include `actual_dates.frozen_starts`: all frozen player keys sorted
by `(season, player_id)`, with actual `bc_date` and `ktc_date` (null where the
frozen source observation is absent), alongside the end observations/cutoffs.
Reporting these dates does not upgrade the raw-start provenance limitation.

The separate `data/ktc_replacement_mapping_snapshot.csv` freezes 532 dynasty
`(format, ktc_id, sleeper_id)` rows read from the local database
`postgres://localhost:5432/thelab` on 2026-10-25, after schema inspection.
Its SHA-256 is
`e4f4434801fcfab7ffa7c1d0d0ab8e74486329c39f0d44a649cb4757c1edc742`.
Only the mapping population was read, with this SELECT:

```sql
SELECT format, ktc_id, sleeper_id
FROM public.ktc_values
WHERE format = 'dynasty' AND sleeper_id IS NOT NULL
ORDER BY format COLLATE "C", ktc_id, sleeper_id COLLATE "C";
```

The canonical file has exactly those three columns, UTF-8 with LF records,
minimal CSV quoting, numeric KTC-ID ordering, and one-to-one nonempty keys.
No history dates or market values were queried to create it. The manifest's
`recipes.ktc.mapping_data` binds these bytes; SQL and mapping code have separate
identities. Registration and the KTC sidecar must bind the same snapshot.
Every future raw row, including rows excluded from scoring, must have an
identical frozen source-to-Sleeper mapping. New, missing, or changed mappings
refuse; do not refresh this snapshot after reviewing outcomes to make a run
pass. The future raw export must still contain every candidate in the fixed
window, and independent review must check completeness.

### Later independent review, registration and one-shot execution

Before either protocol opens outcomes, an independent joint review must name
both already sealed manifest hashes, their fixed recipes, reviewer, evidence
commit/reference and review timestamp. It must attest that the recipes were
committed before outcomes influenced either protocol and the evidence remains
untouched. That is review/Git evidence, not proof of non-observation. If either
protocol exposed outcomes without this review, do not claim an untouched
comparison. The replacement registry's reviewed paths are `sealed -> untouched
-> used` or `sealed -> reused`; `sealed`, `reused`, and `used` cannot evaluate.

Later, independently register the KTC raw end export, normalized target CSV,
2026-only stats ZIP and canonical provenance sidecar, and the MFL target CSV,
stats and registered sidecar in `data/ktc_replacement_2026_inputs.json`.
Bind their byte hashes and sealed SQL/extractor/mapping/build identities.
The replacement registry and registration hash the immutable joint-review
artifact; confirmation's later review/registration use its existing
`evidence_reference` fields to identify the same artifact. Preserve the
sealed confirmation schemas and recipe. Sharing MFL bytes does not waive
independent registration and validation. None of these real registration or
outcome steps is authorized by the current source/seal work.

Only after that later review and registration, the fixed evaluation command
is `python -m predictor_v5.ktc_replacement_2026` (no overrides). Metadata,
code/training identity, training coverage, joint review and both protocol
registrations are checked before any outcome read. Successful preflight
durably creates the manifest-keyed `.claim.json` under
`artifacts/replacements/` before hashing or opening outcomes; all inputs are
then validated before fitting. A canonical, no-replace report is written
under `artifacts/replacements/2026-<manifest-sha256>.json`.

Both `artifacts/replacements/` and `artifacts/confirmations/` must already
exist as regular directories. Tracked `.gitkeep` files provide them in a
fresh checkout; retain them when packaging the repository. Provision and
durably sync any manually restored directories before preflight. The runner
does not create missing parents during a one-shot attempt; a missing required
path or metadata file refuses with exit 2. General I/O and publication/fsync
failures exit 1; they must not be treated as input corrections.

A preclaim refusal (exit 2) reads no outcomes and permits correction and
review before retry. **Once a claim exists, never delete the claim or retry**,
including after a crash, exit 1/2, input-integrity error, or missing report.
The claim consumes the holdout even while the later manual registry update
is pending. Preserve claim, report (if present) and evidence; independently
review the failure and transition the registry to `used` with their identities.
Recovery requires a separately approved future holdout and new protocol for
a decision, not regenerating this result. Postclaim integrity refusals exit 2;
unexpected fit/I/O failures exit 1. Valid but insufficient cohorts produce a
retain-KTC report and exit 0. No result changes `season.py`, API, frontend,
deployment, or the confirmation registry automatically.

## Season tool (2026)

Type a player's projected season points and get their predicted
end-of-season value, starting from their value on 2026-09-09 (the day before
kickoff). It's on the app's **2026 Season** page (`/season`, API under
`/api/v5/season`) and on the command line:

```bash
python -m predictor_v5 season --player "Ja'Marr Chase" --sweep 150,250,350
```

Each player gets two predictions, each from its own source's start value:

- **Backchannel** (primary): trained on 2022–2025 Backchannel labels with
  ≥5 trades. Players with fewer than 5 trades before the cutoff get no
  Backchannel prediction, because their value is Backchannel's ADP prior.
- **KTC**: trained on 2020–2025 KTC labels.

Start values are in `data/start_values_2026.csv`, exported from the thelab
database with `sql/start_values.sql` (Backchannel model `baseline-hl28-v1`;
522 players with a KTC value plus 6 rostered players with ≥20 trades).

Backchannel's value scale shrinks every season: well-traded stars started
near 8,000 in 2022, 5,000 in 2025 and 2,900 in 2026. Read its predictions as
a change from the player's own start value. Re-levelling each season's values
to a common anchor didn't help on the 2025 holdout (skill 43% vs 45%), so the
model uses the raw values.
