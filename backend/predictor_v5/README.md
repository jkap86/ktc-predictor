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
all of them, filled in. Ordinary exclusions are counted, not hidden, in
`<label file>.provenance.json`: per season, the market rows, labelled rows,
pre-NFL seasons, players without QB/RB/WR/TE stats (all of 2019, which the
archive lacks), rows below 5 trades, missing next-board values and labelled
rows by position, plus warnings when a season with stats joins nothing or a
position is empty. The same file records the model version, generation ids
and revisions per season, export time, and SHA-256 of the raw export, label
file and training data. Both files are built and checked in a staging
directory, then moved into place together; if either move fails the previous
CSV and provenance are put back, so a failed build never leaves a mismatched
pair.

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
candidate eligible for promotion. The committed CSVs predate the export's
provenance columns, so their generation ids are not recorded.

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
(2025: reused, because PR #3 picked a window from its scores) or from
`--holdout-status` (`untouched` needs `--holdout-evidence`, and a season on
record can't be redeclared). Any other season is `unknown`, which, like
`reused`, keeps `requires_unseen_season_confirmation` true. Every training
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

## MFL-inclusive labels (2026-10-21, schema 1): gate failed, model unchanged

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

## Schema-2 re-evaluation (2026-10-22): still retain

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
