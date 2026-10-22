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

The MFL-inclusive label refresh is reproducible from the local thelab database:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f predictor_v5/sql/backchannel_labels.sql \
  --csv -o /tmp/backchannel_labels_raw.csv
python -m predictor_v5 build-labels \
  --values /tmp/backchannel_labels_raw.csv \
  --out predictor_v5/data/backchannel_labels_2019_2025_mfl.csv
python -m predictor_v5 compare-labels \
  --baseline-values predictor_v5/data/backchannel_labels_2019_2025.csv \
  --candidate-values predictor_v5/data/backchannel_labels_2019_2025_mfl.csv \
  --out predictor_v5/artifacts/backchannel_mfl_comparison.json
```

`build-labels` joins the exported values to `training-data.zip`, applies the
same pre-NFL exclusion as model training, rejects duplicate player-seasons,
and writes the established 14-column label contract. It deliberately keeps
players with thin trade evidence; `compare-labels --min-trades 5` applies the
evaluation filter later. The comparison evaluates the old 2022–2024 control
and new 2022–2024, 2021–2024, and 2020–2024 candidates on 2025, including
all-row, matched-player, position, and coverage metrics. Candidate selection
ranks matched-cohort skill first and prefers the shorter, more recent window
when skills are within one percentage point; the deployment gate is applied
only after that selection.

## Model

One `HistGradientBoostingRegressor` for all positions, target
`log(end / start)`, absolute-error loss, features `log(start)`, age, points,
position code. Predictions never fall as points rise: the tree is
monotonic-constrained in points and `predict` takes a running max over a
1-point grid, because absolute-error leaf refits can otherwise dip by a few
value points.

## Evaluation (KTC labels, sklearn 1.4)

Each season Y is predicted by a model trained only on seasons before Y.
Skill is `1 - model MAE / no-change MAE`.

| Slice | Skill | n | Model MAE | No-change MAE |
|---|---|---|---|---|
| All | 28% | 1582 | 453 | 624 |
| 2021 | 10% | 181 | 688 | 766 |
| 2022 | 24% | 241 | 512 | 668 |
| 2023 | 26% | 313 | 496 | 672 |
| 2024 | 31% | 384 | 410 | 596 |
| 2025 | 37% | 463 | 337 | 538 |
| QB | 27% | 270 | 542 | 744 |
| RB | 28% | 413 | 457 | 634 |
| TE | 26% | 284 | 438 | 587 |
| WR | 28% | 615 | 419 | 583 |

Full output: `artifacts/ktc_eval.json`. 2021 is weakest because it trains on
2020 alone.

## Backchannel holdout (train 2022–24, test 2025)

```bash
python -m predictor_v5 --labels backchannel:backchannel_labels_2019_2025.csv --min-trades 5 \
  holdout --train 2022-2024 --test 2025
```

Half of the Backchannel rows have no trades in the season, so their end
value equals their start value (the prior). `--min-trades 5` keeps rows with
at least 5 trades in both the start and end windows. That leaves 911 train
and 361 test rows. `--min-trades` and `--end-column` are rejected with
`--labels ktc`.

Skill is the comparison to read: MAE is on each source's own scale, and
Backchannel values are much lower than KTC's. Because the filter selects on
trading activity at season end, the Backchannel skill applies to actively
traded players only, not to every player.

| Labels | Skill | Test rows | Model MAE | No-change MAE |
|---|---|---|---|---|
| KTC | 35% | 463 | 352 | 538 |
| Backchannel `end_value`, ≥5 trades | 44% | 361 | 118 | 209 |
| Same 353 players, KTC | 36% | 353 | 399 | 622 |
| Same 353 players, Backchannel | 44% | 353 | 119 | 212 |
| Backchannel `end_value_next_board`, ≥5 trades | −16% | 338 | 390 | 335 |

The end label is `end_value`. `end_value_next_board` comes from the next
season's board, whose overall level shifts by season (median next-board /
start ranged 0.72–1.02 across 2020–25). A model trained on 2022–24 can't
learn the 2025 shift, so it loses to no-change.

Notes from development:

- Points carry the signal. Without them the same model is worse than
  no-change (MAE about 647).
- Adding games played (points per game plus games) cut MAE to about 426. It
  isn't an input yet because the ask was season points only.

## MFL-inclusive labels (2026-10-21): gate failed, model unchanged

`data/backchannel_labels_2019_2025_mfl.csv` was built as above after
thelab's market history (`baseline-hl28-v1`) was revised with MFL dynasty
trades: 2019–2021 loaded in full (two 1,500-league slices each), 2022–2025 as
120-league samples. The old CSV is unchanged and stays the control.
`artifacts/backchannel_mfl_comparison.json` is the full, deterministic report
(`compare-labels --min-trades 5`; two runs are byte-identical).

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
`backchannel_labels.sql`'s `model_version` changed to `baseline-hl28-v3`, into
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
