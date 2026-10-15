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
  `reference_value` at each season's start and end cutoffs. Use
  `reference_value`, not the published value, because the published value is
  renormalized to that day's top asset.

Outputs are clipped to the source's scale (9,999 for KTC, 10,000 for CSV).

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
