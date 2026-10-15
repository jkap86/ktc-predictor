# predictor_v5: season points → end-of-season value

A what-if model. You give it a player's start-of-season dynasty SF value,
age, position and the season fantasy points you expect them to score; it
returns the predicted end-of-season value. There is no points forecaster:
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
  either value is missing or zero are dropped. KTC gives 1,674 labelled
  player-seasons for 2020–2025.
- **College seasons are dropped.** KTC values devy players, so the zip has
  0-point seasons before a player's NFL debut (222 labelled rows, such as
  Caleb Williams 2023). A season is kept only if it is the player's first
  season with games played or later. Later 0-game seasons (injury) stay.

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

| Slice | n | Model MAE | No-change MAE | Skill |
|---|---|---|---|---|
| All | 1535 | 460 | 636 | 28% |
| 2021 | 177 | 687 | 773 | 11% |
| 2022 | 234 | 510 | 685 | 26% |
| 2023 | 303 | 510 | 690 | 26% |
| 2024 | 371 | 423 | 606 | 30% |
| 2025 | 450 | 342 | 546 | 37% |
| QB | 259 | 543 | 767 | 29% |
| RB | 397 | 463 | 649 | 29% |
| TE | 280 | 437 | 594 | 26% |
| WR | 599 | 433 | 592 | 27% |

Full output: `artifacts/ktc_eval.json`. 2021 is weakest because it trains on
2020 alone.

Notes from development:

- Points carry the signal. Without them the same model is worse than
  no-change (MAE about 660).
- Adding games played (points per game plus games) cut MAE to about 432. It
  isn't an input yet because the ask was season points only.
