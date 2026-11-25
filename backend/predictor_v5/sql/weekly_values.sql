-- Weekly Backchannel values for the short-window study (short_window.py).
--
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
--     -f predictor_v5/sql/weekly_values.sql --csv -o weekly_values.csv
--
-- One row per (season, week, player) on the dynasty superflex board of
-- :model_version (default baseline-hl28-v1), newest revision. `week` w is
-- the Tuesday before week w's games, which is the Tuesday after week w-1's
-- Monday game: the same convention as KTC's `weekly_ktc` in the training
-- zip. Week 1 is the Tuesday before kickoff; the last row of a season is the
-- Tuesday after its final regular-season week (week 18 for 16-game seasons,
-- week 19 from 2021). `value` is the stored reference-scale value.
\if :{?model_version}
\else
  \set model_version baseline-hl28-v1
\endif
WITH seasons(season, week1, weeks) AS (
  VALUES
    (2019, DATE '2019-09-03', 17),
    (2020, DATE '2020-09-08', 17),
    (2021, DATE '2021-09-07', 18),
    (2022, DATE '2022-09-06', 18),
    (2023, DATE '2023-09-05', 18),
    (2024, DATE '2024-09-03', 18),
    (2025, DATE '2025-09-02', 18)
),
tuesdays AS (
  SELECT s.season, w AS week, s.week1 + 7 * (w - 1) AS valuation_date
  FROM seasons s, generate_series(1, 19) w
  WHERE w <= s.weeks + 1
),
ranked AS (
  SELECT g.id, g.season, g.valuation_date, g.revision,
         rank() OVER (PARTITION BY g.season, g.valuation_date ORDER BY g.revision DESC) AS r
  FROM market_history_generations g
  WHERE g.model_version = :'model_version'
)
SELECT t.season, t.week, t.valuation_date, v.asset AS player_id, v.value,
       v.trades_window, v.trades_recent, v.effective_evidence, v.distinct_leagues,
       round(v.uncertainty::numeric, 5) AS uncertainty, g.revision
FROM tuesdays t
JOIN ranked g ON g.season = t.season::text AND g.valuation_date = t.valuation_date AND g.r = 1
JOIN market_history_values v
  ON v.generation_id = g.id AND v.format = 'dynasty' AND v.board = 'superflex'
ORDER BY t.season, t.week, v.value DESC, v.asset;
