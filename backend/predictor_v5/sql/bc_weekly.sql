-- Weekly Backchannel v1 values for the short-window tool, from the database
-- that holds the full trade corpus (thelab-development), read-only:
--
--   heroku pg:psql -a thelab-development -c "\copy (...) to stdout csv header"
--   (or psql "$URL" -v season=2026 -v week1=2026-09-08 -f predictor_v5/sql/bc_weekly.sql --csv -o predictor_v5/data/bc_weekly_2026.csv)
--
-- Week w is the Tuesday :week1 + 7(w-1); newest baseline-hl28-v1 revision on
-- that exact date, dynasty superflex, players only. When
-- data/bc_weekly_<season>.csv exists it replaces window_inputs' Backchannel
-- columns outright.
WITH weeks AS (
  SELECT w AS week, (:'week1'::date + 7 * (w - 1)) AS tuesday FROM generate_series(1, 19) w
), gen AS (
  SELECT DISTINCT ON (g.valuation_date) g.id, g.valuation_date, g.revision
    FROM market_history_generations g JOIN weeks w ON w.tuesday = g.valuation_date
   WHERE g.model_version = 'baseline-hl28-v1' AND g.season = :'season'
   ORDER BY g.valuation_date, g.revision DESC
)
SELECT v.asset AS player_id, w.week, w.tuesday, v.value AS bc_value, v.trades_window AS bc_trades, g.revision
  FROM gen g JOIN weeks w ON w.tuesday = g.valuation_date
  JOIN market_history_values v ON v.generation_id = g.id AND v.format = 'dynasty' AND v.board = 'superflex'
 WHERE v.asset !~ '[|]'
 ORDER BY w.week, v.value DESC;
