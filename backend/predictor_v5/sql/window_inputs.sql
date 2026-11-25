-- Live inputs for the short-window tool (window_tool.py), one row per
-- player and week of :season, read from the thelab database.
--
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -v season=2026 -v week1=2026-09-08 \
--     -f predictor_v5/sql/window_inputs.sql --csv -o predictor_v5/data/window_inputs_2026.csv
--
-- Week w is the Tuesday :week1 + 7(w-1) (the Tuesday before week w's games),
-- the convention the short-window models were trained on. Weeks 1-19.
-- KTC: the latest dynasty superflex value on or before that Tuesday, within
-- 7 days (ktc_date says which day). Backchannel: the dynasty superflex value
-- of the newest baseline-hl28-v1 generation on exactly that Tuesday, with its
-- 150-day trade count (empty once v1 history stops). Points: Sleeper full-PPR
-- points and games played for week w, newest observation.
WITH weeks AS (
  SELECT w AS week, (:'week1'::date + 7 * (w - 1)) AS tuesday FROM generate_series(1, 19) w
), roster AS (
  SELECT p.player_id, p.full_name AS name, p.position, p.team,
         CASE WHEN p.data->>'birth_date' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN left(p.data->>'birth_date', 10) END AS birth_date,
         k.ktc_id
    FROM players p
    JOIN ktc_values k ON k.sleeper_id = p.player_id AND k.format = 'dynasty'
   WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
), ktc AS (
  SELECT r.player_id, w.week, h.sf_value AS ktc_value, h.date AS ktc_date
    FROM roster r CROSS JOIN weeks w
    CROSS JOIN LATERAL (
      SELECT sf_value, date FROM ktc_value_history h
       WHERE h.format = 'dynasty' AND h.ktc_id = r.ktc_id AND h.sf_value > 0
         AND h.date BETWEEN w.tuesday - 7 AND w.tuesday
       ORDER BY h.date DESC LIMIT 1
    ) h
), gen AS (
  SELECT DISTINCT ON (g.valuation_date) g.id, g.valuation_date
    FROM market_history_generations g JOIN weeks w ON w.tuesday = g.valuation_date
   WHERE g.model_version = 'baseline-hl28-v1' AND g.season = :'season'
   ORDER BY g.valuation_date, g.revision DESC
), bc AS (
  SELECT v.asset AS player_id, w.week, v.value AS bc_value, v.trades_window AS bc_trades
    FROM gen g JOIN weeks w ON w.tuesday = g.valuation_date
    JOIN market_history_values v ON v.generation_id = g.id AND v.format = 'dynasty' AND v.board = 'superflex'
), pts AS (
  SELECT DISTINCT ON (player_id, week) player_id, week,
         (components->>'pts_ppr')::float AS points, coalesce((components->>'gp')::float, 0) AS played
    FROM market_stat_observations
   WHERE source = 'sleeper' AND season = :'season'
   ORDER BY player_id, week, observed_at DESC
)
SELECT r.player_id, r.name, r.position, r.team, r.birth_date, w.week, w.tuesday,
       ktc.ktc_value, ktc.ktc_date, bc.bc_value, bc.bc_trades, pts.points, pts.played
  FROM roster r CROSS JOIN weeks w
  LEFT JOIN ktc ON ktc.player_id = r.player_id AND ktc.week = w.week
  LEFT JOIN bc ON bc.player_id = r.player_id AND bc.week = w.week
  LEFT JOIN pts ON pts.player_id = r.player_id AND pts.week = w.week
 WHERE EXISTS (SELECT 1 FROM ktc k2 WHERE k2.player_id = r.player_id)
 ORDER BY r.player_id, w.week;
