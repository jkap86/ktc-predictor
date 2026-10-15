-- Start-of-season values for predictor_v5's season tool, read from the
-- thelab database (Backchannel market history + the KTC scrape).
--
--   psql "$DATABASE_URL" -v season=2026 -v cutoff=2026-09-09 \
--     -f predictor_v5/sql/start_values.sql --csv -o start_values_2026.csv
--
-- One row per Sleeper player (QB/RB/WR/TE) with a value on either source.
-- Backchannel: the dynasty superflex board of the newest generation dated on
-- or before :cutoff (stored `value` is the fixed reference scale, the same
-- scale as the training labels). KTC: each player's latest dynasty SF value
-- on or before :cutoff, within 14 days of it.
WITH gen AS (
    SELECT id, valuation_date, model_version, revision
      FROM market_history_generations
     WHERE season = :'season' AND valuation_date <= :'cutoff'::date
     ORDER BY valuation_date DESC, generated_at DESC, revision DESC
     LIMIT 1
), bc AS (
    SELECT v.asset AS player_id, v.value, v.trades_window, g.valuation_date, g.model_version
      FROM gen g
      JOIN market_history_values v ON v.generation_id = g.id
     WHERE v.format = 'dynasty' AND v.board = 'superflex' AND v.asset !~ '[|]'
), ktc AS (
    SELECT DISTINCT ON (k.sleeper_id) k.sleeper_id AS player_id, h.sf_value AS value, h.date
      FROM ktc_value_history h
      JOIN ktc_values k USING (format, ktc_id)
     WHERE h.format = 'dynasty' AND k.sleeper_id IS NOT NULL AND h.sf_value > 0
       AND h.date BETWEEN :'cutoff'::date - 14 AND :'cutoff'::date
     ORDER BY k.sleeper_id, h.date DESC
)
SELECT p.player_id,
       p.full_name AS name,
       p.position,
       p.team,
       CASE WHEN p.data->>'birth_date' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
            THEN left(p.data->>'birth_date', 10) END AS birth_date,
       bc.value AS bc_value,
       bc.trades_window AS bc_trades,
       bc.valuation_date AS bc_date,
       bc.model_version AS bc_model_version,
       ktc.value AS ktc_value,
       ktc.date AS ktc_date
  FROM players p
  LEFT JOIN bc ON bc.player_id = p.player_id
  LEFT JOIN ktc ON ktc.player_id = p.player_id
 WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
   AND (bc.value > 0 OR ktc.value > 0)
 ORDER BY coalesce(bc.value, 0) DESC, coalesce(ktc.value, 0) DESC;
