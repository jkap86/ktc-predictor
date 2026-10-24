-- Start/end-of-season Backchannel values for predictor_v5's label CSVs, read
-- from the thelab database.
--
--   psql "$DATABASE_URL" [-v model_version=baseline-hl28-v3] \
--     -f predictor_v5/sql/backchannel_labels.sql --csv -o backchannel_raw.csv
--   python -m predictor_v5 build-labels --values backchannel_raw.csv \
--     --out labels.csv [--model-version baseline-hl28-v3]
--
-- :model_version defaults to baseline-hl28-v1, the model the committed
-- labels and the season tool use (thelab's own default moved to v3). Each
-- cutoff reads the highest revision of that model's generation for the
-- season and date. Every season needs a start and an end generation with a
-- dynasty superflex board, and the newest revision must be unique; otherwise
-- the query stops with an error naming the season. Plain SQL can't RAISE, so
-- the guard casts its message to integer, and the error reads
-- `invalid input syntax for type integer: "backchannel_labels: ..."`.
-- The next season's board on the end date is optional (v3 has no 2026 board
-- on 2026-01-04), so end_value_next_board may be empty.
--
-- Columns after end_evidence_next_board record provenance; build-labels
-- checks them and writes them to the label file's .provenance.json.
\if :{?model_version}
\else
  \set model_version baseline-hl28-v1
\endif
WITH cutoffs(season, start_date, end_date) AS (
  VALUES
    (2019, DATE '2019-09-05', DATE '2019-12-29'),
    (2020, DATE '2020-09-10', DATE '2021-01-03'),
    (2021, DATE '2021-09-09', DATE '2022-01-09'),
    (2022, DATE '2022-09-08', DATE '2023-01-08'),
    (2023, DATE '2023-09-07', DATE '2024-01-07'),
    (2024, DATE '2024-09-05', DATE '2025-01-05'),
    (2025, DATE '2025-09-04', DATE '2026-01-04'),
    (2026, DATE '2026-09-09', DATE '2027-01-10')
),
params AS (
  SELECT CAST(:'model_version' AS text) AS model_version
),
ranked AS (
  SELECT g.id, g.season, g.valuation_date, g.model_version, g.revision,
         rank() OVER (PARTITION BY g.season, g.valuation_date ORDER BY g.revision DESC) AS revision_rank
  FROM market_history_generations g
  JOIN params p ON g.model_version = p.model_version
),
latest AS (
  SELECT id, season, valuation_date, model_version, revision,
         count(*) OVER (PARTITION BY season, valuation_date) AS ties
  FROM ranked
  WHERE revision_rank = 1
),
board_sizes AS (
  SELECT generation_id, count(*) AS board_rows
  FROM market_history_values
  WHERE format = 'dynasty' AND board = 'superflex'
    AND generation_id IN (SELECT id FROM latest)
  GROUP BY generation_id
),
selected AS (
  SELECT
    c.season, c.start_date, c.end_date,
    s.id AS start_generation_id, s.revision AS start_revision, s.ties AS start_ties,
    coalesce(sb.board_rows, 0) AS start_board_rows,
    e.id AS end_generation_id, e.revision AS end_revision, e.ties AS end_ties,
    coalesce(eb.board_rows, 0) AS end_board_rows,
    n.id AS next_generation_id, n.revision AS next_revision, coalesce(n.ties, 1) AS next_ties
  FROM cutoffs c
  LEFT JOIN latest s ON s.season = c.season::text AND s.valuation_date = c.start_date
  LEFT JOIN board_sizes sb ON sb.generation_id = s.id
  LEFT JOIN latest e ON e.season = c.season::text AND e.valuation_date = c.end_date
  LEFT JOIN board_sizes eb ON eb.generation_id = e.id
  LEFT JOIN latest n ON n.season = (c.season + 1)::text AND n.valuation_date = c.end_date
),
guard AS MATERIALIZED (
  SELECT count(*) AS checked
  FROM selected r CROSS JOIN params p
  WHERE CASE
    WHEN r.start_generation_id IS NULL THEN CAST(
      'backchannel_labels: no ' || p.model_version || ' generation for season ' || r.season
      || ' on its start cutoff ' || r.start_date AS integer)
    WHEN r.end_generation_id IS NULL THEN CAST(
      'backchannel_labels: no ' || p.model_version || ' generation for season ' || r.season
      || ' on its end cutoff ' || r.end_date AS integer)
    WHEN r.start_ties > 1 OR r.end_ties > 1 OR r.next_ties > 1 THEN CAST(
      'backchannel_labels: several ' || p.model_version || ' generations share the newest revision for season '
      || r.season AS integer)
    WHEN r.start_board_rows = 0 THEN CAST(
      'backchannel_labels: start generation ' || r.start_generation_id || ' for season ' || r.season
      || ' has no dynasty superflex values' AS integer)
    WHEN r.end_board_rows = 0 THEN CAST(
      'backchannel_labels: end generation ' || r.end_generation_id || ' for season ' || r.season
      || ' has no dynasty superflex values' AS integer)
    ELSE 0
  END = 0
),
boards AS (
  SELECT
    r.season,
    start_values.asset AS player_id,
    start_values.value AS start_value,
    end_values.value AS end_value,
    start_values.effective_evidence AS start_evidence,
    end_values.effective_evidence AS end_evidence,
    start_values.trades_window AS start_trades_window,
    end_values.trades_window AS end_trades_window,
    next_values.value AS end_value_next_board,
    next_values.effective_evidence AS end_evidence_next_board,
    r.start_generation_id, r.start_revision, r.end_generation_id, r.end_revision,
    r.next_generation_id, r.next_revision, r.start_board_rows, r.end_board_rows
  FROM selected r
  JOIN market_history_values start_values
    ON start_values.generation_id = r.start_generation_id
   AND start_values.format = 'dynasty'
   AND start_values.board = 'superflex'
  JOIN market_history_values end_values
    ON end_values.generation_id = r.end_generation_id
   AND end_values.format = 'dynasty'
   AND end_values.board = 'superflex'
   AND end_values.asset = start_values.asset
  LEFT JOIN market_history_values next_values
    ON next_values.generation_id = r.next_generation_id
   AND next_values.format = 'dynasty'
   AND next_values.board = 'superflex'
   AND next_values.asset = start_values.asset
)
SELECT
  b.player_id,
  b.season,
  b.start_value,
  b.end_value,
  b.start_evidence,
  b.end_evidence,
  b.start_trades_window,
  b.end_trades_window,
  b.end_value_next_board,
  b.end_evidence_next_board,
  p.model_version,
  b.start_generation_id,
  b.start_revision,
  b.end_generation_id,
  b.end_revision,
  b.next_generation_id,
  b.next_revision,
  b.start_board_rows,
  b.end_board_rows,
  to_char(statement_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS exported_at
FROM boards b
CROSS JOIN params p
-- A constant condition: Postgres checks it once, before reading any board,
-- so the guard runs even when a missing generation leaves no rows.
WHERE (SELECT checked FROM guard) = (SELECT count(*) FROM cutoffs)
ORDER BY b.season, b.start_value DESC, b.player_id;
