WITH cutoffs(season, start_date, end_date) AS (
  VALUES
    (2019, DATE '2019-09-05', DATE '2019-12-29'),
    (2020, DATE '2020-09-10', DATE '2021-01-03'),
    (2021, DATE '2021-09-09', DATE '2022-01-09'),
    (2022, DATE '2022-09-08', DATE '2023-01-08'),
    (2023, DATE '2023-09-07', DATE '2024-01-07'),
    (2024, DATE '2024-09-05', DATE '2025-01-05'),
    (2025, DATE '2025-09-04', DATE '2026-01-04')
),
latest AS (
  SELECT DISTINCT ON (season, valuation_date)
    id, season, valuation_date
  FROM market_history_generations
  WHERE model_version = 'baseline-hl28-v1'
  ORDER BY season, valuation_date, revision DESC
),
boards AS (
  SELECT
    c.season,
    start_values.asset AS player_id,
    start_values.value AS start_value,
    end_values.value AS end_value,
    start_values.effective_evidence AS start_evidence,
    end_values.effective_evidence AS end_evidence,
    start_values.trades_window AS start_trades_window,
    end_values.trades_window AS end_trades_window,
    next_values.value AS end_value_next_board,
    next_values.effective_evidence AS end_evidence_next_board
  FROM cutoffs c
  JOIN latest start_generation
    ON start_generation.season = c.season::text
   AND start_generation.valuation_date = c.start_date
  JOIN market_history_values start_values
    ON start_values.generation_id = start_generation.id
   AND start_values.format = 'dynasty'
   AND start_values.board = 'superflex'
  JOIN latest end_generation
    ON end_generation.season = c.season::text
   AND end_generation.valuation_date = c.end_date
  JOIN market_history_values end_values
    ON end_values.generation_id = end_generation.id
   AND end_values.format = 'dynasty'
   AND end_values.board = 'superflex'
   AND end_values.asset = start_values.asset
  LEFT JOIN latest next_generation
    ON next_generation.season = (c.season + 1)::text
   AND next_generation.valuation_date = c.end_date
  LEFT JOIN market_history_values next_values
    ON next_values.generation_id = next_generation.id
   AND next_values.format = 'dynasty'
   AND next_values.board = 'superflex'
   AND next_values.asset = start_values.asset
)
SELECT
  player_id,
  season,
  start_value,
  end_value,
  start_evidence,
  end_evidence,
  start_trades_window,
  end_trades_window,
  end_value_next_board,
  end_evidence_next_board
FROM boards
ORDER BY season, start_value DESC, player_id;
