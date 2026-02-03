export interface WeeklyStat {
  week: number;
  fantasy_points: number;
  games_played: number;
  snap_pct: number;
}

export interface WeeklyKTC {
  week: number;
  ktc: number;
  date: string;
}

export interface PlayerSeason {
  year: number;
  age: number;
  years_exp: number;
  start_ktc: number;
  end_ktc: number;
  fantasy_points: number;
  games_played: number;
  ktc_30d_trend: number | null;
  ktc_90d_trend: number | null;
  ktc_volatility: number;
  prior_year_fp: number;
  prior_year_games: number;
  fp_change_yoy: number;
  start_position_rank: number;
  weekly_stats: WeeklyStat[];
  weekly_ktc: WeeklyKTC[];
}

export interface Player {
  player_id: string;
  name: string;
  position: string;
  seasons: PlayerSeason[];
}

export interface PlayerSummary {
  player_id: string;
  name: string;
  position: string;
  latest_ktc: number | null;
  latest_year: number | null;
}

export interface PlayerList {
  players: PlayerSummary[];
  total: number;
}

export interface PredictionResponse {
  player_id: string;
  name: string;
  position: string;
  current_ktc: number;
  predicted_ktc: number;
  ktc_change: number;
  ktc_change_pct: number;
  confidence?: number;
}

export interface PlayerComparison {
  player_id: string;
  name: string;
  position: string;
  current_ktc: number;
  predicted_ktc: number;
  ktc_change: number;
  seasons: PlayerSeason[];
}

export interface CompareResponse {
  players: PlayerComparison[];
}

export interface PredictionWithPPG {
  player_id: string;
  name: string;
  position: string;
  ppg: number;
  predicted_ktc: number;
  current_ktc: number;
  ktc_change_pct: number;
}

// ========== Weekly Simulation Types ==========

export interface CurvePoint {
  ppg: number;
  predicted_ktc: number;
}

export interface SimulateCurveResponse {
  player_id: string;
  name: string;
  position: string;
  starting_ktc: number;
  current_ppg: number;
  games: number;
  curve: CurvePoint[];
}

export interface WeeklyProjection {
  week: number;
  ktc: number;
  fp: number;
  change: number;
}

export interface SimulationResponse {
  player_id: string;
  name: string;
  position: string;
  starting_ktc: number;
  final_ktc: number;
  total_change: number;
  total_change_pct: number;
  games: number;
  ppg: number;
  trajectory: WeeklyProjection[];
}

// ========== Model Metrics Types ==========

export interface YearMetrics {
  year: number;
  r2: number;
  mae: number;
  n_samples: number;
}

export interface OverallMetrics {
  test_r2: number;
  test_mae: number;
  train_r2: number;
  train_mae: number;
  n_train: number;
  n_test: number;
}

export interface MetricsByYearResponse {
  overall: OverallMetrics;
  by_year: YearMetrics[];
}
