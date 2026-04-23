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
  ppg?: number;
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
  live_ktc?: number;
}

export interface PlayerSummary {
  player_id: string;
  name: string;
  position: string;
  latest_ktc: number | null;
  ppg: number | null;
  predicted_end_ktc: number | null;
}

export interface PlayerList {
  players: PlayerSummary[];
  total: number;
}

export interface EOSPrediction {
  player_id?: string;
  name?: string;
  position: string;
  start_ktc: number;
  predicted_end_ktc: number;
  predicted_delta_ktc: number;
  predicted_pct_change: number;
  low_end_ktc?: number;
  high_end_ktc?: number;
  model_version: string;
  anchor_year?: number;
  anchor_source?: string;
  baseline_year?: number;
}

export interface EOSPredictRequest {
  position: string;
  start_ktc: number;
  games_played: number;
  ppg: number;
  age?: number;
  weeks_missed?: number;
  draft_pick?: number;
  years_remaining?: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  created: string;
  features_count?: number | Record<string, number>;
  positions: string[];
  is_default: boolean;
  metrics: Record<string, unknown>;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  default_model: string;
}

export interface EOSBatchRequest {
  position: string;
  start_ktc: number;
  games_played: number;
  ppg_values: number[];
  age?: number;
  weeks_missed?: number;
  draft_pick?: number;
  years_remaining?: number;
}

export interface EOSBatchResponse {
  predictions: EOSPrediction[];
  model_version: string;
}

export interface HistoricalPrediction {
  year: number;
  model_version: string;
  start_ktc: number;
  actual_end_ktc: number | null;
  predicted_end_ktc: number;
  predicted_delta_ktc: number;
  predicted_pct_change: number;
  low_end_ktc: number | null;
  high_end_ktc: number | null;
  error: number | null;
  games_played: number;
  ppg: number;
}

export interface HistoricalResponse {
  player_id: string;
  name: string;
  position: string;
  predictions: HistoricalPrediction[];
}
