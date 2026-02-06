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

export interface EOSPrediction {
  player_id?: string;
  name?: string;
  position: string;
  start_ktc: number;
  predicted_end_ktc: number;
  predicted_delta_ktc: number;
  predicted_pct_change: number;
  model_version: string;
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

export interface PlayerComparison {
  player_id: string;
  name: string;
  position: string;
  start_ktc: number;
  predicted_end_ktc: number;
  predicted_delta_ktc: number;
  predicted_pct_change: number;
  model_version: string;
  seasons: PlayerSeason[];
}

export interface CompareResponse {
  players: PlayerComparison[];
}
