import type {
  Player,
  PlayerList,
  PredictionResponse,
  CompareResponse,
  PredictionWithPPG,
  SimulateCurveResponse,
  MetricsByYearResponse,
} from '@/types/player';

const API_BASE = '/api';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function searchPlayers(
  query: string = '',
  position?: string,
  limit: number = 50,
  sortBy: 'name' | 'ktc' = 'name',
  sortOrder: 'asc' | 'desc' = 'asc'
): Promise<PlayerList> {
  const params = new URLSearchParams();
  if (query) params.set('q', query);
  if (position) params.set('position', position);
  params.set('limit', limit.toString());
  params.set('sort_by', sortBy);
  params.set('sort_order', sortOrder);

  return fetchApi<PlayerList>(`/players?${params}`);
}

export async function getPlayer(playerId: string): Promise<Player> {
  return fetchApi<Player>(`/players/${playerId}`);
}

export async function getPrediction(playerId: string): Promise<PredictionResponse> {
  return fetchApi<PredictionResponse>(`/players/${playerId}/predict`);
}

export async function comparePlayers(playerIds: string[]): Promise<CompareResponse> {
  return fetchApi<CompareResponse>('/compare', {
    method: 'POST',
    body: JSON.stringify({ player_ids: playerIds }),
  });
}

export async function getPositions(): Promise<{ positions: string[] }> {
  return fetchApi<{ positions: string[] }>('/players/positions');
}

export async function getAllPredictions(
  position?: string
): Promise<PredictionWithPPG[]> {
  const params = position ? `?position=${position}` : '';
  return fetchApi<PredictionWithPPG[]>(`/predictions/all${params}`);
}

export async function simulateCurve(
  playerId: string,
  games: number
): Promise<SimulateCurveResponse> {
  return fetchApi<SimulateCurveResponse>(`/players/${playerId}/simulate-curve`, {
    method: 'POST',
    body: JSON.stringify({ games }),
  });
}

export async function getModelMetricsByYear(): Promise<MetricsByYearResponse> {
  return fetchApi<MetricsByYearResponse>('/model/metrics-by-year');
}
