import type {
  Player,
  PlayerList,
  EOSPrediction,
  EOSPredictRequest,
  CompareResponse,
} from '../types/player';

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

export async function getPrediction(playerId: string): Promise<EOSPrediction | null> {
  const response = await fetch(`${API_BASE}/players/${playerId}/predict`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) return null;
  return response.json();
}

export async function predictEos(payload: EOSPredictRequest): Promise<EOSPrediction | null> {
  const response = await fetch(`${API_BASE}/predict/eos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) return null;
  return response.json();
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
