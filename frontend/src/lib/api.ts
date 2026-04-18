import type {
  Player,
  PlayerList,
  EOSPrediction,
  EOSPredictRequest,
  EOSBatchRequest,
  EOSBatchResponse,
  HistoricalResponse,
} from '../types/player';

const API_BASE = '/api';
const DEFAULT_TIMEOUT_MS = 30_000;

function fetchWithTimeout(
  url: string,
  options?: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  // If the caller already provided a signal (e.g. for cancellation on unmount),
  // forward its abort to our controller so both timeout and caller can cancel.
  if (options?.signal) {
    options.signal.addEventListener('abort', () => controller.abort());
  }
  const id = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id));
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetchWithTimeout(`${API_BASE}${endpoint}`, {
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

function modelParam(modelId?: string | null): string {
  return modelId ? `model=${encodeURIComponent(modelId)}` : '';
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

export async function getPrediction(
  playerId: string,
  modelId?: string | null
): Promise<EOSPrediction | null> {
  const mp = modelParam(modelId);
  const qs = mp ? `?${mp}` : '';
  const response = await fetchWithTimeout(`${API_BASE}/players/${playerId}/predict${qs}`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) return null;
  return response.json();
}

export async function predictEos(
  payload: EOSPredictRequest,
  modelId?: string | null
): Promise<EOSPrediction | null> {
  const mp = modelParam(modelId);
  const qs = mp ? `?${mp}` : '';
  const response = await fetchWithTimeout(`${API_BASE}/predict/eos${qs}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) return null;
  return response.json();
}

export async function predictEosBatch(
  payload: EOSBatchRequest,
  modelId?: string | null,
): Promise<EOSBatchResponse | null> {
  const mp = modelParam(modelId);
  const qs = mp ? `?${mp}` : '';
  const response = await fetchWithTimeout(`${API_BASE}/predict/eos/batch${qs}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) return null;
  return response.json();
}

export interface TopMover {
  player_id: string;
  name: string;
  position: string;
  start_ktc: number;
  predicted_end_ktc: number;
  predicted_delta_ktc: number;
  predicted_pct_change: number;
}

export async function getTopMovers(limit: number = 10): Promise<{ risers: TopMover[]; fallers: TopMover[] } | null> {
  try {
    return await fetchApi(`/top-movers?limit=${limit}`);
  } catch {
    return null;
  }
}

export async function getHistorical(playerId: string): Promise<HistoricalResponse | null> {
  const response = await fetchWithTimeout(`${API_BASE}/players/${playerId}/historical`, {
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) return null;
  return response.json();
}
