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

// ── Simple in-memory cache ───────────────────────────────────────────────

const _cache = new Map<string, { data: unknown; ts: number }>();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const CACHE_MAX_SIZE = 500;

function getCached<T>(key: string): T | undefined {
  const entry = _cache.get(key);
  if (!entry) return undefined;
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    _cache.delete(key);
    return undefined;
  }
  return entry.data as T;
}

function setCache(key: string, data: unknown): void {
  // Evict oldest entries when cache exceeds max size
  if (_cache.size >= CACHE_MAX_SIZE) {
    const it = _cache.keys();
    for (let i = 0; i < 50; i++) {
      const oldest = it.next();
      if (oldest.done) break;
      _cache.delete(oldest.value);
    }
  }
  _cache.set(key, { data, ts: Date.now() });
}

// ── Fetch helpers ────────────────────────────────────────────────────────

function fetchWithTimeout(
  url: string,
  options?: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
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

// ── API functions ────────────────────────────────────────────────────────

export async function searchPlayers(
  query: string = '',
  position?: string,
  limit: number = 50,
  sortBy: 'name' | 'ktc' | 'predicted' | 'change' | 'ppg' = 'name',
  sortOrder: 'asc' | 'desc' = 'asc'
): Promise<PlayerList> {
  const params = new URLSearchParams();
  if (query) params.set('q', query);
  if (position) params.set('position', position);
  params.set('limit', limit.toString());
  params.set('sort_by', sortBy);
  params.set('sort_order', sortOrder);

  const key = `search:${params}`;
  const cached = getCached<PlayerList>(key);
  if (cached) return cached;

  const result = await fetchApi<PlayerList>(`/players?${params}`);
  setCache(key, result);
  return result;
}

export async function getPlayer(playerId: string): Promise<Player> {
  const key = `player:${playerId}`;
  const cached = getCached<Player>(key);
  if (cached) return cached;

  const result = await fetchApi<Player>(`/players/${playerId}`);
  setCache(key, result);
  return result;
}

export async function getPrediction(
  playerId: string,
  modelId?: string | null
): Promise<EOSPrediction | null> {
  const key = `pred:${playerId}:${modelId ?? 'default'}`;
  const cached = getCached<EOSPrediction | null>(key);
  if (cached !== undefined) return cached;

  const mp = modelParam(modelId);
  const qs = mp ? `?${mp}` : '';
  const response = await fetchWithTimeout(`${API_BASE}/players/${playerId}/predict${qs}`, {
    headers: { 'Content-Type': 'application/json' },
  });

  const result = response.ok ? await response.json() : null;
  setCache(key, result);
  return result;
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

export async function predictPlayerWhatIfBatch(
  playerId: string,
  payload: { games_played: number; ppg_values: number[] },
  modelId?: string | null,
): Promise<EOSBatchResponse | null> {
  // Cache by player + games + ppg_values + model
  const key = `whatif:${playerId}:${payload.games_played}:${JSON.stringify(payload.ppg_values)}:${modelId ?? 'default'}`;
  const cached = getCached<EOSBatchResponse | null>(key);
  if (cached !== undefined) return cached;

  const mp = modelParam(modelId);
  const qs = mp ? `?${mp}` : '';
  const response = await fetchWithTimeout(
    `${API_BASE}/players/${playerId}/predict/whatif-batch${qs}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  );
  const result = response.ok ? await response.json() : null;
  setCache(key, result);
  return result;
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
  const key = `movers:${limit}`;
  const cached = getCached<{ risers: TopMover[]; fallers: TopMover[] }>(key);
  if (cached) return cached;

  try {
    const result = await fetchApi<{ risers: TopMover[]; fallers: TopMover[] }>(`/top-movers?limit=${limit}`);
    setCache(key, result);
    return result;
  } catch {
    return null;
  }
}

export interface CompPlayer {
  player_id: string;
  name: string;
  position: string;
  year: number;
  age: number;
  games_played: number;
  ppg: number;
  start_ktc: number;
  end_ktc: number;
  delta_ktc: number;
  pct_change: number;
  similarity: number;
}

export interface CompsResponse {
  player_id: string;
  name: string;
  position: string;
  query: {
    start_ktc: number;
    ppg: number;
    age: number;
    games_played: number;
  };
  comps: CompPlayer[];
}

export async function getComps(playerId: string, k: number = 10, modelId?: string | null): Promise<CompsResponse | null> {
  const key = `comps:${playerId}:${k}:${modelId ?? 'default'}`;
  const cached = getCached<CompsResponse | null>(key);
  if (cached !== undefined) return cached;

  const params = new URLSearchParams({ k: k.toString() });
  if (modelId) params.set('model', modelId);
  const response = await fetchWithTimeout(`${API_BASE}/players/${playerId}/comps?${params}`, {
    headers: { 'Content-Type': 'application/json' },
  });
  const result = response.ok ? await response.json() : null;
  setCache(key, result);
  return result;
}

export async function getHistorical(playerId: string): Promise<HistoricalResponse | null> {
  const key = `hist:${playerId}`;
  const cached = getCached<HistoricalResponse | null>(key);
  if (cached !== undefined) return cached;

  const response = await fetchWithTimeout(`${API_BASE}/players/${playerId}/historical`, {
    headers: { 'Content-Type': 'application/json' },
  });
  const result = response.ok ? await response.json() : null;
  setCache(key, result);
  return result;
}
