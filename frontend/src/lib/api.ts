import type {
  Player,
  PlayerList,
  EOSPrediction,
  EOSPredictRequest,
  CompareResponse,
} from '../types/player';

const API_BASE = '/api';

// Simple client-side response cache for slider interactions
const eosCache = new Map<string, { data: EOSPrediction; ts: number }>();
const CACHE_TTL = 60_000; // 1 minute

function eosKey(payload: EOSPredictRequest): string {
  return JSON.stringify(payload);
}

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

export async function getPrediction(
  playerId: string,
  blendWeekly: boolean = false
): Promise<EOSPrediction | null> {
  const params = blendWeekly ? '?blend_weekly=true' : '';
  const response = await fetch(`${API_BASE}/players/${playerId}/predict${params}`, {
    headers: { 'Content-Type': 'application/json' },
  });

  if (!response.ok) return null;
  return response.json();
}

export async function predictEos(payload: EOSPredictRequest): Promise<EOSPrediction | null> {
  // Check cache first
  const key = eosKey(payload);
  const cached = eosCache.get(key);
  if (cached && Date.now() - cached.ts < CACHE_TTL) {
    return cached.data;
  }

  const response = await fetch(`${API_BASE}/predict/eos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) return null;
  const data: EOSPrediction = await response.json();

  // Store in cache
  eosCache.set(key, { data, ts: Date.now() });

  // Evict old entries if cache grows too large
  if (eosCache.size > 500) {
    const cutoff = Date.now() - CACHE_TTL;
    eosCache.forEach((v, k) => {
      if (v.ts < cutoff) eosCache.delete(k);
    });
  }

  return data;
}

export async function comparePlayers(
  playerIds: string[],
  blendWeekly: boolean = false
): Promise<CompareResponse> {
  return fetchApi<CompareResponse>('/compare', {
    method: 'POST',
    body: JSON.stringify({ player_ids: playerIds, blend_weekly: blendWeekly }),
  });
}

export async function getPositions(): Promise<{ positions: string[] }> {
  return fetchApi<{ positions: string[] }>('/players/positions');
}

// Live KTC from database (cached hourly on backend)
export interface LiveKTC {
  player_id: string;
  ktc: number;
  date: string;
  overall_rank: number | null;
  position_rank: number | null;
}

export async function getLiveKtc(playerId: string): Promise<LiveKTC | null> {
  try {
    return await fetchApi<LiveKTC>(`/ktc/${playerId}`);
  } catch {
    return null;
  }
}

export async function getBatchLiveKtc(playerIds: string[]): Promise<LiveKTC[]> {
  if (playerIds.length === 0) return [];
  return fetchApi<LiveKTC[]>('/ktc/batch', {
    method: 'POST',
    body: JSON.stringify(playerIds),
  });
}
