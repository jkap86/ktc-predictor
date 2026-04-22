'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { searchPlayers, getPlayer, getPrediction, predictPlayerWhatIfBatch } from '../lib/api';
import { formatKtc } from '../lib/format';
import { useModel } from '../context/ModelContext';
import WhatIfChart from '../components/WhatIfChart';
import PlayerComps from '../components/PlayerComps';
import HistoricalAccuracy from '../components/HistoricalAccuracy';
import TopMovers from '../components/TopMovers';
import type { Player, PlayerSummary, EOSPrediction } from '../types/player';

const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];

// ── Sub-components ───────────────────────────────────────────────────────

function ConfidenceBand({ prediction, color = 'blue' }: { prediction: EOSPrediction; color?: 'blue' | 'orange' }) {
  if (!prediction.low_end_ktc || !prediction.high_end_ktc) return null;
  const isOrange = color === 'orange';
  return (
    <div className={`mt-2 px-3 py-2 rounded-lg border ${isOrange ? 'bg-orange-50 dark:bg-orange-900/20 border-orange-100 dark:border-orange-800' : 'bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800'}`}>
      <div className="flex items-center gap-2 text-xs">
        <span className="text-gray-500 dark:text-gray-400">{formatKtc(prediction.low_end_ktc)}</span>
        <div className={`flex-1 h-1.5 rounded-full relative ${isOrange ? 'bg-orange-100 dark:bg-orange-800' : 'bg-blue-100 dark:bg-blue-800'}`}>
          <div
            className={`absolute h-1.5 rounded-full ${isOrange ? 'bg-orange-500' : 'bg-blue-500'}`}
            style={{
              left: `${Math.max(0, Math.min(100, ((prediction.predicted_end_ktc - prediction.low_end_ktc) / (prediction.high_end_ktc - prediction.low_end_ktc)) * 100))}%`,
              width: '3px', transform: 'translateX(-50%)',
            }}
          />
        </div>
        <span className="text-gray-500 dark:text-gray-400">{formatKtc(prediction.high_end_ktc)}</span>
      </div>
    </div>
  );
}

function PredictionStats({ prediction, label, color = 'blue' }: {
  prediction: EOSPrediction; label: string; color?: 'blue' | 'orange';
}) {
  const pct = prediction.predicted_pct_change;
  const delta = prediction.predicted_delta_ktc;
  const changeColor = pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
  const bg = color === 'orange'
    ? 'bg-orange-50/50 dark:bg-orange-950/20 border border-orange-100 dark:border-orange-900/30'
    : 'bg-blue-50/50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/30';
  const labelColor = color === 'orange' ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400';
  return (
    <div className={`${bg} rounded-lg p-3`}>
      <div className={`text-xs font-medium ${labelColor} mb-1`}>{label}</div>
      <div className="flex items-baseline gap-3">
        <span className="text-xl font-bold text-gray-900 dark:text-white">
          {formatKtc(prediction.predicted_end_ktc)}
        </span>
        <span className={`text-sm font-bold ${changeColor}`}>
          {delta >= 0 ? '+' : ''}{delta.toLocaleString()}
        </span>
        <span className={`text-sm font-bold ${changeColor}`}>
          ({pct >= 0 ? '+' : ''}{pct.toFixed(1)}%)
        </span>
      </div>
      <ConfidenceBand prediction={prediction} color={color} />
    </div>
  );
}

function SeasonRow({ season }: {
  season: { year: number; fantasy_points: number; games_played: number; age: number; start_position_rank: number };
}) {
  const ppg = season.games_played > 0 ? (season.fantasy_points / season.games_played).toFixed(1) : '0';
  return (
    <div className="flex items-center gap-4 text-sm">
      <span className="text-gray-500 dark:text-gray-400 w-10">{season.year}</span>
      <span className="text-gray-900 dark:text-white font-medium w-14">{ppg} ppg</span>
      <span className="text-gray-600 dark:text-gray-400">{season.games_played}gp</span>
      <span className="text-gray-600 dark:text-gray-400">age {season.age}</span>
    </div>
  );
}

function PlayerHeader({ player, prediction, color = 'blue', onRemove, onSwap }: {
  player: Player; prediction: EOSPrediction | null; color?: 'blue' | 'orange'; onRemove?: () => void; onSwap?: () => void;
}) {
  const accent = color === 'orange' ? 'text-orange-500 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400';
  const border = color === 'orange' ? 'border-orange-200 dark:border-orange-800' : 'border-blue-200 dark:border-blue-800';
  const bg = color === 'orange' ? 'bg-orange-50 dark:bg-orange-950/30' : 'bg-blue-50 dark:bg-blue-950/30';
  return (
    <div className={`${bg} rounded-xl shadow-sm border ${border} p-4 flex items-center justify-between gap-2`}>
      <div className="min-w-0">
        <h2 className={`text-lg font-bold truncate ${accent}`}>{player.name}</h2>
        <span className="inline-block px-2 py-0.5 bg-white/60 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded text-xs mt-0.5">
          {player.position}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {onSwap && (
          <button onClick={onSwap} className={`p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors ${accent}`} title="Swap players">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
            </svg>
          </button>
        )}
        {onRemove && (
          <button onClick={onRemove} className={`p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors ${accent}`} title="Remove">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
        <div className="text-right shrink-0">
          <div className={`text-2xl font-bold ${accent}`}>
            {formatKtc(player.live_ktc ?? prediction?.start_ktc ?? 0)}
          </div>
          <div className="text-xs text-gray-400">{player.live_ktc ? 'Live' : 'KTC'}</div>
        </div>
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────

export default function Home() {
  const { selectedModelId } = useModel();

  // Search state
  const [query, setQuery] = useState('');
  const [position, setPosition] = useState('All');
  const [searchResults, setSearchResults] = useState<PlayerSummary[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  // Selected players
  const [primaryId, setPrimaryId] = useState<string | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);

  // Player data
  const [primary, setPrimary] = useState<{ player: Player; prediction: EOSPrediction | null } | null>(null);
  const [compare, setCompare] = useState<{ player: Player; prediction: EOSPrediction | null } | null>(null);
  const [primaryLoading, setPrimaryLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);

  // What-If
  const [whatIfPpg, setWhatIfPpg] = useState(15);
  const [whatIfResult, setWhatIfResult] = useState<EOSPrediction | null>(null);
  const [compareWhatIfResult, setCompareWhatIfResult] = useState<EOSPrediction | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Search
  useEffect(() => {
    const fetchPlayers = async () => {
      setSearchLoading(true);
      try {
        const pos = position === 'All' ? undefined : position;
        const result = await searchPlayers(query, pos, 200, 'ktc', 'desc');
        setSearchResults(result.players);
      } catch { /* */ }
      finally { setSearchLoading(false); }
    };
    const debounce = setTimeout(fetchPlayers, 300);
    return () => clearTimeout(debounce);
  }, [query, position]);

  // Fetch primary player data
  useEffect(() => {
    if (!primaryId) { setPrimary(null); setWhatIfResult(null); return; }
    setPrimaryLoading(true);
    (async () => {
      try {
        const player = await getPlayer(primaryId);
        const prediction = await getPrediction(primaryId, selectedModelId);
        setPrimary({ player, prediction });
        if (player.seasons.length > 0) {
          const latest = player.seasons.reduce((a, b) => (a.year > b.year ? a : b));
          setWhatIfPpg(Math.round((latest.games_played > 0 ? latest.fantasy_points / latest.games_played : 15) * 2) / 2);
        }
      } catch { setPrimary(null); }
      finally { setPrimaryLoading(false); }
    })();
  }, [primaryId, selectedModelId]);

  // Fetch compare player data
  useEffect(() => {
    if (!compareId) { setCompare(null); setCompareWhatIfResult(null); return; }
    setCompareLoading(true);
    (async () => {
      try {
        const player = await getPlayer(compareId);
        const prediction = await getPrediction(compareId, selectedModelId);
        setCompare({ player, prediction });
      } catch { setCompare(null); }
      finally { setCompareLoading(false); }
    })();
  }, [compareId, selectedModelId]);

  // What-If for primary
  const fetchWhatIf = useCallback(async () => {
    if (!primaryId || !primary?.prediction) return;
    try {
      const batch = await predictPlayerWhatIfBatch(primaryId, { games_played: 17, ppg_values: [whatIfPpg] }, selectedModelId);
      setWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
  }, [primaryId, primary?.prediction, whatIfPpg, selectedModelId]);

  // What-If for compare
  const fetchCompareWhatIf = useCallback(async () => {
    if (!compareId || !compare?.prediction) return;
    try {
      const batch = await predictPlayerWhatIfBatch(compareId, { games_played: 17, ppg_values: [whatIfPpg] }, selectedModelId);
      setCompareWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
  }, [compareId, compare?.prediction, whatIfPpg, selectedModelId]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => { fetchWhatIf(); fetchCompareWhatIf(); }, 200);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchWhatIf, fetchCompareWhatIf]);

  // Click handler for search result cards
  const handleSelectPlayer = (player: PlayerSummary) => {
    const pid = player.player_id;
    if (pid === primaryId) {
      // Deselect primary — promote compare if it exists
      setPrimaryId(compareId);
      setCompareId(null);
    } else if (pid === compareId) {
      // Deselect compare
      setCompareId(null);
    } else if (!primaryId) {
      setPrimaryId(pid);
    } else if (!compareId) {
      setCompareId(pid);
    } else {
      // Both slots filled — replace compare
      setCompareId(pid);
    }
  };

  const hasCompare = !!compare && !!compareId;
  const hasAnySelected = !!primaryId;

  const primaryPrediction = primary?.prediction ?? null;
  const comparePrediction = compare?.prediction ?? null;
  const primaryPlayer = primary?.player ?? null;
  const comparePlayer = compare?.player ?? null;
  const latestSeason = primaryPlayer?.seasons.length ? primaryPlayer.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null;
  const compareLatest = comparePlayer?.seasons.length ? comparePlayer.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="text-center">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-1">KTC Value Predictor</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">Select up to two players to compare predictions</p>
      </div>

      {/* Search bar + position filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <input
          type="text"
          placeholder="Search players..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-base placeholder-gray-400 dark:placeholder-gray-500 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <div className="flex gap-2">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                position === pos
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {/* Search results as selectable cards */}
      {searchLoading ? (
        <div className="flex justify-center py-8">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="max-h-[520px] overflow-y-auto rounded-xl border border-gray-100 dark:border-gray-700 p-2">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {searchResults.map((player) => {
            const isPrimary = player.player_id === primaryId;
            const isCompare = player.player_id === compareId;
            const isSelected = isPrimary || isCompare;
            return (
              <button
                key={player.player_id}
                onClick={() => handleSelectPlayer(player)}
                className={`text-left p-4 rounded-xl shadow-sm border transition-all duration-200 ${
                  isPrimary
                    ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-700 ring-2 ring-blue-400'
                    : isCompare
                    ? 'bg-orange-50 dark:bg-orange-950/30 border-orange-300 dark:border-orange-700 ring-2 ring-orange-400'
                    : 'bg-white dark:bg-gray-800 border-gray-100 dark:border-gray-700 hover:shadow-md hover:border-gray-200 dark:hover:border-gray-600 hover:-translate-y-0.5'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className={`font-semibold text-lg ${isPrimary ? 'text-blue-700 dark:text-blue-300' : isCompare ? 'text-orange-700 dark:text-orange-300' : 'text-gray-900 dark:text-white'}`}>
                      {player.name}
                    </h3>
                    <span className="inline-block px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded mt-1">
                      {player.position}
                    </span>
                  </div>
                  {player.latest_ktc != null && (
                    <div className="text-right">
                      <div className={`text-xl font-bold ${isPrimary ? 'text-blue-600 dark:text-blue-400' : isCompare ? 'text-orange-500 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'}`}>
                        {formatKtc(player.latest_ktc)}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">KTC</div>
                    </div>
                  )}
                </div>
                {isSelected && (
                  <div className={`mt-2 text-xs font-medium ${isPrimary ? 'text-blue-500' : 'text-orange-500'}`}>
                    {isPrimary ? 'Primary' : 'Compare'}
                  </div>
                )}
              </button>
            );
          })}
        </div>
        </div>
      )}

      {!searchLoading && searchResults.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">No players found.</div>
      )}

      {/* Show TopMovers only when no player is selected */}
      {!hasAnySelected && <TopMovers />}

      {/* ══════════════════════════════════════════════════════════════════
          Player detail sections (shown when at least one player selected)
          ══════════════════════════════════════════════════════════════════ */}
      {hasAnySelected && (
        <div className="space-y-6 pt-2 border-t border-gray-200 dark:border-gray-700">
          {/* Player Headers */}
          {(primaryLoading || compareLoading) && (
            <div className="flex justify-center py-6">
              <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          )}

          {primaryPlayer && (
            <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
              <PlayerHeader player={primaryPlayer} prediction={primaryPrediction} color="blue"
                onSwap={hasCompare ? () => { const tmp = primaryId; setPrimaryId(compareId); setCompareId(tmp); } : undefined}
                onRemove={() => {
                  if (compareId) { setPrimaryId(compareId); setCompareId(null); }
                  else { setPrimaryId(null); }
                }}
              />
              {hasCompare && comparePlayer && (
                <PlayerHeader player={comparePlayer} prediction={comparePrediction} color="orange"
                  onSwap={() => { const tmp = primaryId; setPrimaryId(compareId); setCompareId(tmp); }}
                  onRemove={() => setCompareId(null)}
                />
              )}
            </div>
          )}

          {/* EOS Predictions */}
          {primaryPlayer && (primaryPrediction || comparePrediction) && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">End-of-Season Prediction</div>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
                {primaryPrediction && <PredictionStats prediction={primaryPrediction} label={hasCompare ? primaryPlayer.name : 'Predicted'} color="blue" />}
                {hasCompare && comparePrediction && (
                  <PredictionStats prediction={comparePrediction} label={comparePlayer!.name} color="orange" />
                )}
              </div>
            </div>
          )}

          {/* Season Stats */}
          {primaryPlayer && (latestSeason || compareLatest) && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">Latest Season</div>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
                {latestSeason && (
                  <div className={hasCompare ? 'bg-blue-50/50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/30 rounded-lg p-3' : ''}>
                    {hasCompare && <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">{primaryPlayer.name}</div>}
                    <SeasonRow season={latestSeason} />
                  </div>
                )}
                {hasCompare && compareLatest && (
                  <div className="bg-orange-50/50 dark:bg-orange-950/20 border border-orange-100 dark:border-orange-900/30 rounded-lg p-3">
                    <div className="text-xs font-medium text-orange-500 dark:text-orange-400 mb-1">{comparePlayer!.name}</div>
                    <SeasonRow season={compareLatest} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* What-If */}
          {primaryPlayer && primaryPrediction && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4 sm:p-6">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">What-If Scenario</h3>
              <div className="flex items-center gap-3">
                <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-14">PPG</label>
                <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfPpg}</span>
              </div>

              <WhatIfChart
                position={primaryPrediction.position}
                startKtc={primaryPrediction.start_ktc}
                gamesPlayed={17}
                currentPpg={whatIfPpg}
                modelId={selectedModelId}
                playerId={primaryId!}
                compare={hasCompare && comparePlayer && comparePrediction ? {
                  name: comparePlayer.name,
                  position: comparePlayer.position,
                  startKtc: comparePrediction.start_ktc,
                  age: compareLatest?.age,
                  playerId: compareId!,
                } : undefined}
              />

              {(whatIfResult || compareWhatIfResult) && (
                <div className={`mt-4 grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
                  {whatIfResult ? (
                    <PredictionStats prediction={whatIfResult} label={hasCompare ? primaryPlayer.name : 'What-If'} color="blue" />
                  ) : <div />}
                  {hasCompare && compareWhatIfResult && (
                    <PredictionStats prediction={compareWhatIfResult} label={comparePlayer!.name} color="orange" />
                  )}
                </div>
              )}
            </div>
          )}

          {/* Comps */}
          {primaryId && (
            <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
              <PlayerComps playerId={primaryId} modelId={selectedModelId} />
              {hasCompare && compareId && <PlayerComps playerId={compareId} modelId={selectedModelId} />}
            </div>
          )}

          {/* Historical */}
          {primaryId && (
            <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
              <HistoricalAccuracy playerId={primaryId} />
              {hasCompare && compareId && <HistoricalAccuracy playerId={compareId} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
