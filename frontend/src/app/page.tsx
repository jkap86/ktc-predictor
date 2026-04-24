'use client';

import { useState, useEffect, useCallback, useRef, useMemo, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { searchPlayers, getPlayer, getPrediction, predictPlayerWhatIfBatch } from '../lib/api';
import { formatKtc } from '../lib/format';
import { useModel } from '../context/ModelContext';
import WhatIfChart from '../components/WhatIfChart';
import PlayerComps from '../components/PlayerComps';
import HistoricalAccuracy from '../components/HistoricalAccuracy';
import TopMovers from '../components/TopMovers';
import { ConfidenceBand, PredictionStats, SeasonRow, PlayerHeader } from '../components/PlayerCards';
import type { Player, PlayerSummary, EOSPrediction } from '../types/player';

const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];

// ── Main page ────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <Suspense fallback={<div className="flex justify-center py-12"><div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" /></div>}>
      <HomeContent />
    </Suspense>
  );
}

function HomeContent() {
  const { selectedModelId } = useModel();
  const searchParams = useSearchParams();
  const router = useRouter();

  // Search state
  const [query, setQuery] = useState('');
  const [position, setPosition] = useState('All');
  const [sortBy, setSortBy] = useState<'ktc' | 'predicted' | 'change' | 'ppg'>('ktc');
  const [sortDesc, setSortDesc] = useState(true);
  const [searchResults, setSearchResults] = useState<PlayerSummary[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  // Selected players — initialize from URL params
  const [primaryId, setPrimaryIdRaw] = useState<string | null>(searchParams.get('p1'));
  const [compareId, setCompareIdRaw] = useState<string | null>(searchParams.get('p2'));

  // Sync selection to URL params
  const setPrimaryId = useCallback((id: string | null) => {
    setPrimaryIdRaw(id);
    const params = new URLSearchParams(window.location.search);
    if (id) params.set('p1', id); else params.delete('p1');
    // Keep p2 only if p1 exists
    if (!id) params.delete('p2');
    const qs = params.toString();
    router.replace(qs ? `?${qs}` : '/', { scroll: false });
  }, [router]);

  const setCompareId = useCallback((id: string | null) => {
    setCompareIdRaw(id);
    const params = new URLSearchParams(window.location.search);
    if (id) params.set('p2', id); else params.delete('p2');
    const qs = params.toString();
    router.replace(qs ? `?${qs}` : '/', { scroll: false });
  }, [router]);

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
        const result = await searchPlayers(query, pos, 200, sortBy, sortDesc ? 'desc' : 'asc');
        setSearchResults(result.players);
      } catch { /* */ }
      finally { setSearchLoading(false); }
    };
    const debounce = setTimeout(fetchPlayers, 300);
    return () => clearTimeout(debounce);
  }, [query, position, sortBy, sortDesc]);

  // Fetch primary player data
  useEffect(() => {
    if (!primaryId) { setPrimary(null); setWhatIfResult(null); return; }
    setPrimary(null);
    setWhatIfResult(null);
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
    setCompare(null);
    setCompareWhatIfResult(null);
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
    if (!primaryId || !primary?.player) return;
    try {
      const batch = await predictPlayerWhatIfBatch(primaryId, { games_played: 17, ppg_values: [whatIfPpg] }, selectedModelId);
      setWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
  }, [primaryId, primary?.player, whatIfPpg, selectedModelId]);

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
  const latestSeason = useMemo(() => primaryPlayer?.seasons.length ? primaryPlayer.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null, [primaryPlayer]);
  const compareLatest = useMemo(() => comparePlayer?.seasons.length ? comparePlayer.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null, [comparePlayer]);

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
              aria-label={`Filter by ${pos === 'All' ? 'all positions' : pos}`}
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

      {/* Sort options */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 dark:text-gray-400">Sort:</span>
        {([['ktc', 'Value'], ['predicted', 'Predicted'], ['change', 'Change'], ['ppg', 'PPG']] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => {
              if (sortBy === key) setSortDesc((d) => !d);
              else { setSortBy(key); setSortDesc(true); }
            }}
            aria-label={`Sort by ${label}`}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
              sortBy === key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            {label}
            {sortBy === key && (
              <span className="ml-1">{sortDesc ? '\u25BE' : '\u25B4'}</span>
            )}
          </button>
        ))}
      </div>

      {/* Search results as selectable cards */}
      {searchLoading ? (
        <div className="flex justify-center py-8">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="rounded-xl border border-gray-100 dark:border-gray-700">
        {/* Fixed header */}
        <div className="px-4 py-2 flex items-center gap-3 bg-gray-50 dark:bg-gray-800 border-b border-gray-100 dark:border-gray-700 sticky top-0 z-10 text-xs text-gray-500 dark:text-gray-400 font-medium">
          <span className="w-6 text-right shrink-0">#</span>
          <span className="flex-1 min-w-0">Player</span>
          <span className="w-7 text-center shrink-0">Pos</span>
          <span className="w-10 text-right shrink-0">PPG</span>
          <span className="w-14 text-right shrink-0">Value</span>
          <span className="w-14 text-right shrink-0">Pred</span>
          <span className="w-14 text-right shrink-0">+/-</span>
        </div>
        <div className="max-h-[480px] overflow-y-auto">
        <div className="divide-y divide-gray-100 dark:divide-gray-700">
          {searchResults.map((player, idx) => {
            const isPrimary = player.player_id === primaryId;
            const isCompare = player.player_id === compareId;
            const delta = (player.predicted_end_ktc != null && player.latest_ktc != null)
              ? player.predicted_end_ktc - player.latest_ktc : null;
            return (
              <button
                key={player.player_id}
                onClick={() => handleSelectPlayer(player)}
                className={`w-full text-left px-4 py-2.5 flex items-center gap-3 transition-colors ${
                  isPrimary
                    ? 'bg-blue-50 dark:bg-blue-950/30'
                    : isCompare
                    ? 'bg-orange-50 dark:bg-orange-950/30'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <span className="text-xs text-gray-400 dark:text-gray-500 w-6 text-right shrink-0">{idx + 1}</span>
                <span className={`font-medium text-sm truncate min-w-0 flex-1 ${isPrimary ? 'text-blue-700 dark:text-blue-300' : isCompare ? 'text-orange-700 dark:text-orange-300' : 'text-gray-900 dark:text-white'}`}>
                  {player.name}
                  {(isPrimary || isCompare) && (
                    <span className={`ml-1.5 text-xs ${isPrimary ? 'text-blue-400' : 'text-orange-400'}`}>
                      {isPrimary ? '(1)' : '(2)'}
                    </span>
                  )}
                </span>
                <span className="text-xs text-gray-400 dark:text-gray-500 w-7 text-center shrink-0">{player.position}</span>
                <span className="text-xs text-gray-500 dark:text-gray-400 w-10 text-right shrink-0">{player.ppg ?? '—'}</span>
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-14 text-right shrink-0">
                  {player.latest_ktc != null ? formatKtc(player.latest_ktc) : '—'}
                </span>
                <span className="text-xs text-gray-400 dark:text-gray-500 w-14 text-right shrink-0">
                  {player.predicted_end_ktc != null ? formatKtc(player.predicted_end_ktc) : '—'}
                </span>
                <span className={`text-xs font-semibold w-14 text-right shrink-0 ${
                  delta == null ? 'text-gray-400' : delta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                }`}>
                  {delta != null ? `${delta >= 0 ? '+' : ''}${Math.round(delta).toLocaleString()}` : '—'}
                </span>
              </button>
            );
          })}
        </div>
        </div>
        </div>
      )}

      {!searchLoading && searchResults.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">No players found.</div>
      )}

      {/* Show TopMovers only when no player is selected */}
      {!hasAnySelected && <TopMovers onSelect={(id) => setPrimaryId(id)} />}

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
            <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
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
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">End-of-Season Prediction</h3>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
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
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Latest Season</h3>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
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
          {primaryPlayer && (primaryPrediction || primaryPlayer.live_ktc || primaryPlayer.seasons?.length > 0) && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4 sm:p-6">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">What-If Scenario</h3>
              <div className="flex items-center gap-3">
                <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-14">PPG</label>
                <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" aria-label="PPG slider" />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfPpg}</span>
              </div>
              {latestSeason && latestSeason.games_played > 0 && (
                <div className="text-xs text-gray-400 dark:text-gray-500 mt-1 ml-14">
                  Last season: {(latestSeason.fantasy_points / latestSeason.games_played).toFixed(1)} ppg
                </div>
              )}

              <WhatIfChart
                position={primaryPrediction?.position ?? primaryPlayer.position}
                startKtc={primaryPrediction?.start_ktc ?? primaryPlayer.live_ktc ?? 0}
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
                <div className={`mt-4 grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
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
            <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
              <PlayerComps playerId={primaryId} modelId={selectedModelId} />
              {hasCompare && compareId && <PlayerComps playerId={compareId} modelId={selectedModelId} />}
            </div>
          )}

          {/* Historical */}
          {primaryId && (
            <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
              <HistoricalAccuracy playerId={primaryId} />
              {hasCompare && compareId && <HistoricalAccuracy playerId={compareId} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
