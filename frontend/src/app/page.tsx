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
import { EditablePpg } from '../components/EditablePpg';
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

  // What-If (independent per player)
  const [whatIfPpg, setWhatIfPpg] = useState(15);
  const [compareWhatIfPpg, setCompareWhatIfPpg] = useState(15);
  const [whatIfResult, setWhatIfResult] = useState<EOSPrediction | null>(null);
  const [compareWhatIfResult, setCompareWhatIfResult] = useState<EOSPrediction | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const compareDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // PPG overrides for inline editing in ranking list
  const [ppgOverrides, setPpgOverrides] = useState<Map<string, number>>(new Map());
  const [overridePredictions, setOverridePredictions] = useState<Map<string, {
    loading: boolean;
    predicted_end_ktc: number | null;
  }>>(new Map());

  const handlePpgCommit = useCallback(async (playerId: string, ppg: number) => {
    setPpgOverrides(prev => { const next = new Map(prev); next.set(playerId, ppg); return next; });
    setOverridePredictions(prev => {
      const next = new Map(prev);
      next.set(playerId, { loading: true, predicted_end_ktc: prev.get(playerId)?.predicted_end_ktc ?? null });
      return next;
    });
    try {
      const batch = await predictPlayerWhatIfBatch(playerId, { games_played: 17, ppg_values: [ppg] }, selectedModelId);
      const pred = batch?.predictions[0]?.predicted_end_ktc ?? null;
      setOverridePredictions(prev => {
        const next = new Map(prev);
        next.set(playerId, { loading: false, predicted_end_ktc: pred });
        return next;
      });
    } catch {
      setOverridePredictions(prev => {
        const next = new Map(prev);
        next.set(playerId, { loading: false, predicted_end_ktc: null });
        return next;
      });
    }
  }, [selectedModelId]);

  const handlePpgReset = useCallback((playerId: string) => {
    setPpgOverrides(prev => { const next = new Map(prev); next.delete(playerId); return next; });
    setOverridePredictions(prev => { const next = new Map(prev); next.delete(playerId); return next; });
  }, []);

  const handleResetAllPpg = useCallback(() => {
    setPpgOverrides(new Map());
    setOverridePredictions(new Map());
  }, []);

  // Clear stale override predictions when model changes
  useEffect(() => {
    if (ppgOverrides.size === 0) return;
    setOverridePredictions(new Map());
    ppgOverrides.forEach((ppg, playerId) => { handlePpgCommit(playerId, ppg); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId]);

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

  // Re-sort client-side when PPG overrides affect sort order
  const sortedResults = useMemo(() => {
    if (ppgOverrides.size === 0 && overridePredictions.size === 0) return searchResults;

    const sign = sortDesc ? -1 : 1;
    return [...searchResults].sort((a, b) => {
      const getValue = (p: PlayerSummary) => {
        const oPpg = ppgOverrides.get(p.player_id);
        const oPred = overridePredictions.get(p.player_id);
        const pred = oPred ? oPred.predicted_end_ktc : p.predicted_end_ktc;

        if (sortBy === 'ppg') return oPpg ?? p.ppg ?? null;
        if (sortBy === 'predicted') return pred ?? null;
        if (sortBy === 'change') {
          return (pred != null && p.latest_ktc != null) ? pred - p.latest_ktc : null;
        }
        if (sortBy === 'ktc') return p.latest_ktc;
        return null;
      };
      const va = getValue(a);
      const vb = getValue(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return (va - vb) * sign;
    });
  }, [searchResults, ppgOverrides, overridePredictions, sortBy, sortDesc]);

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
        // Initialize What-If PPG: use override > search result projected PPG > last-season PPG
        const overridePpgVal = ppgOverrides.get(primaryId);
        if (overridePpgVal !== undefined) {
          setWhatIfPpg(overridePpgVal);
        } else {
          const searchEntry = searchResults.find(p => p.player_id === primaryId);
          if (searchEntry?.ppg != null) {
            setWhatIfPpg(searchEntry.ppg);
          } else if (player.seasons.length > 0) {
            const latest = player.seasons.reduce((a, b) => (a.year > b.year ? a : b));
            setWhatIfPpg(Math.round((latest.games_played > 0 ? latest.fantasy_points / latest.games_played : 15) * 2) / 2);
          }
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
        // Initialize compare PPG from override > search projected > last-season
        const cOverride = ppgOverrides.get(compareId);
        if (cOverride !== undefined) {
          setCompareWhatIfPpg(cOverride);
        } else {
          const searchEntry = searchResults.find(p => p.player_id === compareId);
          if (searchEntry?.ppg != null) {
            setCompareWhatIfPpg(searchEntry.ppg);
          } else if (player.seasons.length > 0) {
            const latest = player.seasons.reduce((a, b) => (a.year > b.year ? a : b));
            setCompareWhatIfPpg(Math.round((latest.games_played > 0 ? latest.fantasy_points / latest.games_played : 15) * 2) / 2);
          }
        }
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

  // What-If for compare (independent PPG)
  const fetchCompareWhatIf = useCallback(async () => {
    if (!compareId || !compare?.prediction) return;
    try {
      const batch = await predictPlayerWhatIfBatch(compareId, { games_played: 17, ppg_values: [compareWhatIfPpg] }, selectedModelId);
      setCompareWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
  }, [compareId, compare?.prediction, compareWhatIfPpg, selectedModelId]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 200);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchWhatIf]);

  useEffect(() => {
    if (compareDebounceRef.current) clearTimeout(compareDebounceRef.current);
    compareDebounceRef.current = setTimeout(fetchCompareWhatIf, 200);
    return () => { if (compareDebounceRef.current) clearTimeout(compareDebounceRef.current); };
  }, [fetchCompareWhatIf]);

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
        <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-1 tracking-tight">KTC Value Predictor</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400/80">Select up to two players to compare predictions</p>
      </div>

      {/* Search bar + position filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <input
          type="text"
          placeholder="Search players..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 px-4 py-3 rounded-xl border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white text-base placeholder-gray-400 dark:placeholder-gray-500 input-recessed backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
        />
        <div className="flex gap-2">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              aria-label={`Filter by ${pos === 'All' ? 'all positions' : pos}`}
              className={`px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                position === pos
                  ? 'btn-glossy text-white'
                  : 'bg-white/60 dark:bg-white/5 text-gray-700 dark:text-gray-300 border border-gray-200/60 dark:border-white/10 hover:bg-white dark:hover:bg-white/10 btn-metal'
              }`}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {/* Sort options */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500 dark:text-gray-400/80">Sort:</span>
        {([['ktc', 'Value'], ['predicted', 'Predicted'], ['change', 'Change'], ['ppg', 'PPG']] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => {
              if (sortBy === key) setSortDesc((d) => !d);
              else { setSortBy(key); setSortDesc(true); }
            }}
            aria-label={`Sort by ${label}`}
            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
              sortBy === key
                ? 'btn-glossy text-white'
                : 'bg-white/50 dark:bg-white/5 text-gray-600 dark:text-gray-400 hover:bg-white/80 dark:hover:bg-white/10 border border-transparent dark:border-white/5'
            }`}
          >
            {label}
            {sortBy === key && (
              <span className="ml-1">{sortDesc ? '\u25BE' : '\u25B4'}</span>
            )}
          </button>
        ))}
        {ppgOverrides.size > 0 && (
          <button
            onClick={handleResetAllPpg}
            className="ml-auto px-2 py-1 rounded-md text-xs font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all"
          >
            Reset PPG edits ({ppgOverrides.size})
          </button>
        )}
      </div>

      {/* Search results as selectable cards */}
      {searchLoading ? (
        <div className="flex justify-center py-8">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
        {/* Fixed header */}
        <div className="px-4 py-2.5 flex items-center gap-3 bg-gray-100/50 dark:bg-white/[0.03] border-b border-gray-200/50 dark:border-white/5 sticky top-0 z-10 text-xs text-gray-500 dark:text-gray-400/80 font-semibold uppercase tracking-wider">
          <span className="w-6 text-right shrink-0">#</span>
          <span className="flex-1 min-w-0">Player</span>
          <span className="w-7 text-center shrink-0">Pos</span>
          <span className="w-12 text-right shrink-0">PPG</span>
          <span className="w-14 text-right shrink-0">Value</span>
          <span className="w-14 text-right shrink-0">Pred</span>
          <span className="w-14 text-right shrink-0">+/-</span>
        </div>
        <div className="max-h-[480px] overflow-y-auto">
        <div className="divide-y divide-gray-100/50 dark:divide-white/[0.04]">
          {sortedResults.map((player, idx) => {
            const isPrimary = player.player_id === primaryId;
            const isCompare = player.player_id === compareId;
            const override = overridePredictions.get(player.player_id);
            const overridePpg = ppgOverrides.get(player.player_id);
            const displayPred = override ? override.predicted_end_ktc : player.predicted_end_ktc;
            const isRowLoading = override?.loading ?? false;
            const delta = (displayPred != null && player.latest_ktc != null)
              ? displayPred - player.latest_ktc : null;
            return (
              <div
                key={player.player_id}
                onClick={() => handleSelectPlayer(player)}
                className={`w-full text-left px-4 py-2.5 flex items-center gap-3 transition-all duration-150 cursor-pointer shine-hover ${
                  isPrimary
                    ? 'bg-blue-500/10 dark:bg-blue-500/10 border-l-2 border-l-blue-500'
                    : isCompare
                    ? 'bg-orange-500/10 dark:bg-orange-500/10 border-l-2 border-l-orange-500'
                    : 'border-l-2 border-l-transparent hover:bg-white/50 dark:hover:bg-white/[0.04]'
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
                <EditablePpg
                  defaultValue={player.ppg}
                  overrideValue={overridePpg}
                  isLoading={isRowLoading}
                  onCommit={(ppg) => handlePpgCommit(player.player_id, ppg)}
                  onReset={() => handlePpgReset(player.player_id)}
                />
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-14 text-right shrink-0">
                  {player.latest_ktc != null ? formatKtc(player.latest_ktc) : '—'}
                </span>
                {isRowLoading ? (
                  <span className="w-14 flex justify-end shrink-0">
                    <span className="w-3 h-3 border border-gray-300 dark:border-gray-600 border-t-blue-500 rounded-full animate-spin" />
                  </span>
                ) : (
                  <span className="text-xs text-gray-400 dark:text-gray-500 w-14 text-right shrink-0">
                    {displayPred != null ? formatKtc(displayPred) : '—'}
                  </span>
                )}
                {isRowLoading ? (
                  <span className="w-14 flex justify-end shrink-0">
                    <span className="w-3 h-3 border border-gray-300 dark:border-gray-600 border-t-blue-500 rounded-full animate-spin" />
                  </span>
                ) : (
                  <span className={`text-xs font-semibold w-14 text-right shrink-0 ${
                    delta == null ? 'text-gray-400' : delta >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {delta != null ? `${delta >= 0 ? '+' : ''}${Math.round(delta).toLocaleString()}` : '—'}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        </div>
        </div>
      )}

      {!searchLoading && sortedResults.length === 0 && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-400">No players found.</div>
      )}

      {/* Show TopMovers only when no player is selected */}
      {!hasAnySelected && <TopMovers onSelect={(id) => setPrimaryId(id)} />}

      {/* ══════════════════════════════════════════════════════════════════
          Player detail sections (shown when at least one player selected)
          ══════════════════════════════════════════════════════════════════ */}
      {hasAnySelected && (
        <div className="space-y-6 pt-4 border-t border-gray-200/40 dark:border-white/[0.06]">
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

          {/* EOS Predictions — use What-If result (tied to current PPG) over base prediction */}
          {primaryPlayer && (whatIfResult || primaryPrediction || compareWhatIfResult || comparePrediction) && (
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">End-of-Season Prediction</h3>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
                {(whatIfResult || primaryPrediction) && <PredictionStats prediction={whatIfResult ?? primaryPrediction!} label={hasCompare ? primaryPlayer.name : 'Predicted'} color="blue" />}
                {hasCompare && (compareWhatIfResult || comparePrediction) && (
                  <PredictionStats prediction={compareWhatIfResult ?? comparePrediction!} label={comparePlayer!.name} color="orange" />
                )}
              </div>
            </div>
          )}

          {/* Season Stats */}
          {primaryPlayer && (latestSeason || compareLatest) && (
            <div className="glass-card p-4">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Latest Season</h3>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
                {latestSeason && (
                  <div className={hasCompare ? 'bg-blue-500/5 dark:bg-blue-500/10 border border-blue-200/30 dark:border-blue-500/15 rounded-xl p-3' : ''}>
                    {hasCompare && <div className="text-xs font-medium text-blue-500 dark:text-blue-400 mb-1">{primaryPlayer.name}</div>}
                    <SeasonRow season={latestSeason} />
                  </div>
                )}
                {hasCompare && compareLatest && (
                  <div className="bg-orange-500/5 dark:bg-orange-500/10 border border-orange-200/30 dark:border-orange-500/15 rounded-xl p-3">
                    <div className="text-xs font-medium text-orange-500 dark:text-orange-400 mb-1">{comparePlayer!.name}</div>
                    <SeasonRow season={compareLatest} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* What-If */}
          {primaryPlayer && (primaryPrediction || primaryPlayer.live_ktc || primaryPlayer.seasons?.length > 0) && (
            <div className="glass-card p-4 sm:p-6">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">What-If Scenario</h3>
              <div className={`grid gap-4 ${hasCompare ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
                {/* Primary slider */}
                <div>
                  {hasCompare && <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">{primaryPlayer.name}</div>}
                  <div className="flex items-center gap-3">
                    <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-10">PPG</label>
                    <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" aria-label={`${primaryPlayer.name} PPG slider`} />
                    <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfPpg}</span>
                  </div>
                  {latestSeason && latestSeason.games_played > 0 && (
                    <div className="text-xs text-gray-400 dark:text-gray-500 mt-1 ml-10">
                      Last season: {(latestSeason.fantasy_points / latestSeason.games_played).toFixed(1)} ppg
                    </div>
                  )}
                </div>
                {/* Compare slider */}
                {hasCompare && comparePlayer && (
                  <div>
                    <div className="text-xs font-medium text-orange-500 dark:text-orange-400 mb-1">{comparePlayer.name}</div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-10">PPG</label>
                      <input type="range" min="0" max="25" step="0.5" value={compareWhatIfPpg} onChange={(e) => setCompareWhatIfPpg(parseFloat(e.target.value))} className="flex-1" aria-label={`${comparePlayer.name} PPG slider`} />
                      <span className="text-sm font-bold text-orange-500 dark:text-orange-400 w-8 text-center">{compareWhatIfPpg}</span>
                    </div>
                    {compareLatest && compareLatest.games_played > 0 && (
                      <div className="text-xs text-gray-400 dark:text-gray-500 mt-1 ml-10">
                        Last season: {(compareLatest.fantasy_points / compareLatest.games_played).toFixed(1)} ppg
                      </div>
                    )}
                  </div>
                )}
              </div>

              <WhatIfChart
                position={primaryPrediction?.position ?? primaryPlayer.position}
                startKtc={primaryPlayer.live_ktc ?? primaryPrediction?.start_ktc ?? 0}
                gamesPlayed={17}
                currentPpg={whatIfPpg}
                modelId={selectedModelId}
                playerId={primaryId!}
                comparePpg={hasCompare ? compareWhatIfPpg : undefined}
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
