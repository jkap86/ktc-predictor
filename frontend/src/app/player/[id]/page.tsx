'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { getPlayer, getPrediction, predictPlayerWhatIfBatch } from '../../../lib/api';
import { formatKtc } from '../../../lib/format';
import { useModel } from '../../../context/ModelContext';
import WhatIfChart from '../../../components/WhatIfChart';
import ComparePlayerPicker from '../../../components/ComparePlayerPicker';
import HistoricalAccuracy from '../../../components/HistoricalAccuracy';
import PlayerComps from '../../../components/PlayerComps';
import type { Player, PlayerSummary, EOSPrediction } from '../../../types/player';

// ── Shared sub-components ──────────────────────────────────────────────

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

/** Compact player header: name + position + KTC, with optional remove/swap */
function PlayerHeader({ player, prediction, color = 'blue', onRemove, onSwap, showActions = false }: {
  player: Player; prediction: EOSPrediction | null; color?: 'blue' | 'orange';
  onRemove?: () => void; onSwap?: () => void; showActions?: boolean;
}) {
  const accent = color === 'orange' ? 'text-orange-500 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400';
  const border = color === 'orange' ? 'border-orange-200 dark:border-orange-800' : 'border-blue-200 dark:border-blue-800';
  const bg = color === 'orange'
    ? 'bg-orange-50 dark:bg-orange-950/30'
    : 'bg-blue-50 dark:bg-blue-950/30';
  return (
    <div className={`${bg} rounded-xl shadow-sm border ${border} p-4 flex items-center justify-between gap-2`}>
      <div className="min-w-0">
        <h2 className={`text-lg font-bold truncate ${accent}`}>{player.name}</h2>
        <span className="inline-block px-2 py-0.5 bg-white/60 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded text-xs mt-0.5">
          {player.position}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {showActions && (
          <div className="flex items-center gap-1">
            {onSwap && (
              <button onClick={onSwap} className={`p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors ${accent}`} title="Swap players">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                </svg>
              </button>
            )}
            {onRemove && (
              <button onClick={onRemove} className={`p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors ${accent}`} title="Remove player">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
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

/** Single prediction stat row */
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

/** Season stats as a horizontal row */
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
      <span className="text-gray-600 dark:text-gray-400">#{season.start_position_rank}</span>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────

export default function PlayerPage() {
  const params = useParams();
  const router = useRouter();
  const playerId = params.id as string;
  const { selectedModelId } = useModel();

  const [player, setPlayer] = useState<Player | null>(null);
  const [prediction, setPrediction] = useState<EOSPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [whatIfGames, setWhatIfGames] = useState(17);
  const [whatIfPpg, setWhatIfPpg] = useState(15);
  const [whatIfResult, setWhatIfResult] = useState<EOSPrediction | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [comparePlayer, setComparePlayer] = useState<PlayerSummary | null>(null);
  const [compareData, setCompareData] = useState<Player | null>(null);
  const [comparePrediction, setComparePrediction] = useState<EOSPrediction | null>(null);
  const [compareResult, setCompareResult] = useState<EOSPrediction | null>(null);

  // Fetch primary
  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const pd = await getPlayer(playerId);
        setPlayer(pd);
        if (pd.seasons.length > 0) {
          const latest = pd.seasons.reduce((a, b) => (a.year > b.year ? a : b));
          setWhatIfGames(latest.games_played);
          setWhatIfPpg(Math.round((latest.games_played > 0 ? latest.fantasy_points / latest.games_played : 15) * 2) / 2);
        }
        const pred = await getPrediction(playerId, selectedModelId);
        setPrediction(pred);
      } catch { setError('Failed to load player data'); }
      finally { setLoading(false); }
    })();
  }, [playerId, selectedModelId]);

  // Fetch compare
  useEffect(() => {
    if (!comparePlayer) { setCompareData(null); setComparePrediction(null); setCompareResult(null); return; }
    (async () => {
      try {
        const pd = await getPlayer(comparePlayer.player_id);
        setCompareData(pd);
        setComparePrediction(await getPrediction(comparePlayer.player_id, selectedModelId));
      } catch { setCompareData(null); setComparePrediction(null); }
    })();
  }, [comparePlayer, selectedModelId]);

  // What-if for compare (player-aware)
  useEffect(() => {
    if (!compareData || !comparePlayer) { setCompareResult(null); return; }
    (async () => {
      try {
        const batch = await predictPlayerWhatIfBatch(
          comparePlayer.player_id,
          { games_played: whatIfGames, ppg_values: [whatIfPpg] },
          selectedModelId,
        );
        setCompareResult(batch?.predictions[0] ?? null);
      } catch { setCompareResult(null); }
    })();
  }, [compareData, comparePlayer, whatIfGames, whatIfPpg, selectedModelId]);

  // What-if for primary (player-aware: uses full feature context)
  const fetchWhatIf = useCallback(async () => {
    if (!prediction) return;
    setWhatIfLoading(true);
    try {
      const batch = await predictPlayerWhatIfBatch(
        playerId,
        { games_played: whatIfGames, ppg_values: [whatIfPpg] },
        selectedModelId,
      );
      setWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
    finally { setWhatIfLoading(false); }
  }, [prediction, playerId, whatIfGames, whatIfPpg, selectedModelId]);

  useEffect(() => {
    if (!prediction) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 200);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchWhatIf, prediction]);

  if (loading) return <div className="flex justify-center py-12"><div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" /></div>;
  if (error || !player) return <div className="text-center py-12"><div className="text-red-500 mb-4">{error || 'Player not found'}</div><Link href="/" className="text-blue-600 hover:text-blue-700">Back</Link></div>;

  const latestSeason = player.seasons.length > 0 ? player.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null;
  const compareLatest = compareData?.seasons.length ? compareData.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null;
  const hasCompare = !!compareData && !!comparePlayer;

  return (
    <div className="space-y-6">
      {/* Nav + compare picker */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <Link href="/" className="text-blue-600 dark:text-blue-400 hover:text-blue-700 text-sm">&larr; Back</Link>
        <div className="w-full sm:w-72">
          <ComparePlayerPicker selected={comparePlayer} onSelect={setComparePlayer} />
        </div>
      </div>

      {/* ── SECTION: Player Headers ── */}
      <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
        <PlayerHeader
          player={player}
          prediction={prediction}
          color="blue"
          showActions={hasCompare}
          onSwap={hasCompare && comparePlayer ? () => {
            const newCompare: PlayerSummary = { player_id: playerId, name: player.name, position: player.position, latest_ktc: player.live_ktc ?? prediction?.start_ktc ?? 0 };
            router.push(`/player/${comparePlayer.player_id}`);
            setTimeout(() => setComparePlayer(newCompare), 100);
          } : undefined}
          onRemove={hasCompare && comparePlayer ? () => {
            router.push(`/player/${comparePlayer.player_id}`);
            setComparePlayer(null);
          } : undefined}
        />
        {hasCompare && compareData && (
          <PlayerHeader
            player={compareData}
            prediction={comparePrediction}
            color="orange"
            showActions
            onSwap={comparePlayer ? () => {
              const newCompare: PlayerSummary = { player_id: playerId, name: player.name, position: player.position, latest_ktc: player.live_ktc ?? prediction?.start_ktc ?? 0 };
              router.push(`/player/${comparePlayer.player_id}`);
              setTimeout(() => setComparePlayer(newCompare), 100);
            } : undefined}
            onRemove={() => setComparePlayer(null)}
          />
        )}
      </div>

      {/* ── SECTION: EOS Predictions side by side ── */}
      {(prediction || comparePrediction) && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">End-of-Season Prediction</div>
          <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
            {prediction && <PredictionStats prediction={prediction} label={hasCompare ? player.name : 'Predicted'} color="blue" />}
            {hasCompare && comparePrediction && comparePlayer && (
              <PredictionStats prediction={comparePrediction} label={comparePlayer.name} color="orange" />
            )}
          </div>
        </div>
      )}

      {/* ── SECTION: Season Stats side by side ── */}
      {(latestSeason || compareLatest) && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-3">Latest Season</div>
          <div className={`grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
            {latestSeason && (
              <div className={hasCompare ? 'bg-blue-50/50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/30 rounded-lg p-3' : ''}>
                {hasCompare && <div className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-1">{player.name}</div>}
                <SeasonRow season={latestSeason} />
              </div>
            )}
            {hasCompare && compareLatest && comparePlayer && (
              <div className="bg-orange-50/50 dark:bg-orange-950/20 border border-orange-100 dark:border-orange-900/30 rounded-lg p-3">
                <div className="text-xs font-medium text-orange-500 dark:text-orange-400 mb-1">{comparePlayer.name}</div>
                <SeasonRow season={compareLatest} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SECTION: What-If Controls + Chart ── */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4 sm:p-6">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">What-If Scenario</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-14">Games</label>
              <input type="range" min="0" max="17" value={whatIfGames} onChange={(e) => setWhatIfGames(parseInt(e.target.value))} className="flex-1" />
              <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfGames}</span>
            </div>
            <div className="flex items-center gap-3">
              <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-14">PPG</label>
              <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" />
              <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfPpg}</span>
            </div>
          </div>

          <WhatIfChart
            position={prediction.position}
            startKtc={prediction.start_ktc}
            gamesPlayed={whatIfGames}
            currentPpg={whatIfPpg}
            modelId={selectedModelId}
            playerId={playerId}
            compare={hasCompare && compareData && comparePlayer ? {
              name: comparePlayer.name,
              position: compareData.position,
              startKtc: comparePlayer.latest_ktc ?? prediction.start_ktc,
              age: compareLatest?.age,
              playerId: comparePlayer.player_id,
            } : undefined}
          />

          {/* What-If results inline */}
          {(whatIfResult || compareResult) && (
            <div className={`mt-4 grid gap-4 ${hasCompare ? 'grid-cols-2' : 'grid-cols-1'}`}>
              {whatIfLoading ? (
                <div className="flex justify-center py-3">
                  <div className="w-5 h-5 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
                </div>
              ) : whatIfResult ? (
                <PredictionStats prediction={whatIfResult} label={hasCompare ? player.name : 'What-If'} color="blue" />
              ) : <div />}
              {hasCompare && compareResult && comparePlayer && (
                <PredictionStats prediction={compareResult} label={comparePlayer.name} color="orange" />
              )}
            </div>
          )}
        </div>
      )}

      {/* ── SECTION: Comps ── */}
      <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
        <PlayerComps playerId={playerId} modelId={selectedModelId} />
        {hasCompare && comparePlayer && <PlayerComps playerId={comparePlayer.player_id} modelId={selectedModelId} />}
      </div>

      {/* ── SECTION: Historical ── */}
      <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>
        <HistoricalAccuracy playerId={playerId} />
        {hasCompare && comparePlayer && <HistoricalAccuracy playerId={comparePlayer.player_id} />}
      </div>
    </div>
  );
}
