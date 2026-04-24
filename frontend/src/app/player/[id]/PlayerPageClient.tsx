'use client';

import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { getPlayer, getPrediction, predictPlayerWhatIfBatch } from '../../../lib/api';
import { useModel } from '../../../context/ModelContext';
import WhatIfChart from '../../../components/WhatIfChart';
import ComparePlayerPicker from '../../../components/ComparePlayerPicker';
import HistoricalAccuracy from '../../../components/HistoricalAccuracy';
import PlayerComps from '../../../components/PlayerComps';
import { PredictionStats, SeasonRow, PlayerHeader } from '../../../components/PlayerCards';
import type { Player, PlayerSummary, EOSPrediction } from '../../../types/player';

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
          { games_played: 17, ppg_values: [whatIfPpg] },
          selectedModelId,
        );
        setCompareResult(batch?.predictions[0] ?? null);
      } catch { setCompareResult(null); }
    })();
  }, [compareData, comparePlayer, 17, whatIfPpg, selectedModelId]);

  // What-if for primary (player-aware: uses full feature context)
  const fetchWhatIf = useCallback(async () => {
    if (!prediction) return;
    setWhatIfLoading(true);
    try {
      const batch = await predictPlayerWhatIfBatch(
        playerId,
        { games_played: 17, ppg_values: [whatIfPpg] },
        selectedModelId,
      );
      setWhatIfResult(batch?.predictions[0] ?? null);
    } catch { /* */ }
    finally { setWhatIfLoading(false); }
  }, [prediction, playerId, 17, whatIfPpg, selectedModelId]);

  useEffect(() => {
    if (!prediction) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 200);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchWhatIf, prediction]);

  if (loading) return <div className="flex justify-center py-12"><div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" /></div>;
  if (error || !player) return <div className="text-center py-12"><div className="text-red-500 mb-4">{error || 'Player not found'}</div><Link href="/" className="text-blue-600 hover:text-blue-700">Back</Link></div>;

  const latestSeason = useMemo(() => player.seasons.length > 0 ? player.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null, [player]);
  const compareLatest = useMemo(() => compareData?.seasons.length ? compareData.seasons.reduce((a, b) => (a.year > b.year ? a : b)) : null, [compareData]);
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
                <SeasonRow season={latestSeason} showRank />
              </div>
            )}
            {hasCompare && compareLatest && comparePlayer && (
              <div className="bg-orange-50/50 dark:bg-orange-950/20 border border-orange-100 dark:border-orange-900/30 rounded-lg p-3">
                <div className="text-xs font-medium text-orange-500 dark:text-orange-400 mb-1">{comparePlayer.name}</div>
                <SeasonRow season={compareLatest} showRank />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SECTION: What-If Controls + Chart ── */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4 sm:p-6">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">What-If Scenario</h3>
          <div className="flex items-center gap-3">
            <label className="text-xs font-medium text-gray-700 dark:text-gray-300 w-14">PPG</label>
            <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" />
            <span className="text-sm font-bold text-blue-600 dark:text-blue-400 w-8 text-center">{whatIfPpg}</span>
          </div>

          <WhatIfChart
            position={prediction.position}
            startKtc={prediction.start_ktc}
            gamesPlayed={17}
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
