'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getPlayer, getPrediction, predictEos } from '../../../lib/api';
import { formatKtc } from '../../../lib/format';
import { useModel } from '../../../context/ModelContext';
import WhatIfChart from '../../../components/WhatIfChart';
import ComparePlayerPicker from '../../../components/ComparePlayerPicker';
import HistoricalAccuracy from '../../../components/HistoricalAccuracy';
import PlayerComps from '../../../components/PlayerComps';
import type { Player, PlayerSummary, EOSPrediction } from '../../../types/player';

function ConfidenceBand({ prediction, color = 'blue' }: { prediction: EOSPrediction; color?: 'blue' | 'orange' }) {
  if (!prediction.low_end_ktc || !prediction.high_end_ktc) return null;
  const bg = color === 'orange' ? 'bg-orange-50 dark:bg-orange-900/20' : 'bg-blue-50 dark:bg-blue-900/20';
  const border = color === 'orange' ? 'border-orange-100 dark:border-orange-800' : 'border-blue-100 dark:border-blue-800';
  const label = color === 'orange' ? 'text-orange-700 dark:text-orange-300' : 'text-blue-700 dark:text-blue-300';
  const bar = color === 'orange' ? 'bg-orange-100 dark:bg-orange-800' : 'bg-blue-100 dark:bg-blue-800';
  const dot = color === 'orange' ? 'bg-orange-500 dark:bg-orange-400' : 'bg-blue-500 dark:bg-blue-400';
  return (
    <div className={`mt-2 px-3 py-2 ${bg} rounded-lg border ${border}`}>
      <div className={`text-xs font-medium ${label} mb-1`}>p20 - p80</div>
      <div className="flex items-center gap-2 text-xs">
        <span className="text-gray-600 dark:text-gray-400">{formatKtc(prediction.low_end_ktc)}</span>
        <div className={`flex-1 h-1.5 ${bar} rounded-full relative`}>
          <div
            className={`absolute h-1.5 ${dot} rounded-full`}
            style={{
              left: `${Math.max(0, Math.min(100, ((prediction.predicted_end_ktc - prediction.low_end_ktc) / (prediction.high_end_ktc - prediction.low_end_ktc)) * 100))}%`,
              width: '3px',
              transform: 'translateX(-50%)',
            }}
          />
        </div>
        <span className="text-gray-600 dark:text-gray-400">{formatKtc(prediction.high_end_ktc)}</span>
      </div>
    </div>
  );
}

function StatCard({ value, label, colored }: { value: string; label: string; colored?: boolean }) {
  return (
    <div className="text-center p-2">
      <div className={`text-lg font-bold ${colored ? '' : 'text-gray-900 dark:text-white'}`}>{value}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
    </div>
  );
}

function PredictionColumn({
  player,
  prediction,
  whatIfResult,
  whatIfLoading,
  color = 'blue',
  latestSeason,
}: {
  player: Player;
  prediction: EOSPrediction | null;
  whatIfResult: EOSPrediction | null;
  whatIfLoading: boolean;
  color?: 'blue' | 'orange';
  latestSeason: { year: number; fantasy_points: number; games_played: number; age: number; start_position_rank: number } | null;
}) {
  const accent = color === 'orange' ? 'text-orange-500 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400';
  const borderAccent = color === 'orange' ? 'border-orange-200 dark:border-orange-800' : 'border-blue-200 dark:border-blue-800';

  return (
    <div className="flex-1 min-w-0 space-y-4">
      {/* Player header */}
      <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-sm border ${borderAccent} p-4`}>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 className={`text-xl font-bold truncate ${accent}`}>{player.name}</h2>
            <span className="inline-block px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full mt-1 text-xs font-medium">
              {player.position}
            </span>
          </div>
          <div className="text-right shrink-0">
            <div className={`text-2xl font-bold ${accent}`}>
              {formatKtc(player.live_ktc ?? prediction?.start_ktc ?? 0)}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {player.live_ktc ? 'Live KTC' : 'KTC'}
            </div>
          </div>
        </div>
      </div>

      {/* EOS Prediction */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">EOS Prediction</div>
          <div className="grid grid-cols-2 gap-2">
            <StatCard value={formatKtc(prediction.predicted_end_ktc)} label="Predicted EOS" />
            <StatCard
              value={`${prediction.predicted_pct_change >= 0 ? '+' : ''}${prediction.predicted_pct_change.toFixed(1)}%`}
              label="Change"
              colored
            />
          </div>
          <ConfidenceBand prediction={prediction} color={color} />
        </div>
      )}

      {/* What-If Result */}
      {whatIfLoading ? (
        <div className="flex justify-center py-4">
          <div className="w-5 h-5 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      ) : whatIfResult ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">What-If Result</div>
          <div className="grid grid-cols-3 gap-2">
            <StatCard value={formatKtc(whatIfResult.predicted_end_ktc)} label="EOS" />
            <StatCard
              value={`${whatIfResult.predicted_delta_ktc >= 0 ? '+' : ''}${whatIfResult.predicted_delta_ktc.toLocaleString()}`}
              label="Delta"
              colored
            />
            <StatCard
              value={`${whatIfResult.predicted_pct_change >= 0 ? '+' : ''}${whatIfResult.predicted_pct_change.toFixed(1)}%`}
              label="% Change"
              colored
            />
          </div>
          <ConfidenceBand prediction={whatIfResult} color={color} />
        </div>
      ) : null}

      {/* Season Stats */}
      {latestSeason && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
            {latestSeason.year} Season
          </div>
          <div className="grid grid-cols-2 gap-2">
            <StatCard value={latestSeason.fantasy_points.toFixed(1)} label="FP" />
            <StatCard value={String(latestSeason.games_played)} label="GP" />
            <StatCard value={String(latestSeason.age)} label="Age" />
            <StatCard value={`#${latestSeason.start_position_rank}`} label="Pos Rank" />
          </div>
        </div>
      )}
    </div>
  );
}

export default function PlayerPage() {
  const params = useParams();
  const playerId = params.id as string;
  const { selectedModelId } = useModel();

  const [player, setPlayer] = useState<Player | null>(null);
  const [prediction, setPrediction] = useState<EOSPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // What-If sliders
  const [whatIfGames, setWhatIfGames] = useState(17);
  const [whatIfPpg, setWhatIfPpg] = useState(15);
  const [whatIfResult, setWhatIfResult] = useState<EOSPrediction | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Compare player
  const [comparePlayer, setComparePlayer] = useState<PlayerSummary | null>(null);
  const [compareData, setCompareData] = useState<Player | null>(null);
  const [comparePrediction, setComparePrediction] = useState<EOSPrediction | null>(null);
  const [compareResult, setCompareResult] = useState<EOSPrediction | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const playerData = await getPlayer(playerId);
        setPlayer(playerData);

        if (playerData.seasons.length > 0) {
          const latest = playerData.seasons.reduce((a, b) => (a.year > b.year ? a : b));
          setWhatIfGames(latest.games_played);
          const ppg = latest.games_played > 0
            ? latest.fantasy_points / latest.games_played
            : 15;
          setWhatIfPpg(Math.round(ppg * 2) / 2);
        }

        const predictionData = await getPrediction(playerId, selectedModelId);
        if (predictionData) {
          setPrediction(predictionData);
        }
      } catch {
        setError('Failed to load player data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [playerId, selectedModelId]);

  // Fetch compare player
  useEffect(() => {
    if (!comparePlayer) {
      setCompareData(null);
      setComparePrediction(null);
      setCompareResult(null);
      return;
    }
    (async () => {
      try {
        const pd = await getPlayer(comparePlayer.player_id);
        setCompareData(pd);
        const pred = await getPrediction(comparePlayer.player_id, selectedModelId);
        setComparePrediction(pred);
      } catch {
        setCompareData(null);
        setComparePrediction(null);
      }
    })();
  }, [comparePlayer, selectedModelId]);

  // What-if for compare player
  useEffect(() => {
    if (!compareData || !comparePlayer) {
      setCompareResult(null);
      return;
    }
    const compareStartKtc = comparePlayer.latest_ktc ?? comparePrediction?.start_ktc ?? 5000;
    const latestCompare = compareData.seasons.length > 0
      ? compareData.seasons.reduce((a, b) => (a.year > b.year ? a : b))
      : null;

    (async () => {
      try {
        const result = await predictEos(
          {
            position: compareData.position,
            start_ktc: compareStartKtc,
            games_played: whatIfGames,
            ppg: whatIfPpg,
            age: latestCompare?.age,
          },
          selectedModelId,
        );
        setCompareResult(result);
      } catch {
        setCompareResult(null);
      }
    })();
  }, [compareData, comparePlayer, comparePrediction, whatIfGames, whatIfPpg, selectedModelId]);

  // Primary what-if
  const fetchWhatIf = useCallback(async () => {
    if (!prediction) return;
    setWhatIfLoading(true);
    try {
      const result = await predictEos(
        {
          position: prediction.position,
          start_ktc: prediction.start_ktc,
          games_played: whatIfGames,
          ppg: whatIfPpg,
        },
        selectedModelId,
      );
      setWhatIfResult(result);
    } catch {
      // failed
    } finally {
      setWhatIfLoading(false);
    }
  }, [prediction, whatIfGames, whatIfPpg, selectedModelId]);

  useEffect(() => {
    if (!prediction) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 200);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [fetchWhatIf, prediction]);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !player) {
    return (
      <div className="text-center py-12">
        <div className="text-red-500 dark:text-red-400 mb-4">{error || 'Player not found'}</div>
        <Link href="/" className="text-blue-600 dark:text-blue-400 hover:text-blue-700">Back to search</Link>
      </div>
    );
  }

  const latestSeason = player.seasons.length > 0
    ? player.seasons.reduce((a, b) => (a.year > b.year ? a : b))
    : null;

  const compareLatest = compareData?.seasons.length
    ? compareData.seasons.reduce((a, b) => (a.year > b.year ? a : b))
    : null;

  const hasCompare = !!compareData && !!comparePlayer;

  return (
    <div className="space-y-6">
      {/* Back + Compare picker row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <Link
          href="/"
          className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1 transition-colors text-sm"
        >
          &larr; Back
        </Link>
        <div className="w-full sm:w-72">
          <ComparePlayerPicker
            selected={comparePlayer}
            onSelect={setComparePlayer}
          />
        </div>
      </div>

      {/* Side-by-side player columns */}
      <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 max-w-lg'}`}>
        <PredictionColumn
          player={player}
          prediction={prediction}
          whatIfResult={whatIfResult}
          whatIfLoading={whatIfLoading}
          color="blue"
          latestSeason={latestSeason}
        />
        {hasCompare && compareData && (
          <PredictionColumn
            player={compareData}
            prediction={comparePrediction}
            whatIfResult={compareResult}
            whatIfLoading={false}
            color="orange"
            latestSeason={compareLatest}
          />
        )}
      </div>

      {/* Shared What-If controls + chart */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            What-If Scenario
          </h3>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-20">Games:</label>
              <input type="range" min="0" max="17" value={whatIfGames} onChange={(e) => setWhatIfGames(parseInt(e.target.value))} className="flex-1" />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-10 text-center">{whatIfGames}</span>
            </div>
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-20">PPG:</label>
              <input type="range" min="0" max="25" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-10 text-center">{whatIfPpg}</span>
            </div>
          </div>

          <WhatIfChart
            position={prediction.position}
            startKtc={prediction.start_ktc}
            gamesPlayed={whatIfGames}
            currentPpg={whatIfPpg}
            modelId={selectedModelId}
            compare={hasCompare && compareData && comparePlayer ? {
              name: comparePlayer.name,
              position: compareData.position,
              startKtc: comparePlayer.latest_ktc ?? prediction.start_ktc,
              age: compareLatest?.age,
            } : undefined}
          />
        </div>
      )}

      {/* Side-by-side Comps */}
      <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
        <PlayerComps playerId={playerId} />
        {hasCompare && comparePlayer && (
          <PlayerComps playerId={comparePlayer.player_id} />
        )}
      </div>

      {/* Side-by-side Historical */}
      <div className={`grid gap-6 ${hasCompare ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
        <HistoricalAccuracy playerId={playerId} />
        {hasCompare && comparePlayer && (
          <HistoricalAccuracy playerId={comparePlayer.player_id} />
        )}
      </div>
    </div>
  );
}
