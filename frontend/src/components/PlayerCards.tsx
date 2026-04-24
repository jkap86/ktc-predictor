'use client';

import { memo } from 'react';
import { formatKtc } from '../lib/format';
import type { Player, EOSPrediction } from '../types/player';

export const ConfidenceBand = memo(function ConfidenceBand({ prediction, color = 'blue' }: { prediction: EOSPrediction; color?: 'blue' | 'orange' }) {
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
              width: '8px', transform: 'translateX(-50%)',
            }}
          />
        </div>
        <span className="text-gray-500 dark:text-gray-400">{formatKtc(prediction.high_end_ktc)}</span>
      </div>
    </div>
  );
});

export const PredictionStats = memo(function PredictionStats({ prediction, label, color = 'blue' }: {
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
});

export const SeasonRow = memo(function SeasonRow({ season, showRank = false }: {
  season: { year: number; fantasy_points: number; games_played: number; age: number; start_position_rank: number };
  showRank?: boolean;
}) {
  const ppg = season.games_played > 0 ? (season.fantasy_points / season.games_played).toFixed(1) : '0';
  return (
    <div className="flex items-center gap-4 text-sm">
      <span className="text-gray-500 dark:text-gray-400 w-10">{season.year}</span>
      <span className="text-gray-900 dark:text-white font-medium w-14">{ppg} ppg</span>
      <span className="text-gray-600 dark:text-gray-400">{season.games_played}gp</span>
      <span className="text-gray-600 dark:text-gray-400">age {season.age}</span>
      {showRank && <span className="text-gray-600 dark:text-gray-400">#{season.start_position_rank}</span>}
    </div>
  );
});

export const PlayerHeader = memo(function PlayerHeader({ player, prediction, color = 'blue', onRemove, onSwap }: {
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
});
