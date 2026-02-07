'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  ComposedChart,
  Scatter,
} from 'recharts';
import { predictEos } from '../lib/api';
import type { EOSPrediction, Player } from '../types/player';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
const GAMES_RANGE = Array.from({ length: 18 }, (_, i) => i); // 0-17

interface ComparisonPredictionChartProps {
  predictions: EOSPrediction[];
  players: Player[];
  whatIfPpg: number;
}

interface ChartDataPoint {
  games: number;
  [key: string]: number | undefined;
}

function formatKtcTick(value: number): string {
  if (value >= 1000) {
    const k = value / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return value.toString();
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    color: string;
    name: string;
  }>;
  label?: number;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null;

  // Filter out confidence band entries (low/high keys)
  const lines = payload.filter(
    (entry) => !entry.dataKey.endsWith('_low') && !entry.dataKey.endsWith('_high')
  );

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <p className="text-sm font-medium mb-2 text-gray-900 dark:text-white">
        Games: {label}
      </p>
      {lines.map((entry, idx) => (
        <p key={idx} className="text-sm" style={{ color: entry.color }}>
          {entry.dataKey}: {Math.round(entry.value).toLocaleString()}
        </p>
      ))}
    </div>
  );
}

export default function ComparisonPredictionChart({
  predictions,
  players,
  whatIfPpg,
}: ComparisonPredictionChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchAllPredictions = useCallback(async () => {
    if (predictions.length === 0) return;

    setLoading(true);
    try {
      // For each player, fetch predictions across all games 0-17
      const playerResults = await Promise.all(
        predictions.map(async (pred) => {
          const results = await Promise.all(
            GAMES_RANGE.map((games) =>
              predictEos({
                position: pred.position,
                start_ktc: pred.start_ktc,
                games_played: games,
                ppg: whatIfPpg,
              })
            )
          );
          return { name: pred.name || 'Unknown', results };
        })
      );

      // Build chart data: one point per games value
      const data: ChartDataPoint[] = GAMES_RANGE.map((games) => {
        const point: ChartDataPoint = { games };
        playerResults.forEach(({ name, results }) => {
          const result = results[games];
          if (result) {
            point[name] = result.predicted_end_ktc;
            if (result.low_end_ktc != null) point[`${name}_low`] = result.low_end_ktc;
            if (result.high_end_ktc != null) point[`${name}_high`] = result.high_end_ktc;
          }
        });
        return point;
      });

      setChartData(data);
    } catch (err) {
      console.error('Prediction chart fetch failed:', err);
    } finally {
      setLoading(false);
    }
  }, [predictions, whatIfPpg]);

  useEffect(() => {
    if (predictions.length === 0) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchAllPredictions, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchAllPredictions]);

  if (predictions.length === 0) return null;

  // Compute Y-axis domain from chart data
  let minKtc = Infinity;
  let maxKtc = -Infinity;
  chartData.forEach((point) => {
    predictions.forEach((pred) => {
      const name = pred.name || 'Unknown';
      const val = point[name];
      const low = point[`${name}_low`];
      const high = point[`${name}_high`];
      if (typeof val === 'number') {
        minKtc = Math.min(minKtc, val);
        maxKtc = Math.max(maxKtc, val);
      }
      if (typeof low === 'number') minKtc = Math.min(minKtc, low);
      if (typeof high === 'number') maxKtc = Math.max(maxKtc, high);
    });
  });
  if (!isFinite(minKtc)) { minKtc = 0; maxKtc = 10000; }
  const padding = (maxKtc - minKtc) * 0.1 || 500;

  // Build reference dot data for each player's baseline games_played
  const referenceDots: Array<{ games: number; ktc: number; color: string; name: string }> = [];
  predictions.forEach((pred, idx) => {
    const name = pred.name || 'Unknown';
    const player = players[idx];
    if (!player) return;
    // Use the most recent season's games_played as baseline
    const latestSeason = player.seasons?.[player.seasons.length - 1];
    if (!latestSeason) return;
    const baselineGames = latestSeason.games_played;
    const dataPoint = chartData.find((d) => d.games === baselineGames);
    if (dataPoint && typeof dataPoint[name] === 'number') {
      referenceDots.push({
        games: baselineGames,
        ktc: dataPoint[name] as number,
        color: COLORS[idx % COLORS.length],
        name,
      });
    }
  });

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        Predicted EOS KTC by Games Played
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        At {whatIfPpg} PPG — hover to compare values
      </p>

      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white/60 dark:bg-gray-800/60 z-10 flex items-center justify-center rounded-lg">
            <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
          </div>
        )}

        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="games"
              tick={{ fontSize: 12 }}
              label={{ value: 'Games Played', position: 'insideBottom', offset: -10, fontSize: 12 }}
            />
            <YAxis
              tickFormatter={formatKtcTick}
              domain={[Math.max(0, Math.floor(minKtc - padding)), Math.ceil(maxKtc + padding)]}
              tick={{ fontSize: 12 }}
              width={55}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              formatter={(value: string) =>
                value.endsWith('_low') || value.endsWith('_high') ? '' : value
              }
            />

            {/* Confidence bands as shaded areas */}
            {predictions.map((pred, idx) => {
              const name = pred.name || 'Unknown';
              const hasConfidence = chartData.some(
                (d) => d[`${name}_low`] != null && d[`${name}_high`] != null
              );
              if (!hasConfidence) return null;
              return (
                <Area
                  key={`band-${pred.player_id}`}
                  dataKey={`${name}_high`}
                  stroke="none"
                  fill={COLORS[idx % COLORS.length]}
                  fillOpacity={0.1}
                  baseValue="dataMin"
                  legendType="none"
                  tooltipType="none"
                />
              );
            })}

            {/* Main prediction lines */}
            {predictions.map((pred, idx) => (
              <Line
                key={pred.player_id || idx}
                type="monotone"
                dataKey={pred.name || 'Unknown'}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, fill: COLORS[idx % COLORS.length], stroke: '#fff', strokeWidth: 2 }}
              />
            ))}

            {/* Reference dots for baseline games played */}
            {referenceDots.map((dot, idx) => (
              <Scatter
                key={`ref-${idx}`}
                data={[{ games: dot.games, [dot.name]: dot.ktc }]}
                fill={dot.color}
                stroke="#fff"
                strokeWidth={2}
                legendType="none"
                tooltipType="none"
                shape={(props: { cx?: number; cy?: number }) => {
                  const { cx = 0, cy = 0 } = props;
                  return (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={6}
                      fill={dot.color}
                      stroke="#fff"
                      strokeWidth={2}
                    />
                  );
                }}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
