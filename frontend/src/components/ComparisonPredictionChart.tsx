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
const PPG_RANGE = Array.from({ length: 21 }, (_, i) => i * 2); // 0, 2, 4, ... 40

interface ComparisonPredictionChartProps {
  predictions: EOSPrediction[];
  players: Player[];
  whatIfGames: number;
}

interface ChartDataPoint {
  ppg: number;
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
        PPG: {label}
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
  whatIfGames,
}: ComparisonPredictionChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchAllPredictions = useCallback(async () => {
    if (predictions.length === 0) return;

    setLoading(true);
    try {
      // For each player, fetch predictions across all PPG values 0-40
      const playerResults = await Promise.all(
        predictions.map(async (pred) => {
          const results = await Promise.all(
            PPG_RANGE.map((ppg) =>
              predictEos({
                position: pred.position,
                start_ktc: pred.start_ktc,
                games_played: whatIfGames,
                ppg: ppg,
              })
            )
          );
          return { name: pred.name || 'Unknown', results };
        })
      );

      // Build chart data: one point per PPG value
      const data: ChartDataPoint[] = PPG_RANGE.map((ppg, idx) => {
        const point: ChartDataPoint = { ppg };
        playerResults.forEach(({ name, results }) => {
          const result = results[idx];
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
  }, [predictions, whatIfGames]);

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
  const padding = (maxKtc - minKtc) * 0.15 || 500;

  // Build reference dot data for each player's baseline PPG
  const referenceDots: Array<{ ppg: number; ktc: number; color: string; name: string }> = [];
  predictions.forEach((pred, idx) => {
    const name = pred.name || 'Unknown';
    const player = players[idx];
    if (!player) return;
    // Use the most recent season to compute PPG from fantasy_points / games_played
    const latestSeason = player.seasons?.[player.seasons.length - 1];
    if (!latestSeason || !latestSeason.games_played || latestSeason.games_played === 0) return;
    const computedPpg = latestSeason.fantasy_points / latestSeason.games_played;
    // Round to nearest even number to match PPG_RANGE
    const baselinePpg = Math.round(computedPpg / 2) * 2;
    const dataPoint = chartData.find((d) => d.ppg === baselinePpg);
    if (dataPoint && typeof dataPoint[name] === 'number') {
      referenceDots.push({
        ppg: baselinePpg,
        ktc: dataPoint[name] as number,
        color: COLORS[idx % COLORS.length],
        name,
      });
    }
  });

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        Predicted EOS KTC by PPG
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        At {whatIfGames} games played — hover to compare values
      </p>

      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white/60 dark:bg-gray-800/60 z-10 flex items-center justify-center rounded-lg">
            <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
          </div>
        )}

        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, bottom: 35, left: 25 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="ppg"
              tick={{ fontSize: 12 }}
              label={{ value: 'PPG', position: 'bottom', offset: 0, fontSize: 12 }}
            />
            <YAxis
              tickFormatter={formatKtcTick}
              domain={[Math.max(0, Math.floor(minKtc - padding)), Math.ceil(maxKtc + padding)]}
              tick={{ fontSize: 12 }}
              width={60}
              label={{ value: 'Predicted EOS KTC', angle: -90, position: 'insideLeft', fontSize: 12, dx: -5 }}
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

            {/* Reference dots for baseline PPG */}
            {referenceDots.map((dot, idx) => (
              <Scatter
                key={`ref-${idx}`}
                data={[{ ppg: dot.ppg, [dot.name]: dot.ktc }]}
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
