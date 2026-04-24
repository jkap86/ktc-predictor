'use client';

import { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { getHistorical } from '../lib/api';
import { formatKtc } from '../lib/format';
import type { HistoricalPrediction } from '../types/player';

interface HistoricalAccuracyProps {
  playerId: string;
}

interface ChartRow {
  year: string;
  start_ktc: number;
  actual_end_ktc: number | null;
  predicted_end_ktc: number;
  error: number | null;
  games_played: number;
  ppg: number;
  low_end_ktc: number | null;
  high_end_ktc: number | null;
}

export default function HistoricalAccuracy({ playerId }: HistoricalAccuracyProps) {
  const [predictions, setPredictions] = useState<HistoricalPrediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const data = await getHistorical(playerId);
        setPredictions(data?.predictions ?? []);
      } catch {
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [playerId]);

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-4">
          Historical Accuracy
        </h3>
        <div className="flex justify-center items-center py-8">
          <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (predictions.length === 0) return null;

  const chartData: ChartRow[] = predictions.map((p) => ({
    year: String(p.year),
    start_ktc: p.start_ktc,
    actual_end_ktc: p.actual_end_ktc,
    predicted_end_ktc: p.predicted_end_ktc,
    error: p.error,
    games_played: p.games_played,
    ppg: p.ppg,
    low_end_ktc: p.low_end_ktc,
    high_end_ktc: p.high_end_ktc,
  }));

  const mae =
    predictions.filter((p) => p.error !== null).length > 0
      ? Math.round(
          predictions
            .filter((p) => p.error !== null)
            .reduce((sum, p) => sum + Math.abs(p.error!), 0) /
            predictions.filter((p) => p.error !== null).length
        )
      : null;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
          Historical Accuracy
        </h3>
        {mae !== null && (
          <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-full">
            MAE: {formatKtc(mae)}
          </span>
        )}
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        Each year&apos;s prediction uses a model trained only on prior seasons — what the model would have predicted before that season started.
      </p>

      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
          <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#9ca3af' }} />
          <YAxis
            tick={{ fontSize: 12, fill: '#9ca3af' }}
            tickFormatter={(v: number) => formatKtc(v)}
            width={55}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(31,41,55,0.95)',
              border: 'none',
              borderRadius: '8px',
              fontSize: 13,
            }}
            labelStyle={{ color: '#d1d5db', fontWeight: 600 }}
            formatter={(value, name) => {
              const labels: Record<string, string> = {
                actual_end_ktc: 'Actual EOS',
                predicted_end_ktc: 'Predicted EOS',
                start_ktc: 'Start KTC',
              };
              return [formatKtc(Number(value)), labels[String(name)] ?? String(name)];
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
            formatter={(value) => {
              const labels: Record<string, string> = {
                actual_end_ktc: 'Actual EOS',
                predicted_end_ktc: 'Predicted EOS',
                start_ktc: 'Start KTC',
              };
              return labels[value] ?? value;
            }}
          />

          {/* Start KTC as faint bars for context */}
          <Bar
            dataKey="start_ktc"
            fill="rgba(156,163,175,0.15)"
            name="start_ktc"
            isAnimationActive={false}
          />

          {/* Actual EOS line */}
          <Line
            dataKey="actual_end_ktc"
            stroke="#10b981"
            strokeWidth={2.5}
            dot={{ r: 5, fill: '#10b981', stroke: 'white', strokeWidth: 2 }}
            isAnimationActive={false}
            name="actual_end_ktc"
            connectNulls
          />

          {/* Predicted EOS line */}
          <Line
            dataKey="predicted_end_ktc"
            stroke="#3b82f6"
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={{ r: 5, fill: '#3b82f6', stroke: 'white', strokeWidth: 2 }}
            isAnimationActive={false}
            name="predicted_end_ktc"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Table view */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700">
              <th className="py-2 pr-3">Year</th>
              <th className="py-2 pr-3 text-right">Start</th>
              <th className="py-2 pr-3 text-right">Predicted</th>
              <th className="py-2 pr-3 text-right">Actual</th>
              <th className="py-2 pr-3 text-right">Error</th>
              <th className="py-2 pr-3 text-right">GP</th>
              <th className="py-2 text-right">PPG</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => (
              <tr key={p.year} className="border-b border-gray-50 dark:border-gray-700/50">
                <td className="py-2 pr-3 font-medium text-gray-900 dark:text-white">{p.year}</td>
                <td className="py-2 pr-3 text-right text-gray-600 dark:text-gray-400">{formatKtc(p.start_ktc)}</td>
                <td className="py-2 pr-3 text-right text-blue-600 dark:text-blue-400 font-medium">{formatKtc(p.predicted_end_ktc)}</td>
                <td className="py-2 pr-3 text-right text-green-600 dark:text-green-400 font-medium">
                  {p.actual_end_ktc !== null ? formatKtc(p.actual_end_ktc) : '—'}
                </td>
                <td className={`py-2 pr-3 text-right font-medium ${
                  p.error === null ? '' :
                  Math.abs(p.error) < 300 ? 'text-green-600 dark:text-green-400' :
                  Math.abs(p.error) < 700 ? 'text-yellow-600 dark:text-yellow-400' :
                  'text-red-600 dark:text-red-400'
                }`}>
                  {p.error !== null ? `${p.error > 0 ? '+' : ''}${formatKtc(p.error)}` : '—'}
                </td>
                <td className="py-2 pr-3 text-right text-gray-600 dark:text-gray-400">{p.games_played}</td>
                <td className="py-2 text-right text-gray-600 dark:text-gray-400">{p.ppg}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
