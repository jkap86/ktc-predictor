'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { PlayerSeason } from '@/types/player';

interface PredictionChartProps {
  seasons: PlayerSeason[];
}

export default function PredictionChart({ seasons }: PredictionChartProps) {
  const sortedSeasons = [...seasons].sort((a, b) => a.year - b.year);

  // Build data: historical seasons only
  const data = sortedSeasons.map((s) => ({
    year: s.year,
    ktc: s.end_ktc || s.start_ktc,
  }));

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">KTC Value Over Time</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" />
          <YAxis
            domain={['auto', 'auto']}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <Tooltip
            formatter={(value: number) => [value.toLocaleString(), 'KTC']}
          />
          <Line
            type="monotone"
            dataKey="ktc"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 4, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
