'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { Player } from '../types/player';

interface ComparisonKTCChartProps {
  players: Player[];
}

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// Format KTC values for axis ticks (e.g., 2000 -> "2K")
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
  }>;
  label?: number;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <p className="text-sm font-medium mb-2 text-gray-900 dark:text-white">Year: {label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm" style={{ color: entry.color }}>
          {entry.dataKey}: {Math.round(entry.value).toLocaleString()}
        </p>
      ))}
    </div>
  );
}

export default function ComparisonKTCChart({ players }: ComparisonKTCChartProps) {
  if (players.length === 0) return null;

  // Get union of all years across all players
  const allYears = new Set<number>();
  players.forEach((p) => (p.seasons ?? []).forEach((s) => allYears.add(s.year)));
  const years = Array.from(allYears).sort((a, b) => a - b);

  // Build data points: { year, PlayerA: ktc, PlayerB: ktc, ... }
  const data = years.map((year) => {
    const point: Record<string, number> = { year };
    players.forEach((p) => {
      const season = (p.seasons ?? []).find((s) => s.year === year);
      if (season) {
        point[p.name] = season.end_ktc || season.start_ktc;
      }
    });
    return point;
  });

  // Find min/max KTC for scaling
  let minKtc = Infinity;
  let maxKtc = -Infinity;
  players.forEach((p) => {
    (p.seasons ?? []).forEach((s) => {
      const ktc = s.end_ktc || s.start_ktc;
      if (ktc > 0) {
        minKtc = Math.min(minKtc, ktc);
        maxKtc = Math.max(maxKtc, ktc);
      }
    });
  });
  const padding = (maxKtc - minKtc) * 0.15 || 500;

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">KTC Value Over Time</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Historical KTC values by season
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data} margin={{ top: 20, right: 30, bottom: 35, left: 25 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="year"
            tick={{ fontSize: 12 }}
            label={{ value: 'Year', position: 'bottom', offset: 0, fontSize: 12 }}
          />
          <YAxis
            tickFormatter={formatKtcTick}
            domain={[Math.max(0, minKtc - padding), maxKtc + padding]}
            tick={{ fontSize: 12 }}
            width={60}
            label={{ value: 'KTC Value', angle: -90, position: 'insideLeft', fontSize: 12, dx: -5 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {players.map((player, idx) => (
            <Line
              key={player.player_id}
              type="monotone"
              dataKey={player.name}
              stroke={COLORS[idx % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 4, fill: COLORS[idx % COLORS.length], stroke: '#fff', strokeWidth: 2 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
