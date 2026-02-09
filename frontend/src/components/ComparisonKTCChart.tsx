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
  ReferenceArea,
} from 'recharts';
import type { Player } from '../types/player';
import { formatKtc, formatKtcTick, KTC_Y_DOMAIN, KTC_Y_TICKS } from '../lib/format';
import { useChartZoom } from '../hooks/useChartZoom';

interface ComparisonKTCChartProps {
  players: Player[];
}

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

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
          {entry.dataKey}: {formatKtc(entry.value)}
        </p>
      ))}
    </div>
  );
}

export default function ComparisonKTCChart({ players }: ComparisonKTCChartProps) {
  const { zoom, handleMouseDown, handleMouseMove, handleMouseUp, resetZoom } = useChartZoom();

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

  // Filter data based on zoom
  const filteredData = zoom.isZoomed
    ? data.filter((d) => {
        const year = d.year;
        return year >= (zoom.left as number) && year <= (zoom.right as number);
      })
    : data;

  // Use fixed Y-axis domain [0, 9999] for all KTC charts

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">KTC Value Over Time</h3>
        {zoom.isZoomed && (
          <button
            onClick={resetZoom}
            className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
          >
            Reset Zoom
          </button>
        )}
      </div>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Historical KTC values by season {!zoom.isZoomed && '(drag to zoom)'}
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart
          data={filteredData}
          margin={{ top: 20, right: 30, bottom: 35, left: 25 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="year"
            tick={{ fontSize: 12 }}
            label={{ value: 'Year', position: 'bottom', offset: 0, fontSize: 12 }}
            allowDataOverflow
          />
          <YAxis
            tickFormatter={formatKtcTick}
            domain={KTC_Y_DOMAIN}
            ticks={KTC_Y_TICKS}
            tick={{ fontSize: 12 }}
            width={60}
            label={{ value: 'KTC Value', angle: -90, position: 'insideLeft', fontSize: 12, dx: -5 }}
            allowDataOverflow
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

          {zoom.refAreaLeft && zoom.refAreaRight && (
            <ReferenceArea
              x1={zoom.refAreaLeft as number}
              x2={zoom.refAreaRight as number}
              strokeOpacity={0.3}
              fill="#2563eb"
              fillOpacity={0.2}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
