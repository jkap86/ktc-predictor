'use client';

import { Fragment, useState } from 'react';
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

interface ComparisonHistoricalChartProps {
  players: Player[];
}

type ViewType = 'ktc' | 'fantasy' | 'weekly';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// Format KTC values for axis ticks (e.g., 2000 -> "2K")
function formatKtcTick(value: number): string {
  if (value >= 1000) {
    const k = value / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return value.toString();
}

export default function ComparisonHistoricalChart({ players }: ComparisonHistoricalChartProps) {
  const [view, setView] = useState<ViewType>('ktc');
  const [selectedYear, setSelectedYear] = useState<number | null>(null);

  if (players.length === 0) return null;

  // Get union of all years across all players
  const allYears = new Set<number>();
  players.forEach((p) => (p.seasons ?? []).forEach((s) => allYears.add(s.year)));
  const years = Array.from(allYears).sort((a, b) => a - b);

  // Initialize selectedYear if not set
  if (view === 'weekly' && selectedYear === null && years.length > 0) {
    setSelectedYear(years[years.length - 1]);
  }

  const getChartData = () => {
    if (view === 'weekly' && selectedYear) {
      // Weekly view: show week-by-week fantasy points for selected year
      const weeks = Array.from({ length: 18 }, (_, i) => i + 1);
      return weeks.map((week) => {
        const point: Record<string, number | undefined> = { week };
        players.forEach((p) => {
          const season = (p.seasons ?? []).find((s) => s.year === selectedYear);
          const weekStat = season?.weekly_stats.find((w) => w.week === week);
          point[p.name] = weekStat?.fantasy_points;
        });
        return point;
      });
    }

    // KTC or Fantasy view: show by year
    return years.map((year) => {
      const point: Record<string, number | undefined> = { year };
      players.forEach((p) => {
        const season = (p.seasons ?? []).find((s) => s.year === year);
        if (view === 'ktc') {
          point[`${p.name} Predicted`] = season?.start_ktc;
          point[`${p.name} Actual`] = season?.end_ktc;
        } else {
          point[p.name] = season?.fantasy_points;
        }
      });
      return point;
    });
  };

  const data = getChartData();

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <div className="flex flex-wrap justify-between items-center mb-4 gap-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Historical Performance</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setView('ktc')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 ${
              view === 'ktc'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            KTC
          </button>
          <button
            onClick={() => setView('fantasy')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 ${
              view === 'fantasy'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            Fantasy Points
          </button>
          <button
            onClick={() => {
              setView('weekly');
              if (!selectedYear && years.length > 0) {
                setSelectedYear(years[years.length - 1]);
              }
            }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 ${
              view === 'weekly'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            Weekly
          </button>
        </div>
      </div>

      {view === 'weekly' && (
        <div className="mb-4">
          <select
            value={selectedYear || ''}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
          >
            {years.map((year) => (
              <option key={year} value={year}>
                {year} Season
              </option>
            ))}
          </select>
        </div>
      )}

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data} margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey={view === 'weekly' ? 'week' : 'year'}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            tickFormatter={view === 'ktc' ? formatKtcTick : undefined}
            tick={{ fontSize: 12 }}
            width={50}
          />
          <Tooltip />
          <Legend />

          {view === 'ktc' &&
            players.map((player, idx) => (
              <Fragment key={player.player_id}>
                <Line
                  key={`${player.player_id}-predicted`}
                  type="monotone"
                  dataKey={`${player.name} Predicted`}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={{ r: 3 }}
                  connectNulls
                />
                <Line
                  key={`${player.player_id}-actual`}
                  type="monotone"
                  dataKey={`${player.name} Actual`}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 4, fill: COLORS[idx % COLORS.length], stroke: '#fff', strokeWidth: 2 }}
                  connectNulls
                />
              </Fragment>
            ))}

          {view === 'fantasy' &&
            players.map((player, idx) => (
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

          {view === 'weekly' &&
            players.map((player, idx) => (
              <Line
                key={player.player_id}
                type="monotone"
                dataKey={player.name}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls
              />
            ))}
        </LineChart>
      </ResponsiveContainer>

      {view === 'ktc' && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-3">
          Dashed lines show predicted KTC, solid lines show actual KTC
        </p>
      )}
    </div>
  );
}
