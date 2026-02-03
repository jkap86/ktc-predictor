'use client';

import { useState } from 'react';
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
import type { PlayerSeason } from '@/types/player';

interface HistoricalChartProps {
  seasons: PlayerSeason[];
}

type ViewType = 'ktc' | 'fantasy' | 'weekly';

// Custom legend for KTC view showing dashed vs solid line styles
function KTCLegend() {
  return (
    <div className="flex justify-center gap-6 text-sm mt-2">
      <div className="flex items-center gap-2">
        <svg width="24" height="12">
          <line x1="0" y1="6" x2="24" y2="6" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 5" />
        </svg>
        <span className="text-gray-600 dark:text-gray-400">Predicted KTC</span>
      </div>
      <div className="flex items-center gap-2">
        <svg width="24" height="12">
          <line x1="0" y1="6" x2="24" y2="6" stroke="#2563eb" strokeWidth="2" />
        </svg>
        <span className="text-gray-600 dark:text-gray-400">Actual KTC</span>
      </div>
    </div>
  );
}

export default function HistoricalChart({ seasons }: HistoricalChartProps) {
  const [view, setView] = useState<ViewType>('ktc');
  const [selectedYear, setSelectedYear] = useState<number | null>(null);

  const sortedSeasons = [...seasons].sort((a, b) => a.year - b.year);
  const years = sortedSeasons.map((s) => s.year);

  const getChartData = () => {
    if (view === 'weekly' && selectedYear) {
      const season = sortedSeasons.find((s) => s.year === selectedYear);
      if (!season) return [];

      return season.weekly_stats.map((ws, idx) => ({
        week: ws.week,
        fantasy_points: ws.fantasy_points,
        ktc: season.weekly_ktc[idx]?.ktc || 0,
      }));
    }

    return sortedSeasons.map((s) => ({
      year: s.year,
      predicted_ktc: s.start_ktc,  // Start of season = market prediction
      actual_ktc: s.end_ktc,       // End of season = actual result
      fantasy_points: s.fantasy_points,
      games: s.games_played,
    }));
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

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={view === 'weekly' ? 'week' : 'year'} />
          <YAxis yAxisId="left" orientation="left" />
          {view !== 'fantasy' && <YAxis yAxisId="right" orientation="right" />}
          <Tooltip />
          {view !== 'ktc' && <Legend />}

          {view === 'ktc' && (
            <>
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="predicted_ktc"
                name="Predicted KTC"
                stroke="#94a3b8"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 3 }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="actual_ktc"
                name="Actual KTC"
                stroke="#2563eb"
                strokeWidth={2}
                dot={{ r: 4, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
              />
            </>
          )}

          {view === 'fantasy' && (
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="fantasy_points"
              name="Fantasy Points"
              stroke="#10b981"
              strokeWidth={2}
            />
          )}

          {view === 'weekly' && (
            <>
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="fantasy_points"
                name="Fantasy Points"
                stroke="#10b981"
                strokeWidth={2}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="ktc"
                name="KTC"
                stroke="#2563eb"
                strokeWidth={2}
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>

      {view === 'ktc' && <KTCLegend />}
    </div>
  );
}
