'use client';

import { useState, useEffect, useRef } from 'react';
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
import type { PlayerSeason } from '../types/player';
import { formatKtcTick, KTC_Y_DOMAIN, KTC_Y_TICKS } from '../lib/format';
import { useChartZoom } from '../hooks/useChartZoom';
import { predictEos } from '../lib/api';

interface HistoricalChartProps {
  seasons: PlayerSeason[];
  position: string;
}

type ViewType = 'ktc' | 'fantasy' | 'weekly';

// Cache for model predictions keyed by year
type PredictionCache = Map<number, number>;

// Custom legend for KTC view showing dashed vs solid line styles
function KTCLegend({ loading }: { loading?: boolean }) {
  return (
    <div className="flex justify-center gap-6 text-sm mt-2">
      <div className="flex items-center gap-2">
        <svg width="24" height="12">
          <line x1="0" y1="6" x2="24" y2="6" stroke="#94a3b8" strokeWidth="2" strokeDasharray="5 5" />
        </svg>
        <span className="text-gray-600 dark:text-gray-400">
          Model Predicted{loading && ' (loading...)'}
        </span>
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

export default function HistoricalChart({ seasons, position }: HistoricalChartProps) {
  const [view, setView] = useState<ViewType>('ktc');
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const { zoom, handleMouseDown, handleMouseMove, handleMouseUp, resetZoom } = useChartZoom();
  const [modelPredictions, setModelPredictions] = useState<PredictionCache>(new Map());
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const fetchedRef = useRef<Set<number>>(new Set());

  const sortedSeasons = [...seasons].sort((a, b) => a.year - b.year);
  const years = sortedSeasons.map((s) => s.year);

  // Fetch model predictions for all historical seasons
  useEffect(() => {
    if (seasons.length === 0 || !position) return;

    const fetchPredictions = async () => {
      const requests: { year: number; season: PlayerSeason }[] = [];

      sortedSeasons.forEach((season) => {
        // Only fetch if we haven't already and season has valid data
        if (!fetchedRef.current.has(season.year) && season.start_ktc > 0 && season.games_played > 0) {
          requests.push({ year: season.year, season });
        }
      });

      if (requests.length === 0) return;

      setLoadingPredictions(true);
      const newPredictions = new Map(modelPredictions);

      // Fetch predictions in parallel
      const results = await Promise.all(
        requests.map(async ({ year, season }) => {
          try {
            const ppg = season.ppg ?? season.fantasy_points / Math.max(1, season.games_played);
            const pred = await predictEos({
              position,
              start_ktc: season.start_ktc,
              games_played: season.games_played,
              ppg,
              age: season.age,
            });
            return { year, value: pred?.predicted_end_ktc ?? null };
          } catch {
            return { year, value: null };
          }
        })
      );

      results.forEach(({ year, value }) => {
        fetchedRef.current.add(year);
        if (value !== null) {
          newPredictions.set(year, value);
        }
      });

      setModelPredictions(newPredictions);
      setLoadingPredictions(false);
    };

    fetchPredictions();
  }, [seasons, position]);

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

    return sortedSeasons.map((s) => {
      // Use model prediction if available, otherwise fall back to start_ktc
      const modelPred = modelPredictions.get(s.year);
      return {
        year: s.year,
        predicted_ktc: modelPred ?? s.start_ktc,
        actual_ktc: s.end_ktc,
        fantasy_points: s.fantasy_points,
        games: s.games_played,
      };
    });
  };

  const data = getChartData();

  // Filter data based on zoom
  const xKey = view === 'weekly' ? 'week' : 'year';
  const filteredData = zoom.isZoomed
    ? data.filter((d) => {
        const x = (d as Record<string, number>)[xKey];
        return x >= (zoom.left as number) && x <= (zoom.right as number);
      })
    : data;

  // Use fixed Y-axis domain [0, 9999] for KTC view

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <div className="flex flex-wrap justify-between items-center mb-4 gap-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Historical Performance</h3>
          {zoom.isZoomed && (
            <button
              onClick={resetZoom}
              className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              Reset Zoom
            </button>
          )}
        </div>
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
        <LineChart
          data={filteredData}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} allowDataOverflow />
          <YAxis
            yAxisId="left"
            orientation="left"
            tickFormatter={view === 'ktc' ? formatKtcTick : undefined}
            domain={view === 'ktc' ? KTC_Y_DOMAIN : undefined}
            ticks={view === 'ktc' ? KTC_Y_TICKS : undefined}
            allowDataOverflow
          />
          {view !== 'fantasy' && <YAxis yAxisId="right" orientation="right" allowDataOverflow />}
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

          {zoom.refAreaLeft && zoom.refAreaRight && (
            <ReferenceArea
              yAxisId="left"
              x1={zoom.refAreaLeft as number}
              x2={zoom.refAreaRight as number}
              strokeOpacity={0.3}
              fill="#2563eb"
              fillOpacity={0.2}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {view === 'ktc' && <KTCLegend loading={loadingPredictions} />}
      {!zoom.isZoomed && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-2">
          Drag to zoom
        </p>
      )}
    </div>
  );
}
