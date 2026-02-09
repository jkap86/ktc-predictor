'use client';

import { Fragment, useState, useEffect, useRef } from 'react';
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
import { formatKtcTick, clampKtc, generateKtcTicks } from '../lib/format';
import { useChartZoom } from '../hooks/useChartZoom';
import { predictEos } from '../lib/api';

interface ComparisonHistoricalChartProps {
  players: Player[];
}

type ViewType = 'ktc' | 'fantasy' | 'weekly';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// Cache for model predictions keyed by player_id + year
type PredictionCache = Map<string, number>;

export default function ComparisonHistoricalChart({ players }: ComparisonHistoricalChartProps) {
  const [view, setView] = useState<ViewType>('ktc');
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const { zoom, handleMouseDown, handleMouseMove, handleMouseUp, resetZoom } = useChartZoom();
  const [modelPredictions, setModelPredictions] = useState<PredictionCache>(new Map());
  const [loadingPredictions, setLoadingPredictions] = useState(false);
  const fetchedRef = useRef<Set<string>>(new Set());

  // Fetch model predictions for all historical seasons
  useEffect(() => {
    if (players.length === 0) return;

    const fetchPredictions = async () => {
      const requests: { key: string; player: Player; season: (typeof players)[0]['seasons'][0] }[] = [];

      players.forEach((player) => {
        (player.seasons ?? []).forEach((season) => {
          const key = `${player.player_id}-${season.year}`;
          // Only fetch if we haven't already
          if (!fetchedRef.current.has(key) && season.start_ktc > 0 && season.games_played > 0) {
            requests.push({ key, player, season });
          }
        });
      });

      if (requests.length === 0) return;

      setLoadingPredictions(true);
      const newPredictions = new Map(modelPredictions);

      // Batch fetch predictions (limit concurrency to avoid rate limits)
      const batchSize = 10;
      for (let i = 0; i < requests.length; i += batchSize) {
        const batch = requests.slice(i, i + batchSize);
        const results = await Promise.all(
          batch.map(async ({ key, player, season }) => {
            try {
              const pred = await predictEos({
                position: player.position,
                start_ktc: season.start_ktc,
                games_played: season.games_played,
                ppg: season.ppg ?? season.fantasy_points / Math.max(1, season.games_played),
                age: season.age,
              });
              return { key, value: pred?.predicted_end_ktc ?? null };
            } catch {
              return { key, value: null };
            }
          })
        );

        results.forEach(({ key, value }) => {
          fetchedRef.current.add(key);
          if (value !== null) {
            newPredictions.set(key, value);
          }
        });
      }

      setModelPredictions(newPredictions);
      setLoadingPredictions(false);
    };

    fetchPredictions();
  }, [players]);

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
          // Use model prediction if available, otherwise fall back to start_ktc
          const predKey = `${p.player_id}-${year}`;
          const modelPred = modelPredictions.get(predKey);
          point[`${p.name} Predicted`] = modelPred ?? season?.start_ktc;
          point[`${p.name} Actual`] = season?.end_ktc;
        } else {
          point[p.name] = season?.fantasy_points;
        }
      });
      return point;
    });
  };

  const data = getChartData();

  // Filter data based on zoom
  const xKey = view === 'weekly' ? 'week' : 'year';
  const filteredData = zoom.isZoomed
    ? data.filter((d) => {
        const x = d[xKey] as number;
        return x >= (zoom.left as number) && x <= (zoom.right as number);
      })
    : data;

  // Helper for dynamic Y-axis label
  const getYAxisLabel = () => {
    if (view === 'ktc') return 'KTC Value';
    if (view === 'fantasy') return 'Fantasy Points';
    return 'Fantasy Points';
  };

  // Calculate Y domain for KTC view for better zoom
  let minY = Infinity;
  let maxY = -Infinity;
  if (view === 'ktc') {
    const dataToAnalyze = zoom.isZoomed ? filteredData : data;
    players.forEach((p) => {
      dataToAnalyze.forEach((d) => {
        const predicted = d[`${p.name} Predicted`];
        const actual = d[`${p.name} Actual`];
        if (predicted && predicted > 0) {
          const clamped = clampKtc(predicted as number);
          minY = Math.min(minY, clamped);
          maxY = Math.max(maxY, clamped);
        }
        if (actual && actual > 0) {
          const clamped = clampKtc(actual as number);
          minY = Math.min(minY, clamped);
          maxY = Math.max(maxY, clamped);
        }
      });
    });
  }
  const yPadding = isFinite(minY) ? (maxY - minY) * 0.15 : 0;
  const yMin = isFinite(minY) ? Math.max(0, minY - yPadding) : 0;
  const yMax = isFinite(maxY) ? maxY + yPadding : 10000;
  const yDomain: [number, number] | undefined = isFinite(minY)
    ? [yMin, yMax]
    : undefined;
  const yTicks = view === 'ktc' && isFinite(minY) ? generateKtcTicks(yMin, yMax) : undefined;

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
            dataKey={xKey}
            tick={{ fontSize: 12 }}
            label={{ value: view === 'weekly' ? 'Week' : 'Year', position: 'bottom', offset: 0, fontSize: 12 }}
            allowDataOverflow
          />
          <YAxis
            tickFormatter={view === 'ktc' ? formatKtcTick : undefined}
            tick={{ fontSize: 12 }}
            width={60}
            domain={view === 'ktc' ? yDomain : undefined}
            ticks={yTicks}
            label={{ value: getYAxisLabel(), angle: -90, position: 'insideLeft', fontSize: 12, dx: -5 }}
            allowDataOverflow
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

      {view === 'ktc' && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-3">
          Dashed lines show model-predicted EOS KTC, solid lines show actual EOS KTC
          {loadingPredictions && ' (loading predictions...)'}
          {!zoom.isZoomed && ' • Drag to zoom'}
        </p>
      )}
    </div>
  );
}
