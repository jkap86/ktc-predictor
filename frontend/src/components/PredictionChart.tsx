'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts';
import type { PlayerSeason } from '../types/player';
import { formatKtc, formatKtcTick, clampKtc, generateKtcTicks } from '../lib/format';
import { useChartZoom } from '../hooks/useChartZoom';

interface PredictionChartProps {
  seasons: PlayerSeason[];
}

export default function PredictionChart({ seasons }: PredictionChartProps) {
  const { zoom, handleMouseDown, handleMouseMove, handleMouseUp, resetZoom } = useChartZoom();
  const sortedSeasons = [...seasons].sort((a, b) => a.year - b.year);

  // Build data: historical seasons only, clamping KTC values
  const data = sortedSeasons.map((s) => ({
    year: s.year,
    ktc: clampKtc(s.end_ktc || s.start_ktc),
  }));

  // Filter data based on zoom
  const filteredData = zoom.isZoomed
    ? data.filter((d) => {
        const year = d.year;
        return year >= (zoom.left as number) && year <= (zoom.right as number);
      })
    : data;

  // Calculate Y domain for better ticks
  let minKtc = Infinity;
  let maxKtc = -Infinity;
  const dataToAnalyze = zoom.isZoomed ? filteredData : data;
  dataToAnalyze.forEach((d) => {
    if (d.ktc > 0) {
      minKtc = Math.min(minKtc, d.ktc);
      maxKtc = Math.max(maxKtc, d.ktc);
    }
  });
  const yPadding = isFinite(minKtc) ? (maxKtc - minKtc) * 0.15 : 500;
  const yMin = isFinite(minKtc) ? Math.max(0, minKtc - yPadding) : 0;
  const yMax = isFinite(maxKtc) ? maxKtc + yPadding : 10000;
  const yTicks = isFinite(minKtc) ? generateKtcTicks(yMin, yMax) : undefined;

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <div className="flex justify-between items-center mb-4">
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
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={filteredData}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" allowDataOverflow />
          <YAxis
            domain={[yMin, yMax]}
            ticks={yTicks}
            tickFormatter={formatKtcTick}
            allowDataOverflow
          />
          <Tooltip
            formatter={(value: number) => [formatKtc(value), 'KTC']}
          />
          <Line
            type="monotone"
            dataKey="ktc"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 4, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }}
          />

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
      {!zoom.isZoomed && (
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-2">
          Drag to zoom
        </p>
      )}
    </div>
  );
}
