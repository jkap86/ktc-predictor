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
  ReferenceDot,
} from 'recharts';
import type { SimulateCurveResponse } from '@/types/player';

interface ComparisonChartProps {
  curves: SimulateCurveResponse[];
}

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
const MAX_KTC = 9999;

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
      <p className="text-sm font-medium mb-2 text-gray-900 dark:text-white">PPG: {label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm" style={{ color: entry.color }}>
          {entry.dataKey}: {Math.round(entry.value).toLocaleString()}
        </p>
      ))}
    </div>
  );
}

export default function ComparisonChart({ curves }: ComparisonChartProps) {
  if (curves.length === 0) return null;

  // Transform curve data into chart format
  // Each point has ppg and a value for each player
  const ppgValues = curves[0]?.curve.map((p) => p.ppg) || [];

  const data = ppgValues.map((ppg) => {
    const point: Record<string, number> = { ppg };
    curves.forEach((curve) => {
      const curvePoint = curve.curve.find((p) => p.ppg === ppg);
      point[curve.name] = curvePoint
        ? Math.min(curvePoint.predicted_ktc, MAX_KTC)
        : 0;
    });
    return point;
  });

  // Find min/max KTC for scaling
  let minKtc = Infinity;
  let maxKtc = -Infinity;
  curves.forEach((curve) => {
    curve.curve.forEach((p) => {
      const ktc = Math.min(p.predicted_ktc, MAX_KTC);
      minKtc = Math.min(minKtc, ktc);
      maxKtc = Math.max(maxKtc, ktc);
    });
    minKtc = Math.min(minKtc, curve.starting_ktc);
    maxKtc = Math.max(maxKtc, curve.starting_ktc);
  });
  const padding = (maxKtc - minKtc) * 0.1;

  // Get current PPG positions for reference dots
  const currentPpgPositions = curves.map((curve, idx) => ({
    ppg: curve.current_ppg,
    name: curve.name,
    color: COLORS[idx % COLORS.length],
    ktc: (() => {
      const ppgRounded = Math.round(curve.current_ppg / 2) * 2;
      const point = curve.curve.find((p) => p.ppg === ppgRounded);
      return point ? Math.min(point.predicted_ktc, MAX_KTC) : curve.starting_ktc;
    })(),
  }));

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        PPG vs Predicted KTC ({curves[0]?.games || 0} games)
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Compare how different PPG affects predicted KTC value
      </p>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data} margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="ppg"
            ticks={[0, 10, 20, 30, 40]}
            domain={[0, 40]}
            tick={{ fontSize: 12 }}
            label={{ value: 'PPG', position: 'bottom', offset: 5, fontSize: 13 }}
          />
          <YAxis
            tickFormatter={formatKtcTick}
            domain={[Math.max(0, minKtc - padding), Math.min(maxKtc + padding, MAX_KTC)]}
            tick={{ fontSize: 12 }}
            width={50}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {curves.map((curve, idx) => (
            <Line
              key={curve.player_id}
              type="monotone"
              dataKey={curve.name}
              stroke={COLORS[idx % COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }}
            />
          ))}

          {/* Reference dots for current PPG positions */}
          {currentPpgPositions.map((pos, idx) => (
            <ReferenceDot
              key={`dot-${idx}`}
              x={Math.round(pos.ppg / 2) * 2}
              y={pos.ktc}
              r={8}
              fill={pos.color}
              stroke="#fff"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-3">
        Dots show each player&apos;s current PPG position
      </p>
    </div>
  );
}
