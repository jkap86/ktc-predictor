'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
  ReferenceLine,
} from 'recharts';
import type { SimulateCurveResponse } from '../types/player';

const MAX_KTC = 9999;

// Format KTC values for axis ticks (e.g., 2000 -> "2K")
function formatKtcTick(value: number): string {
  if (value >= 1000) {
    const k = value / 1000;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
  }
  return value.toString();
}

interface PPGValueCurveProps {
  data: SimulateCurveResponse;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: {
      ppg: number;
      predicted_ktc: number;
    };
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const point = payload[0].payload;

  return (
    <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
      <p className="text-sm text-gray-900 dark:text-white">
        <span className="font-medium">PPG:</span> {point.ppg}
      </p>
      <p className="text-sm text-gray-900 dark:text-white">
        <span className="font-medium">Predicted KTC:</span>{' '}
        {point.predicted_ktc.toLocaleString()}
      </p>
    </div>
  );
}

export default function PPGValueCurve({ data }: PPGValueCurveProps) {
  // Cap curve values at MAX_KTC
  const cappedCurve = data.curve.map((p) => ({
    ...p,
    predicted_ktc: Math.min(p.predicted_ktc, MAX_KTC),
  }));

  // Find the curve point closest to current PPG for the marker
  const currentPpgRounded = Math.round(data.current_ppg / 2) * 2;
  const rawPoint = cappedCurve.find((p) => p.ppg === currentPpgRounded);
  const currentPoint = rawPoint || {
    ppg: data.current_ppg,
    predicted_ktc: Math.min(data.starting_ktc, MAX_KTC),
  };

  // Calculate change at current PPG
  const changeAtCurrentPpg = currentPoint.predicted_ktc - data.starting_ktc;
  const changePercent =
    data.starting_ktc > 0
      ? ((changeAtCurrentPpg / data.starting_ktc) * 100).toFixed(1)
      : '0';

  // Find min/max for good scaling (capped at MAX_KTC)
  const ktcValues = cappedCurve.map((p) => p.predicted_ktc);
  const minKtc = Math.min(...ktcValues, data.starting_ktc);
  const maxKtc = Math.min(Math.max(...ktcValues, data.starting_ktc), MAX_KTC);
  const padding = (maxKtc - minKtc) * 0.1;

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
        KTC Value Simulator ({data.games} games)
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        See how different PPG affects predicted KTC value
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart
          data={cappedCurve}
          margin={{ top: 20, right: 30, bottom: 30, left: 20 }}
        >
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

          {/* Reference line for starting KTC */}
          <ReferenceLine
            y={data.starting_ktc}
            stroke="#9ca3af"
            strokeDasharray="5 5"
            label={{
              value: `Current: ${data.starting_ktc.toLocaleString()}`,
              position: 'right',
              fill: '#6b7280',
              fontSize: 12,
            }}
          />

          {/* Reference line for current PPG */}
          <ReferenceLine
            x={data.current_ppg}
            stroke="#9ca3af"
            strokeDasharray="5 5"
          />

          {/* Main curve */}
          <Line
            type="monotone"
            dataKey="predicted_ktc"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: '#2563eb' }}
          />

          {/* Marker for current PPG */}
          <ReferenceDot
            x={data.current_ppg}
            y={currentPoint.predicted_ktc}
            r={8}
            fill="#10b981"
            stroke="#fff"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Stats summary */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-sm text-gray-500 dark:text-gray-400">Current PPG</div>
          <div className="text-xl font-bold text-gray-900 dark:text-white">{data.current_ppg.toFixed(1)}</div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-sm text-gray-500 dark:text-gray-400">Projected KTC</div>
          <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
            {currentPoint.predicted_ktc.toLocaleString()}
          </div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-sm text-gray-500 dark:text-gray-400">Change</div>
          <div
            className={`text-xl font-bold ${
              changeAtCurrentPpg >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
            }`}
          >
            {changeAtCurrentPpg >= 0 ? '+' : ''}
            {changePercent}%
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-400 dark:text-gray-500 text-center mt-3">
        Green dot shows prediction at current PPG. Dashed line shows starting
        KTC.
      </p>
    </div>
  );
}
