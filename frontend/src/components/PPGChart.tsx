'use client';

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts';
import type { PredictionWithPPG } from '@/types/player';

interface PPGChartProps {
  data: PredictionWithPPG[];
}

const POSITION_COLORS: Record<string, string> = {
  QB: '#2563eb', // blue
  RB: '#10b981', // green
  WR: '#f59e0b', // orange
  TE: '#8b5cf6', // purple
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: PredictionWithPPG;
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) {
    return null;
  }

  const player = payload[0].payload;
  const changeColor =
    player.ktc_change_pct >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
      <p className="font-semibold">{player.name}</p>
      <p className="text-sm text-gray-500">{player.position}</p>
      <div className="mt-2 space-y-1 text-sm">
        <p>
          PPG: <span className="font-medium">{player.ppg.toFixed(1)}</span>
        </p>
        <p>
          Predicted KTC:{' '}
          <span className="font-medium">
            {player.predicted_ktc.toLocaleString()}
          </span>
        </p>
        <p>
          Current KTC:{' '}
          <span className="font-medium">
            {player.current_ktc.toLocaleString()}
          </span>
        </p>
        <p className={changeColor}>
          Change: {player.ktc_change_pct >= 0 ? '+' : ''}
          {player.ktc_change_pct.toFixed(1)}%
        </p>
      </div>
    </div>
  );
}

export default function PPGChart({ data }: PPGChartProps) {
  // Group data by position for legend
  const positions = ['QB', 'RB', 'WR', 'TE'];

  return (
    <div className="bg-white p-6 rounded-xl shadow-soft border border-gray-100">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        PPG vs Predicted 2026 KTC
      </h3>
      <ResponsiveContainer width="100%" height={500}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="ppg"
            name="PPG"
            label={{ value: 'Points Per Game', position: 'bottom', offset: 0 }}
            domain={['auto', 'auto']}
          />
          <YAxis
            type="number"
            dataKey="predicted_ktc"
            name="Predicted KTC"
            label={{
              value: 'Predicted 2026 KTC',
              angle: -90,
              position: 'insideLeft',
            }}
            tickFormatter={(value) => value.toLocaleString()}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            payload={positions.map((pos) => ({
              value: pos,
              type: 'circle',
              color: POSITION_COLORS[pos],
            }))}
          />
          <Scatter name="Players" data={data}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={POSITION_COLORS[entry.position] || '#666'}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <p className="text-sm text-gray-500 text-center mt-2">
        Hover over points to see player details
      </p>
    </div>
  );
}
