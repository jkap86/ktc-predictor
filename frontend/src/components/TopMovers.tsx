'use client';

import { useEffect, useState } from 'react';
import { getTopMovers, TopMover } from '../lib/api';
import { formatKtc } from '../lib/format';

function MoverRow({ player, onSelect }: { player: TopMover; onSelect?: (id: string) => void }) {
  const isUp = player.predicted_pct_change >= 0;
  return (
    <button
      onClick={() => onSelect?.(player.player_id)}
      className="w-full flex items-center justify-between py-2.5 px-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
    >
      <div className="flex items-center gap-3">
        <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 w-8 text-center">
          {player.position}
        </span>
        <span className="text-sm font-medium text-gray-900 dark:text-white">{player.name}</span>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-xs text-gray-400 dark:text-gray-500">{formatKtc(player.start_ktc)}</span>
        <span className={`text-sm font-bold ${isUp ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
          {isUp ? '+' : ''}{player.predicted_pct_change.toFixed(1)}%
        </span>
      </div>
    </button>
  );
}

export default function TopMovers({ onSelect }: { onSelect?: (id: string) => void } = {}) {
  const [risers, setRisers] = useState<TopMover[]>([]);
  const [fallers, setFallers] = useState<TopMover[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const data = await getTopMovers(10);
      if (data) {
        setRisers(data.risers);
        setFallers(data.fallers);
      }
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (risers.length === 0 && fallers.length === 0) return null;

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {risers.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-green-600 dark:text-green-400 mb-3 px-3">
            Predicted Risers
          </h3>
          <div className="divide-y divide-gray-50 dark:divide-gray-700/50">
            {risers.map((p) => <MoverRow key={p.player_id} player={p} onSelect={onSelect} />)}
          </div>
        </div>
      )}
      {fallers.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-red-600 dark:text-red-400 mb-3 px-3">
            Predicted Fallers
          </h3>
          <div className="divide-y divide-gray-50 dark:divide-gray-700/50">
            {fallers.map((p) => <MoverRow key={p.player_id} player={p} onSelect={onSelect} />)}
          </div>
        </div>
      )}
    </div>
  );
}
