'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getComps, CompPlayer } from '../lib/api';
import { formatKtc } from '../lib/format';

interface PlayerCompsProps {
  playerId: string;
  modelId?: string | null;
}

export default function PlayerComps({ playerId, modelId }: PlayerCompsProps) {
  const [comps, setComps] = useState<CompPlayer[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const data = await getComps(playerId, 10, modelId);
        setComps(data?.comps ?? []);
      } catch {
        setComps([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [playerId, modelId]);

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Historical Comps
        </h3>
        <div className="flex justify-center items-center py-8">
          <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (comps.length === 0) return null;

  // Summary stats
  const avgDelta = comps.reduce((sum, c) => sum + c.delta_ktc, 0) / comps.length;
  const avgPpg = comps.reduce((sum, c) => sum + c.ppg, 0) / comps.length;
  const avgStartKtc = comps.reduce((sum, c) => sum + c.start_ktc, 0) / comps.length;
  const avgEndKtc = comps.reduce((sum, c) => sum + c.end_ktc, 0) / comps.length;
  const risers = comps.filter((c) => c.delta_ktc > 0).length;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Historical Comps
        </h3>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {risers}/{comps.length} rose
        </span>
      </div>

      {/* Summary row */}
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
          avgDelta >= 0
            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
            : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
        }`}>
          Avg Change: {avgDelta >= 0 ? '+' : ''}{Math.round(avgDelta).toLocaleString()}
        </span>
        <span className="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 font-medium">
          Avg PPG: {avgPpg.toFixed(1)}
        </span>
        <span className="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 font-medium">
          Avg Start: {formatKtc(Math.round(avgStartKtc))}
        </span>
        <span className="text-xs px-2 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 font-medium">
          Avg End: {formatKtc(Math.round(avgEndKtc))}
        </span>
      </div>

      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        Similar players from prior seasons based on model features. Shows what actually happened to their value.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700">
              <th className="py-2 pr-3">Player</th>
              <th className="py-2 pr-3 text-right">Year</th>
              <th className="py-2 pr-3 text-right">Age</th>
              <th className="py-2 pr-3 text-right">GP</th>
              <th className="py-2 pr-3 text-right">PPG</th>
              <th className="py-2 pr-3 text-right">Start</th>
              <th className="py-2 pr-3 text-right">End</th>
              <th className="py-2 pr-3 text-right">Change</th>
              <th className="py-2 text-right">Sim</th>
            </tr>
          </thead>
          <tbody>
            {comps.map((c, i) => (
              <tr key={`${c.player_id}-${c.year}`} className="border-b border-gray-50 dark:border-gray-700/50">
                <td className="py-2.5 pr-3">
                  <Link
                    href={`/player/${c.player_id}`}
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium transition-colors"
                  >
                    {c.name}
                  </Link>
                </td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.year}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.age}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.games_played}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{c.ppg}</td>
                <td className="py-2.5 pr-3 text-right text-gray-600 dark:text-gray-400">{formatKtc(c.start_ktc)}</td>
                <td className="py-2.5 pr-3 text-right font-medium text-gray-900 dark:text-white">{formatKtc(c.end_ktc)}</td>
                <td className={`py-2.5 pr-3 text-right font-bold ${
                  c.delta_ktc >= 0
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-red-600 dark:text-red-400'
                }`}>
                  {c.delta_ktc >= 0 ? '+' : ''}{Math.round(c.delta_ktc).toLocaleString()}
                </td>
                <td className="py-2.5 text-right text-gray-400 dark:text-gray-500 text-xs">
                  {Math.round(c.similarity * 100)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
