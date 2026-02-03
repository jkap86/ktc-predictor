'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { searchPlayers } from '@/lib/api';
import type { PlayerSummary } from '@/types/player';

const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];

export default function PlayerSearch() {
  const [query, setQuery] = useState('');
  const [position, setPosition] = useState('All');
  const [players, setPlayers] = useState<PlayerSummary[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPlayers = async () => {
      setLoading(true);
      try {
        const pos = position === 'All' ? undefined : position;
        // Always sort by KTC value (highest first)
        const result = await searchPlayers(query, pos, 50, 'ktc', 'desc');
        setPlayers(result.players);
      } catch (error) {
        console.error('Failed to fetch players:', error);
      } finally {
        setLoading(false);
      }
    };

    const debounce = setTimeout(fetchPlayers, 300);
    return () => clearTimeout(debounce);
  }, [query, position]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-4">
        <input
          type="text"
          placeholder="Search players..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 shadow-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <div className="flex gap-2">
          {POSITIONS.map((pos) => (
            <button
              key={pos}
              onClick={() => setPosition(pos)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 ${
                position === pos
                  ? 'bg-blue-600 text-white shadow-sm hover:bg-blue-700'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              {pos}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-12">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {players.map((player) => (
            <Link
              key={player.player_id}
              href={`/player/${player.player_id}`}
              className="block p-4 bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 hover:shadow-soft-lg hover:border-gray-200 dark:hover:border-gray-600 hover:-translate-y-0.5 transition-all duration-200"
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-lg text-gray-900 dark:text-white">
                    {player.name}
                  </h3>
                  <span className="inline-block px-2 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded mt-1">
                    {player.position}
                  </span>
                </div>
                {player.latest_ktc && (
                  <div className="text-right">
                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                      {player.latest_ktc.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">KTC Value</div>
                  </div>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}

      {!loading && players.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          No players found. Try a different search.
        </div>
      )}
    </div>
  );
}
