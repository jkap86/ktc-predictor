'use client';

import { useEffect, useState, Suspense, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { searchPlayers, simulateCurve, getPlayer } from '../../lib/api';
import type { Player, PlayerSummary, SimulateCurveResponse } from '../../types/player';
import ComparisonChart from '../../components/ComparisonChart';
import ComparisonKTCChart from '../../components/ComparisonKTCChart';
import ComparisonHistoricalChart from '../../components/ComparisonHistoricalChart';

function CompareContent() {
  const searchParams = useSearchParams();
  const initialPlayers = searchParams.get('players')?.split(',') || [];

  const [selectedIds, setSelectedIds] = useState<string[]>(initialPlayers);
  const [curveDataList, setCurveDataList] = useState<SimulateCurveResponse[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [allPlayers, setAllPlayers] = useState<PlayerSummary[]>([]);
  const [loadingPlayers, setLoadingPlayers] = useState(true);
  const [loading, setLoading] = useState(false);
  const [games, setGames] = useState(17);

  // Fetch all players on mount (sorted by KTC descending)
  useEffect(() => {
    setLoadingPlayers(true);
    searchPlayers('', undefined, 200, 'ktc', 'desc')
      .then((result) => setAllPlayers(result.players))
      .catch(console.error)
      .finally(() => setLoadingPlayers(false));
  }, []);

  // Fetch curve data and full player data for all selected players
  const fetchData = useCallback(async () => {
    if (selectedIds.length === 0) {
      setCurveDataList([]);
      setPlayers([]);
      return;
    }

    setLoading(true);
    try {
      const [curves, playerData] = await Promise.all([
        Promise.all(selectedIds.map((id) => simulateCurve(id, games))),
        Promise.all(selectedIds.map((id) => getPlayer(id))),
      ]);
      setCurveDataList(curves);
      setPlayers(playerData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedIds, games]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Filter players based on search query (local filtering)
  const filteredPlayers = allPlayers.filter(
    (p) =>
      !selectedIds.includes(p.player_id) &&
      p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const addPlayer = (playerId: string) => {
    if (selectedIds.length < 5 && !selectedIds.includes(playerId)) {
      setSelectedIds([...selectedIds, playerId]);
      setSearchQuery('');
    }
  };

  const removePlayer = (playerId: string) => {
    setSelectedIds(selectedIds.filter((id) => id !== playerId));
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <Link
          href="/"
          className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1 transition-colors"
        >
          &larr; Back
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Compare Players</h1>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Add Players to Compare (max 5)
        </h2>

        <div className="mb-4">
          <input
            type="text"
            placeholder="Search for a player..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            disabled={selectedIds.length >= 5}
            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 shadow-sm transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 dark:disabled:bg-gray-900 disabled:cursor-not-allowed"
          />
        </div>

        {/* Player List */}
        <div className="border border-gray-200 dark:border-gray-700 rounded-xl max-h-64 overflow-y-auto mb-4">
          {loadingPlayers ? (
            <div className="flex justify-center py-8">
              <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          ) : filteredPlayers.length === 0 ? (
            <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
              {searchQuery ? 'No players found' : 'No more players available'}
            </div>
          ) : (
            filteredPlayers.map((player) => (
              <button
                key={player.player_id}
                onClick={() => addPlayer(player.player_id)}
                disabled={selectedIds.length >= 5}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 flex justify-between items-center transition-colors border-b border-gray-100 dark:border-gray-700 last:border-b-0 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="font-medium text-gray-900 dark:text-white">
                  {player.name}{' '}
                  <span className="text-gray-500 dark:text-gray-400 font-normal">
                    ({player.position})
                  </span>
                </span>
                {player.latest_ktc && (
                  <span className="text-blue-600 dark:text-blue-400 font-semibold">
                    {player.latest_ktc.toLocaleString()}
                  </span>
                )}
              </button>
            ))
          )}
        </div>

        {curveDataList.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {curveDataList.map((data) => (
              <div
                key={data.player_id}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full border border-blue-200 dark:border-blue-800"
              >
                <span className="font-medium">{data.name}</span>
                <button
                  onClick={() => removePlayer(data.player_id)}
                  className="w-5 h-5 flex items-center justify-center rounded-full hover:bg-blue-100 dark:hover:bg-blue-800/50 transition-colors"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Games Slider */}
      {selectedIds.length >= 1 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Simulate Future Value
          </h3>
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Games to simulate:
            </label>
            <input
              type="range"
              min="0"
              max="17"
              value={games}
              onChange={(e) => setGames(parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-8 text-center">
              {games}
            </span>
          </div>
        </div>
      )}

      {selectedIds.length < 1 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          Select at least 1 player to compare
        </div>
      )}

      {loading && (
        <div className="flex justify-center items-center py-12">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      )}

      {!loading && curveDataList.length >= 1 && (
        <>
          <ComparisonChart curves={curveDataList} />
          <ComparisonKTCChart players={players} />
          <ComparisonHistoricalChart players={players} />
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <div className="flex justify-center items-center py-12">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      }
    >
      <CompareContent />
    </Suspense>
  );
}
