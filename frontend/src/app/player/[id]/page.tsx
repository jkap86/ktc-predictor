'use client';

import { useEffect, useState, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getPlayer, simulateCurve } from '../../../lib/api';
import type { Player, SimulateCurveResponse } from '../../../types/player';
import PredictionChart from '../../../components/PredictionChart';
import HistoricalChart from '../../../components/HistoricalChart';
import PPGValueCurve from '../../../components/PPGValueCurve';

export default function PlayerPage() {
  const params = useParams();
  const playerId = params.id as string;

  const [player, setPlayer] = useState<Player | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Simulator state
  const [games, setGames] = useState(17);
  const [curveData, setCurveData] = useState<SimulateCurveResponse | null>(null);
  const [simulatorLoading, setSimulatorLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const playerData = await getPlayer(playerId);
        setPlayer(playerData);
      } catch (err) {
        setError('Failed to load player data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [playerId]);

  // Fetch curve data when games changes
  const fetchCurveData = useCallback(async () => {
    if (!playerId) return;
    try {
      setSimulatorLoading(true);
      const data = await simulateCurve(playerId, games);
      setCurveData(data);
    } catch (err) {
      console.error('Failed to load simulation data:', err);
    } finally {
      setSimulatorLoading(false);
    }
  }, [playerId, games]);

  useEffect(() => {
    // Only fetch curve data after initial player data is loaded
    if (player && !loading) {
      fetchCurveData();
    }
  }, [player, loading, fetchCurveData]);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
      </div>
    );
  }

  if (error || !player) {
    return (
      <div className="text-center py-12">
        <div className="text-red-500 dark:text-red-400 mb-4">{error || 'Player not found'}</div>
        <Link
          href="/"
          className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
        >
          Back to search
        </Link>
      </div>
    );
  }

  const latestSeason =
    player.seasons.length > 0
      ? player.seasons.reduce((a, b) => (a.year > b.year ? a : b))
      : null;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <Link
          href="/"
          className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1 transition-colors"
        >
          &larr; Back
        </Link>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{player.name}</h1>
        <span className="inline-block px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full mt-2 font-medium">
          {player.position}
        </span>
      </div>

      {player.seasons.length > 0 && (
        <>
          <PredictionChart seasons={player.seasons} />
          <HistoricalChart seasons={player.seasons} />
        </>
      )}

      {latestSeason && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Latest Season Stats ({latestSeason.year})
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {latestSeason.fantasy_points.toFixed(1)}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Fantasy Points</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {latestSeason.games_played}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Games Played</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {latestSeason.age}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Age</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                #{latestSeason.start_position_rank}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Position Rank</div>
            </div>
          </div>
        </div>
      )}

      {/* KTC Value Simulator */}
      <div className="space-y-4">
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

        {simulatorLoading ? (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
            <div className="flex justify-center items-center py-12">
              <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          </div>
        ) : curveData ? (
          <PPGValueCurve data={curveData} />
        ) : null}
      </div>

      <div className="text-center">
        <Link
          href={`/compare?players=${playerId}`}
          className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg font-medium shadow-sm hover:bg-blue-700 hover:shadow transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
        >
          Compare with other players
        </Link>
      </div>
    </div>
  );
}
