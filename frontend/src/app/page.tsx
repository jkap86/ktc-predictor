'use client';

import { useEffect, useState } from 'react';
import PlayerSearch from '@/components/PlayerSearch';
import PPGChart from '@/components/PPGChart';
import { getAllPredictions } from '@/lib/api';
import type { PredictionWithPPG } from '@/types/player';

const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];

export default function Home() {
  const [predictions, setPredictions] = useState<PredictionWithPPG[]>([]);
  const [selectedPosition, setSelectedPosition] = useState('All');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPredictions = async () => {
      setLoading(true);
      try {
        const position = selectedPosition === 'All' ? undefined : selectedPosition;
        const data = await getAllPredictions(position);
        setPredictions(data);
      } catch (error) {
        console.error('Failed to fetch predictions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [selectedPosition]);

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
          KTC Value Predictor
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Search for players to see their predicted KTC values
        </p>
      </div>
      <PlayerSearch />

      <div className="mt-12">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            2026 KTC Predictions by PPG
          </h2>
          <div className="flex gap-2">
            {POSITIONS.map((pos) => (
              <button
                key={pos}
                onClick={() => setSelectedPosition(pos)}
                className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 ${
                  selectedPosition === pos
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
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
            <div className="flex justify-center items-center py-12">
              <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          </div>
        ) : predictions.length > 0 ? (
          <PPGChart data={predictions} />
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6 text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">No prediction data available</p>
          </div>
        )}

        <p className="text-sm text-gray-500 dark:text-gray-400 mt-3">
          Showing {predictions.length} players
        </p>
      </div>
    </div>
  );
}
