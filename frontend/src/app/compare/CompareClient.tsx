'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { searchPlayers, getPlayer, getPrediction, predictEos } from '../../lib/api';
import { formatKtc } from '../../lib/format';
import type { Player, PlayerSummary, EOSPrediction } from '../../types/player';
import ComparisonKTCChart from '../../components/ComparisonKTCChart';
import ComparisonHistoricalChart from '../../components/ComparisonHistoricalChart';
import ComparisonPredictionChart from '../../components/ComparisonPredictionChart';

export default function CompareClient() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialPlayers = searchParams.get('players')?.split(',') || [];

  const [selectedIds, setSelectedIds] = useState<string[]>(initialPlayers);
  const [predictions, setPredictions] = useState<EOSPrediction[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [allPlayers, setAllPlayers] = useState<PlayerSummary[]>([]);
  const [loadingPlayers, setLoadingPlayers] = useState(true);
  const [loading, setLoading] = useState(false);

  // What-If sliders
  const [whatIfGames, setWhatIfGames] = useState(17);
  const [whatIfResults, setWhatIfResults] = useState<EOSPrediction[]>([]);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch all players on mount (sorted by KTC descending)
  useEffect(() => {
    setLoadingPlayers(true);
    searchPlayers('', undefined, 200, 'ktc', 'desc')
      .then((result) => setAllPlayers(result.players))
      .catch(console.error)
      .finally(() => setLoadingPlayers(false));
  }, []);

  // Fetch prediction + player data for all selected players
  const fetchData = useCallback(async () => {
    if (selectedIds.length === 0) {
      setPredictions([]);
      setPlayers([]);
      return;
    }

    setLoading(true);
    try {
      // Fetch each player + prediction individually so one failure doesn't kill all
      const results = await Promise.all(
        selectedIds.map(async (id) => {
          try {
            const [pred, player] = await Promise.all([
              getPrediction(id),
              getPlayer(id),
            ]);
            if (!pred) return null;
            return { pred, player };
          } catch {
            return null;
          }
        })
      );

      const valid = results.filter(
        (r): r is { pred: EOSPrediction; player: Player } => r !== null
      );
      setPredictions(valid.map((r) => r.pred));
      setPlayers(valid.map((r) => r.player));
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  }, [selectedIds]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Sync selected players to URL params for deep linking
  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedIds.length > 0) {
      params.set('players', selectedIds.join(','));
    }
    const newUrl = selectedIds.length > 0 ? `/compare?${params.toString()}` : '/compare';
    router.replace(newUrl, { scroll: false });
  }, [selectedIds, router]);

  // Debounced what-if predictions for all selected players
  const fetchWhatIf = useCallback(async () => {
    if (predictions.length === 0) return;
    setWhatIfLoading(true);
    try {
      const results = await Promise.all(
        predictions.map((pred) =>
          predictEos({
            position: pred.position,
            start_ktc: pred.start_ktc,
            games_played: whatIfGames,
            ppg: 15, // Default PPG for what-if cards
          })
        )
      );
      // Keep aligned with predictions array (null entries skipped in render)
      setWhatIfResults(results as EOSPrediction[]);
    } catch (err) {
      console.error('What-if prediction failed:', err);
    } finally {
      setWhatIfLoading(false);
    }
  }, [predictions, whatIfGames]);

  useEffect(() => {
    if (predictions.length === 0) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchWhatIf, predictions]);

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
                    {formatKtc(player.latest_ktc)}
                  </span>
                )}
              </button>
            ))
          )}
        </div>

        {predictions.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {predictions.map((pred) => (
              <div
                key={pred.player_id}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full border border-blue-200 dark:border-blue-800"
              >
                <span className="font-medium">{pred.name}</span>
                <button
                  onClick={() => removePlayer(pred.player_id!)}
                  className="w-5 h-5 flex items-center justify-center rounded-full hover:bg-blue-100 dark:hover:bg-blue-800/50 transition-colors"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* EOS Prediction Cards Side-by-Side */}
      {!loading && predictions.length >= 1 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {predictions.map((pred) => (
            <div
              key={pred.player_id}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-gray-900 dark:text-white truncate">
                  {pred.name}
                </h3>
                <span className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full">
                  {pred.position}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Current</span>
                  <span className="font-medium text-gray-900 dark:text-white">{formatKtc(pred.start_ktc)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Predicted EOS</span>
                  <span className="font-medium text-gray-900 dark:text-white">{formatKtc(pred.predicted_end_ktc)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Change</span>
                  <span className={`font-medium ${pred.predicted_delta_ktc >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {pred.predicted_delta_ktc >= 0 ? '+' : ''}{pred.predicted_delta_ktc.toLocaleString()} ({pred.predicted_pct_change >= 0 ? '+' : ''}{pred.predicted_pct_change.toFixed(1)}%)
                  </span>
                </div>
              </div>
              {pred.anchor_year && (
                <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
                  Anchor: {pred.anchor_year} {pred.anchor_source === 'end_ktc' ? 'end' : 'start'}
                  {pred.baseline_year && <span> | Stats: {pred.baseline_year}</span>}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* What-If Scenario */}
      {selectedIds.length >= 1 && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              What-If Scenario
            </h3>
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-24">
                Games:
              </label>
              <input
                type="range"
                min="0"
                max="17"
                step="1"
                value={whatIfGames}
                onChange={(e) => setWhatIfGames(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-12 text-center">
                {whatIfGames}
              </span>
            </div>
          </div>

          {!loading && predictions.length > 0 && (
            <ComparisonPredictionChart
              predictions={predictions}
              players={players}
              whatIfGames={whatIfGames}
            />
          )}

          {whatIfLoading ? (
            <div className="flex justify-center items-center py-6">
              <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          ) : whatIfResults.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {whatIfResults.map((result, i) => {
                if (!result) return null;
                return (
                  <div key={predictions[i]?.player_id || i} className="p-3 bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
                    <div className="font-medium text-sm text-gray-900 dark:text-white mb-1 truncate">
                      {predictions[i]?.name}
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-500 dark:text-gray-400">EOS ({whatIfGames} games): </span>
                      <span className="font-medium text-gray-900 dark:text-white">{formatKtc(result.predicted_end_ktc)}</span>
                      <span className={`ml-2 ${result.predicted_delta_ktc >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        ({result.predicted_pct_change >= 0 ? '+' : ''}{result.predicted_pct_change.toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
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

      {!loading && players.length >= 1 && (
        <>
          <ComparisonKTCChart players={players} />
          <ComparisonHistoricalChart players={players} />
        </>
      )}
    </div>
  );
}
