'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { getPlayer, getPrediction, predictEos } from '../../../lib/api';
import { formatKtc } from '../../../lib/format';
import type { Player, EOSPrediction } from '../../../types/player';

// Lazy-load charts so initial render is fast
const PredictionChart = dynamic(() => import('../../../components/PredictionChart'), {
  loading: () => <div className="h-64 flex items-center justify-center"><div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" /></div>,
});
const HistoricalChart = dynamic(() => import('../../../components/HistoricalChart'), {
  loading: () => <div className="h-64 flex items-center justify-center"><div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" /></div>,
});

function ConfidenceBand({ prediction }: { prediction: EOSPrediction }) {
  if (!prediction.low_end_ktc || !prediction.high_end_ktc) return null;
  return (
    <div className="mt-3 px-4 py-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800">
      <div className="text-xs font-medium text-blue-700 dark:text-blue-300 mb-1">
        Confidence Range (p20 – p80)
      </div>
      <div className="flex items-center gap-3 text-sm">
        <span className="text-gray-600 dark:text-gray-400">{formatKtc(prediction.low_end_ktc)}</span>
        <div className="flex-1 h-2 bg-blue-100 dark:bg-blue-800 rounded-full relative">
          <div
            className="absolute h-2 bg-blue-500 dark:bg-blue-400 rounded-full"
            style={{
              left: `${Math.max(0, Math.min(100, ((prediction.predicted_end_ktc - prediction.low_end_ktc) / (prediction.high_end_ktc - prediction.low_end_ktc)) * 100))}%`,
              width: '4px',
              transform: 'translateX(-50%)',
            }}
          />
        </div>
        <span className="text-gray-600 dark:text-gray-400">{formatKtc(prediction.high_end_ktc)}</span>
      </div>
    </div>
  );
}

export default function PlayerPage() {
  const params = useParams();
  const playerId = params.id as string;

  const [player, setPlayer] = useState<Player | null>(null);
  const [prediction, setPrediction] = useState<EOSPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // What-If sliders
  const [whatIfGames, setWhatIfGames] = useState(17);
  const [whatIfPpg, setWhatIfPpg] = useState(15);
  const [whatIfResult, setWhatIfResult] = useState<EOSPrediction | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Advanced inputs
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advAge, setAdvAge] = useState<number | undefined>(undefined);
  const [advDraftPick, setAdvDraftPick] = useState<number | undefined>(undefined);
  const [advYearsLeft, setAdvYearsLeft] = useState<number | undefined>(undefined);
  const [advWeeksMissed, setAdvWeeksMissed] = useState<number | undefined>(undefined);

  // Charts collapsed by default
  const [chartsOpen, setChartsOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const playerData = await getPlayer(playerId);
        setPlayer(playerData);

        // Initialize what-if sliders from latest season
        if (playerData.seasons.length > 0) {
          const latest = playerData.seasons.reduce((a, b) => (a.year > b.year ? a : b));
          setWhatIfGames(latest.games_played);
          const ppg = latest.games_played > 0
            ? latest.fantasy_points / latest.games_played
            : 15;
          setWhatIfPpg(Math.round(ppg * 2) / 2);
          setAdvAge(latest.age);
        }

        // Prediction may return null if player has no valid seasons
        // Use weekly blend for better early/mid-season accuracy
        const predictionData = await getPrediction(playerId, true);
        if (predictionData) {
          setPrediction(predictionData);
        }
      } catch (err) {
        setError('Failed to load player data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [playerId]);

  // Debounced what-if prediction
  const fetchWhatIf = useCallback(async () => {
    if (!prediction) return;
    setWhatIfLoading(true);
    try {
      const result = await predictEos({
        position: prediction.position,
        start_ktc: prediction.start_ktc,
        games_played: whatIfGames,
        ppg: whatIfPpg,
        age: advAge,
        draft_pick: advDraftPick,
        years_remaining: advYearsLeft,
        weeks_missed: advWeeksMissed,
      });
      setWhatIfResult(result);
    } catch (err) {
      console.error('What-if prediction failed:', err);
    } finally {
      setWhatIfLoading(false);
    }
  }, [prediction, whatIfGames, whatIfPpg, advAge, advDraftPick, advYearsLeft, advWeeksMissed]);

  useEffect(() => {
    if (!prediction) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchWhatIf, 200);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchWhatIf, prediction]);

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

      {/* EOS Prediction Card */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              End-of-Season Prediction
            </h3>
            <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full font-mono">
              {prediction.model_version}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatKtc(prediction.start_ktc)}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Current KTC</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatKtc(prediction.predicted_end_ktc)}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Predicted EOS</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className={`text-2xl font-bold ${prediction.predicted_delta_ktc >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                {prediction.predicted_delta_ktc >= 0 ? '+' : ''}{prediction.predicted_delta_ktc.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">Delta</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className={`text-2xl font-bold ${prediction.predicted_pct_change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                {prediction.predicted_pct_change >= 0 ? '+' : ''}{prediction.predicted_pct_change.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">% Change</div>
            </div>
          </div>
          <ConfidenceBand prediction={prediction} />
          {(prediction.anchor_year || prediction.baseline_year) && (
            <div className="mt-2 text-xs text-gray-400 dark:text-gray-500 px-1">
              {prediction.anchor_year && (
                <span>
                  Anchor: {formatKtc(prediction.start_ktc)} from {prediction.anchor_year}{' '}
                  {prediction.anchor_source === 'end_ktc' ? 'end' : 'start'}
                </span>
              )}
              {prediction.anchor_year && prediction.baseline_year && (
                <span className="mx-1">|</span>
              )}
              {prediction.baseline_year && (
                <span>Baseline stats: {prediction.baseline_year} season</span>
              )}
            </div>
          )}
        </div>
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

      {/* What-If Sliders */}
      {prediction && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            What-If Scenario
          </h3>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-24">Games:</label>
              <input type="range" min="0" max="17" value={whatIfGames} onChange={(e) => setWhatIfGames(parseInt(e.target.value))} className="flex-1" />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-12 text-center">{whatIfGames}</span>
            </div>
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-24">PPG:</label>
              <input type="range" min="0" max="40" step="0.5" value={whatIfPpg} onChange={(e) => setWhatIfPpg(parseFloat(e.target.value))} className="flex-1" />
              <span className="text-lg font-bold text-blue-600 dark:text-blue-400 w-12 text-center">{whatIfPpg}</span>
            </div>
          </div>

          {/* Advanced Inputs Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="mt-3 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1"
          >
            <svg className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            Advanced inputs
          </button>

          {showAdvanced && (
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Age</label>
                <input
                  type="number" min="18" max="45" step="1"
                  value={advAge ?? ''}
                  onChange={(e) => setAdvAge(e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                  placeholder="—"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Draft Pick</label>
                <input
                  type="number" min="1" max="260" step="1"
                  value={advDraftPick ?? ''}
                  onChange={(e) => setAdvDraftPick(e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                  placeholder="—"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Contract Yrs</label>
                <input
                  type="number" min="0" max="6" step="1"
                  value={advYearsLeft ?? ''}
                  onChange={(e) => setAdvYearsLeft(e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                  placeholder="—"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Weeks Missed</label>
                <input
                  type="number" min="0" max="17" step="1"
                  value={advWeeksMissed ?? ''}
                  onChange={(e) => setAdvWeeksMissed(e.target.value ? Number(e.target.value) : undefined)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                  placeholder="—"
                />
              </div>
            </div>
          )}

          {/* What-If Result */}
          {whatIfLoading ? (
            <div className="flex justify-center items-center py-6">
              <div className="w-6 h-6 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
            </div>
          ) : whatIfResult ? (
            <div className="mt-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {formatKtc(whatIfResult.predicted_end_ktc)}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Predicted EOS</div>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className={`text-xl font-bold ${whatIfResult.predicted_delta_ktc >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {whatIfResult.predicted_delta_ktc >= 0 ? '+' : ''}{whatIfResult.predicted_delta_ktc.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Delta</div>
                </div>
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className={`text-xl font-bold ${whatIfResult.predicted_pct_change >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {whatIfResult.predicted_pct_change >= 0 ? '+' : ''}{whatIfResult.predicted_pct_change.toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">% Change</div>
                </div>
              </div>
              <ConfidenceBand prediction={whatIfResult} />
            </div>
          ) : null}
        </div>
      )}

      {/* Collapsible Charts */}
      {player.seasons.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-soft border border-gray-100 dark:border-gray-700">
          <button
            onClick={() => setChartsOpen(!chartsOpen)}
            className="w-full px-6 py-4 flex items-center justify-between text-left"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Historical Charts
            </h3>
            <svg
              className={`w-5 h-5 text-gray-500 transition-transform ${chartsOpen ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {chartsOpen && (
            <div className="px-6 pb-6 space-y-6">
              <PredictionChart seasons={player.seasons} />
              <HistoricalChart seasons={player.seasons} />
            </div>
          )}
        </div>
      )}

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
