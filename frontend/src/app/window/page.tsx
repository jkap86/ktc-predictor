'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getWindowInfo, predictWindow, searchWindowPlayers } from '../../lib/api';
import type { WindowInfo, WindowPlayer, WindowPrediction, WindowSourceValue } from '../../types/player';

const SEASON = 2026;
const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];
const WEEKS = 18;
// Sweep up to 35 points a game.
const sweepFor = (games: number) => Array.from({ length: 36 }, (_, i) => i * games);

function fmt(v?: number | null): string {
  return v == null ? '—' : Math.round(v).toLocaleString();
}

function pct(change: number, start: number): string {
  const p = (100 * change) / start;
  return `${p >= 0 ? '+' : ''}${p.toFixed(0)}%`;
}

function signed(v: number): string {
  return `${v >= 0 ? '+' : ''}${fmt(v)}`;
}

function SourceCard({ src, finished, primary }: { src: WindowSourceValue; finished: boolean; primary?: boolean }) {
  const pred = src.predictions?.[0];
  const start = src.start_value;
  return (
    <div className={`glass-card p-4 ${primary ? 'ring-1 ring-blue-500/40' : ''}`}>
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{src.label}</h3>
        {src.start_date && (
          <span className="text-xs text-gray-500 dark:text-gray-400">start value on {src.start_date}</span>
        )}
      </div>
      {pred && start ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline gap-3">
            <span className="text-gray-500 dark:text-gray-400 text-lg">{fmt(start)}</span>
            <span className="text-gray-400">{'→'}</span>
            <span className="text-3xl font-extrabold text-blue-600 dark:text-blue-400">{fmt(pred.value)}</span>
            <span className={`text-sm font-semibold ${pred.change >= 0 ? 'text-emerald-500' : 'text-red-400'}`}>
              {signed(pred.change)} ({pct(pred.change, start)})
            </span>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            80% band {fmt(pred.low)} to {fmt(pred.high)} ({pct(pred.low - start, start)} to {pct(pred.high - start, start)})
          </div>
          {src.pace && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              At their season pace ({src.pace.ppg} a game, {src.pace.points} pts): {fmt(src.pace.value)}. These
              games: <span className="font-semibold">{signed(pred.value - src.pace.value)}</span>
            </div>
          )}
          {finished && src.actual_value != null && src.actual_change != null && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Actual value after the window: {fmt(src.actual_value)} ({signed(src.actual_change)},{' '}
              {pct(src.actual_change, start)})
            </div>
          )}
          {src.trades != null && (
            <div className="text-xs text-gray-500 dark:text-gray-400">{Math.round(src.trades)} trades in the last 150 days</div>
          )}
        </div>
      ) : (
        <div className="text-sm text-gray-500 dark:text-gray-400">{src.note}</div>
      )}
    </div>
  );
}

export default function WindowPage() {
  const [info, setInfo] = useState<WindowInfo | null>(null);
  const [query, setQuery] = useState('');
  const [position, setPosition] = useState('All');
  const [players, setPlayers] = useState<WindowPlayer[]>([]);
  const [selected, setSelected] = useState<WindowPlayer | null>(null);
  const [games, setGames] = useState(3);
  const [startWeek, setStartWeek] = useState<number | null>(null);
  const [points, setPoints] = useState('');
  const [windowGames, setWindowGames] = useState('');
  const [result, setResult] = useState<WindowPrediction | null>(null);
  const [curve, setCurve] = useState<WindowPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getWindowInfo(SEASON)
      .then(setInfo)
      .catch(() => setError(`No ${SEASON} weekly inputs are loaded on the server yet.`));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => {
      searchWindowPlayers(query, position === 'All' ? undefined : position, 50, SEASON)
        .then(setPlayers)
        .catch(() => setPlayers([]));
    }, 200);
    return () => clearTimeout(t);
  }, [query, position]);

  const lastPlayed = info?.last_played_week ?? 0;
  const maxStart = WEEKS - games + 1;
  // Default: the latest finished window ("what did the last 3 games do").
  const week = Math.min(startWeek ?? Math.max(1, lastPlayed - games + 1), maxStart);
  const finished = week + games - 1 <= lastPlayed;
  const weekOptions = Array.from({ length: Math.min(maxStart, lastPlayed + 1) }, (_, i) => i + 1);

  // A finished window defaults to what happened; reset the what-if when the window moves.
  useEffect(() => {
    setPoints('');
    setWindowGames('');
  }, [selected, games, week]);

  const pts = points === '' ? null : Number(points);
  const wg = windowGames === '' ? null : Number(windowGames);
  const inputsOk =
    (pts == null ? finished : Number.isFinite(pts) && pts >= 0 && pts <= 250) &&
    (wg == null || (Number.isFinite(wg) && wg >= 0 && wg <= games));

  useEffect(() => {
    if (!selected || !inputsOk) {
      setResult(null);
      return;
    }
    let stale = false;
    const t = setTimeout(() => {
      predictWindow(selected.player_id, games, week, pts == null ? null : [pts], wg, SEASON)
        .then((r) => {
          if (stale) return;
          setResult(r);
          setError(null);
        })
        .catch(() => !stale && setError('Prediction failed.'));
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [selected, games, week, pts, wg, inputsOk]);

  const sweep = useMemo(() => sweepFor(games), [games]);
  useEffect(() => {
    if (!selected) return;
    let stale = false;
    setCurve(null);
    predictWindow(selected.player_id, games, week, sweep, wg ?? (finished ? null : games), SEASON)
      .then((r) => !stale && setCurve(r))
      .catch(() => !stale && setCurve(null));
    return () => {
      stale = true;
    };
  }, [selected, games, week, sweep, wg, finished]);

  const chartData = useMemo(() => {
    // Drop a curve left over from another window while the new one loads.
    if (!curve || curve.games !== games || curve.start_week !== week) return [];
    const rel = (v: number, s: number) => (100 * (v - s)) / s;
    const k = curve.ktc;
    const b = curve.backchannel;
    return sweep.map((p, i) => ({
      points: p,
      ktc: k.predictions && k.start_value ? rel(k.predictions[i].value, k.start_value) : null,
      ktcBand:
        k.predictions && k.start_value
          ? [rel(k.predictions[i].low, k.start_value), rel(k.predictions[i].high, k.start_value)]
          : null,
      backchannel: b.predictions && b.start_value ? rel(b.predictions[i].value, b.start_value) : null,
    }));
  }, [curve, sweep, games, week]);

  const shownPoints = pts ?? result?.actual_points ?? null;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-1 tracking-tight">
          What 3 or 6 Games Do to Value
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400/80">
          Pick a player and a stretch of games. A finished stretch shows what those games did: the predicted value from
          the points they scored, against the value at their usual pace. A stretch still to come takes your points as a
          what-if. Points are full PPR; values are dynasty superflex.
        </p>
      </div>

      {error && <div className="text-center text-sm text-red-400">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="space-y-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search players..."
            className="w-full px-4 py-3 rounded-xl border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white text-base placeholder-gray-400 dark:placeholder-gray-500 input-recessed focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
          />
          <div className="flex gap-2">
            {POSITIONS.map((p) => (
              <button
                key={p}
                onClick={() => setPosition(p)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                  position === p
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-white/10'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
          <div className="glass-card overflow-hidden">
            <div className="px-4 py-2.5 flex items-center gap-3 bg-gray-100/50 dark:bg-white/[0.03] border-b border-gray-200/50 dark:border-white/5 text-xs text-gray-500 dark:text-gray-400/80 font-semibold uppercase tracking-wider">
              <span className="flex-1 min-w-0">Player</span>
              <span className="w-7 text-center shrink-0">Pos</span>
              <span className="w-14 text-right shrink-0">KTC</span>
            </div>
            <div className="max-h-[480px] overflow-y-auto divide-y divide-gray-100/50 dark:divide-white/[0.04]">
              {players.map((pl) => (
                <button
                  key={pl.player_id}
                  onClick={() => setSelected(pl)}
                  className={`w-full px-4 py-2 flex items-center gap-3 text-left transition-colors ${
                    selected?.player_id === pl.player_id ? 'bg-blue-500/10' : 'hover:bg-gray-50 dark:hover:bg-white/[0.03]'
                  }`}
                >
                  <span className="flex-1 min-w-0 truncate text-sm text-gray-900 dark:text-white">
                    {pl.name}
                    <span className="ml-1 text-xs text-gray-400">{pl.team ?? 'FA'}</span>
                  </span>
                  <span className="text-xs text-gray-400 w-7 text-center shrink-0">{pl.position}</span>
                  <span className="text-sm font-semibold text-blue-600 dark:text-blue-400 w-14 text-right shrink-0">
                    {fmt(pl.ktc_value)}
                  </span>
                </button>
              ))}
              {players.length === 0 && (
                <div className="text-center py-8 text-sm text-gray-500 dark:text-gray-400">No players found.</div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {!selected ? (
            <div className="glass-card p-8 text-center text-gray-500 dark:text-gray-400">Pick a player.</div>
          ) : (
            <>
              <div className="glass-card p-4 flex flex-wrap items-center gap-4">
                <div className="basis-full min-w-0">
                  <div className="text-lg font-bold text-gray-900 dark:text-white truncate">{selected.name}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {selected.position} · {selected.team ?? 'FA'} · games through week {lastPlayed} are in
                  </div>
                </div>
                <div className="flex rounded-lg overflow-hidden border border-gray-200/60 dark:border-white/10">
                  {(info?.games ?? [3, 6]).map((g) => (
                    <button
                      key={g}
                      onClick={() => setGames(g)}
                      className={`px-3 py-2 text-sm ${
                        games === g ? 'bg-blue-600 text-white' : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-white/10'
                      }`}
                    >
                      {g} games
                    </button>
                  ))}
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                  Weeks
                  <select
                    value={week}
                    onChange={(e) => setStartWeek(Number(e.target.value))}
                    className="px-2 py-2 rounded-lg border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white"
                  >
                    {weekOptions.map((w) => (
                      <option key={w} value={w}>
                        {w}–{w + games - 1}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                  Points
                  <input
                    type="number"
                    min={0}
                    max={250}
                    step={1}
                    value={points}
                    placeholder={finished && result?.actual_points != null ? String(result.actual_points) : 'what-if'}
                    onChange={(e) => setPoints(e.target.value)}
                    className="w-24 px-3 py-2 rounded-lg border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white text-right focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                  Games played
                  <input
                    type="number"
                    min={0}
                    max={games}
                    step={1}
                    value={windowGames}
                    placeholder={String(finished && result ? result.window_games : games)}
                    onChange={(e) => setWindowGames(e.target.value)}
                    className="w-16 px-3 py-2 rounded-lg border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white text-right focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                </label>
              </div>

              {!finished && pts == null && (
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  Weeks {week}–{week + games - 1} haven&apos;t all been played. Type the points as a what-if.
                </div>
              )}
              {result?.note && <p className="text-xs text-amber-500">{result.note}</p>}

              {result && result.player_id === selected.player_id && (
                <>
                  {result.finished && pts == null && (
                    <div className="text-sm text-gray-600 dark:text-gray-300">
                      Weeks {result.start_week}–{result.last_game_week}: {result.actual_points} points in{' '}
                      {result.window_games} games.
                    </div>
                  )}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <SourceCard src={result.ktc} finished={result.finished} primary />
                    <SourceCard src={result.backchannel} finished={result.finished} />
                  </div>
                </>
              )}

              {chartData.length > 0 && (
                <div className="glass-card p-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                    Predicted change by points in the {games} games
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                    Percent change from each source&apos;s own start value. The shaded area is KTC&apos;s 80% band.
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                        <XAxis dataKey="points" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                        <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} stroke="#94a3b8" width={45} />
                        <Tooltip
                          formatter={(v, name) => {
                            if (name === 'ktcBand') {
                              const [lo, hi] = v as [number, number];
                              return [`${lo.toFixed(0)}% to ${hi.toFixed(0)}%`, 'KTC 80% band'];
                            }
                            return [`${Number(v).toFixed(0)}%`, name === 'ktc' ? 'KTC' : 'Backchannel'];
                          }}
                          labelFormatter={(p) => `${p} pts`}
                        />
                        <ReferenceLine y={0} stroke="#94a3b8" />
                        <Area type="monotone" dataKey="ktcBand" stroke="none" fill="#3b82f6" fillOpacity={0.12} isAnimationActive={false} />
                        {shownPoints != null && (
                          <ReferenceLine
                            x={sweep.reduce((a, b) => (Math.abs(b - shownPoints) < Math.abs(a - shownPoints) ? b : a))}
                            stroke="#3b82f6"
                            strokeDasharray="4 4"
                          />
                        )}
                        <Line type="monotone" dataKey="ktc" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="backchannel" stroke="#a3a3a3" strokeWidth={2} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
