'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getSeasonInfo, predictSeason, searchSeasonPlayers } from '../../lib/api';
import type { SeasonInfo, SeasonPlayer, SeasonPrediction, SeasonSourceValue } from '../../types/player';

const SEASON = 2026;
const POSITIONS = ['All', 'QB', 'RB', 'WR', 'TE'];
const SWEEP = Array.from({ length: 46 }, (_, i) => i * 10); // 0..450 pts
const DEFAULT_POINTS = 250;

function fmt(v?: number | null): string {
  return v == null ? '—' : Math.round(v).toLocaleString();
}

function pct(change: number, start: number): string {
  const p = (100 * change) / start;
  return `${p >= 0 ? '+' : ''}${p.toFixed(0)}%`;
}

function SourceCard({
  label,
  src,
  primary,
}: {
  label: string;
  src: SeasonSourceValue;
  primary?: boolean;
}) {
  const pred = src.predictions?.[0];
  return (
    <div className={`glass-card p-4 ${primary ? 'ring-1 ring-blue-500/40' : ''}`}>
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{label}</h3>
        {src.start_date && (
          <span className="text-xs text-gray-500 dark:text-gray-400">start value on {src.start_date}</span>
        )}
      </div>
      {pred && src.start_value ? (
        <>
          <div className="flex items-baseline gap-3">
            <span className="text-gray-500 dark:text-gray-400 text-lg">{fmt(src.start_value)}</span>
            <span className="text-gray-400">{'→'}</span>
            <span className="text-3xl font-extrabold text-blue-600 dark:text-blue-400">{fmt(pred.value)}</span>
            <span
              className={`text-sm font-semibold ${pred.change >= 0 ? 'text-emerald-500' : 'text-red-400'}`}
            >
              {pred.change >= 0 ? '+' : ''}
              {fmt(pred.change)} ({pct(pred.change, src.start_value)})
            </span>
          </div>
          {src.trades != null && (
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {Math.round(src.trades)} trades in the start window
            </div>
          )}
        </>
      ) : (
        <div className="text-sm text-gray-500 dark:text-gray-400">{src.note}</div>
      )}
      {pred && src.note && <p className="text-xs text-amber-500 mt-2">{src.note}</p>}
    </div>
  );
}

export default function SeasonPage() {
  const [info, setInfo] = useState<SeasonInfo | null>(null);
  const [query, setQuery] = useState('');
  const [position, setPosition] = useState('All');
  const [players, setPlayers] = useState<SeasonPlayer[]>([]);
  const [selected, setSelected] = useState<SeasonPlayer | null>(null);
  const [points, setPoints] = useState<string>(String(DEFAULT_POINTS));
  const [result, setResult] = useState<SeasonPrediction | null>(null);
  const [curve, setCurve] = useState<SeasonPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    getSeasonInfo(SEASON)
      .then(setInfo)
      .catch(() => setError(`No ${SEASON} start values are loaded on the server yet.`));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => {
      searchSeasonPlayers(query, position === 'All' ? undefined : position, 50, SEASON)
        .then(setPlayers)
        .catch(() => setPlayers([]));
    }, 200);
    return () => clearTimeout(t);
  }, [query, position]);

  // The full points curve, once per player.
  useEffect(() => {
    if (!selected) return;
    let stale = false;
    setCurve(null);
    predictSeason(selected.player_id, SWEEP, SEASON)
      .then((r) => !stale && setCurve(r))
      .catch(() => !stale && setCurve(null));
    return () => {
      stale = true;
    };
  }, [selected]);

  // The typed points total.
  useEffect(() => {
    if (!selected) return;
    if (debounce.current) clearTimeout(debounce.current);
    const pts = Number(points);
    if (points === '' || !Number.isFinite(pts) || pts < 0 || pts > 600) {
      setResult(null);
      return;
    }
    // Responses from a superseded player or points total are dropped.
    let stale = false;
    debounce.current = setTimeout(() => {
      setLoading(true);
      predictSeason(selected.player_id, [pts], SEASON)
        .then((r) => {
          if (stale) return;
          setResult(r);
          setError(null);
        })
        .catch(() => !stale && setError('Prediction failed.'))
        .finally(() => !stale && setLoading(false));
    }, 250);
    return () => {
      stale = true;
      setLoading(false);
    };
  }, [selected, points]);

  const chartData = useMemo(() => {
    if (!curve) return [];
    const bc = curve.backchannel;
    const ktc = curve.ktc;
    return SWEEP.map((p, i) => ({
      points: p,
      backchannel:
        bc.predictions && bc.start_value ? (100 * bc.predictions[i].change) / bc.start_value : null,
      ktc: ktc.predictions && ktc.start_value ? (100 * ktc.predictions[i].change) / ktc.start_value : null,
    }));
  }, [curve]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-1 tracking-tight">
          {SEASON} Season: Points to Value
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400/80">
          Type a player&apos;s projected {SEASON} fantasy points (full PPR) to see their predicted end-of-season
          dynasty superflex value, starting from their value on {info?.cutoff ?? 'the day before kickoff'}.
        </p>
      </div>

      {error && <div className="text-center text-sm text-red-400">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        {/* Player picker */}
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
              <span className="w-8 text-right shrink-0">Age</span>
              <span className="w-14 text-right shrink-0">BC</span>
              <span className="w-14 text-right shrink-0">KTC</span>
            </div>
            <div className="max-h-[480px] overflow-y-auto divide-y divide-gray-100/50 dark:divide-white/[0.04]">
              {players.map((pl) => (
                <button
                  key={pl.player_id}
                  onClick={() => setSelected(pl)}
                  className={`w-full px-4 py-2 flex items-center gap-3 text-left transition-colors ${
                    selected?.player_id === pl.player_id
                      ? 'bg-blue-500/10'
                      : 'hover:bg-gray-50 dark:hover:bg-white/[0.03]'
                  }`}
                >
                  <span className="flex-1 min-w-0 truncate text-sm text-gray-900 dark:text-white">
                    {pl.name}
                    <span className="ml-1 text-xs text-gray-400">{pl.team ?? 'FA'}</span>
                  </span>
                  <span className="text-xs text-gray-400 w-7 text-center shrink-0">{pl.position}</span>
                  <span className="text-xs text-gray-400 w-8 text-right shrink-0">{pl.age ?? '—'}</span>
                  <span className="text-sm font-semibold text-blue-600 dark:text-blue-400 w-14 text-right shrink-0">
                    {fmt(pl.backchannel.start_value)}
                  </span>
                  <span className="text-xs text-gray-500 w-14 text-right shrink-0">{fmt(pl.ktc.start_value)}</span>
                </button>
              ))}
              {players.length === 0 && (
                <div className="text-center py-8 text-sm text-gray-500 dark:text-gray-400">No players found.</div>
              )}
            </div>
          </div>
        </div>

        {/* Prediction */}
        <div className="space-y-4">
          {!selected ? (
            <div className="glass-card p-8 text-center text-gray-500 dark:text-gray-400">Pick a player.</div>
          ) : (
            <>
              <div className="glass-card p-4 flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-0">
                  <div className="text-lg font-bold text-gray-900 dark:text-white truncate">{selected.name}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {selected.position} · {selected.team ?? 'FA'} · age {selected.age ?? '?'} on {info?.cutoff}
                  </div>
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                  Projected {SEASON} points
                  <input
                    type="number"
                    min={0}
                    max={600}
                    step={5}
                    value={points}
                    onChange={(e) => setPoints(e.target.value)}
                    className="w-24 px-3 py-2 rounded-lg border border-gray-200/60 dark:border-white/10 bg-white/80 dark:bg-white/5 text-gray-900 dark:text-white text-right focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                </label>
                {loading && (
                  <span className="w-4 h-4 border-2 border-gray-300 dark:border-gray-600 border-t-blue-500 rounded-full animate-spin" />
                )}
              </div>

              {result && result.player_id === selected.player_id && (
                <div className="grid gap-4 sm:grid-cols-2">
                  <SourceCard label="Backchannel (trade-derived)" src={result.backchannel} primary />
                  <SourceCard label="KTC" src={result.ktc} />
                </div>
              )}

              {chartData.length > 0 && (
                <div className="glass-card p-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                    Predicted change by season points
                  </h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                    Percent change from each source&apos;s own start value, since the two scales differ.
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                        <XAxis dataKey="points" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                        <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} stroke="#94a3b8" width={45} />
                        <Tooltip
                          formatter={(v, name) => [`${Number(v).toFixed(0)}%`, name === 'ktc' ? 'KTC' : 'Backchannel']}
                          labelFormatter={(p) => `${p} pts`}
                        />
                        <ReferenceLine y={0} stroke="#94a3b8" />
                        {Number.isFinite(Number(points)) && points !== '' && (
                          <ReferenceLine x={Math.round(Number(points) / 10) * 10} stroke="#3b82f6" strokeDasharray="4 4" />
                        )}
                        <Line type="monotone" dataKey="backchannel" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="ktc" stroke="#a3a3a3" strokeWidth={2} dot={false} />
                      </LineChart>
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
