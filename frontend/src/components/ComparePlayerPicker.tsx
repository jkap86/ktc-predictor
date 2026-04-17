'use client';

import { useState, useEffect, useRef } from 'react';
import { searchPlayers } from '../lib/api';
import type { PlayerSummary } from '../types/player';

interface ComparePlayerPickerProps {
  /** Position to optionally filter by. If omitted, shows all positions. */
  position?: string;
  onSelect: (player: PlayerSummary | null) => void;
  selected: PlayerSummary | null;
}

export default function ComparePlayerPicker({
  position,
  onSelect,
  selected,
}: ComparePlayerPickerProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<PlayerSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const data = await searchPlayers(query, position || undefined, 8, 'ktc', 'desc');
        setResults(data.players);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [query, position]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  if (selected) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/20">
        <span className="text-sm font-medium text-orange-700 dark:text-orange-300">
          vs {selected.name}
        </span>
        <span className="text-xs text-orange-500 dark:text-orange-400">
          {selected.position}
        </span>
        <button
          onClick={() => onSelect(null)}
          className="ml-auto text-orange-400 hover:text-orange-600 dark:hover:text-orange-200 transition-colors"
          title="Remove comparison"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <input
        type="text"
        placeholder="Compare with player..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => { if (query.trim()) setOpen(true); }}
        className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-orange-400 focus:border-transparent"
      />
      {open && (results.length > 0 || loading) && (
        <div className="absolute z-20 mt-1 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-48 overflow-y-auto">
          {loading ? (
            <div className="px-3 py-2 text-sm text-gray-400">Searching...</div>
          ) : (
            results.map((p) => (
              <button
                key={p.player_id}
                onClick={() => {
                  onSelect(p);
                  setQuery('');
                  setOpen(false);
                }}
                className="w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center justify-between"
              >
                <span className="text-sm text-gray-900 dark:text-white">{p.name}</span>
                <span className="text-xs text-gray-400">{p.position}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
