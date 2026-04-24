'use client';

import { useModel } from '../context/ModelContext';

export default function ModelSelector() {
  const { models, selectedModelId, setSelectedModelId, loading } = useModel();

  // Hide temporal models (v3_for_2021, etc.) and show only main iterations
  const visibleModels = models.filter((m) => !m.id.startsWith('v3_for_') && !m.id.startsWith('v1_hgb_2021'));

  if (loading || visibleModels.length <= 1) return null;

  return (
    <select
      value={selectedModelId || ''}
      onChange={(e) => setSelectedModelId(e.target.value)}
      className="px-2 sm:px-3 py-1.5 rounded-lg bg-white/5 text-gray-300 text-xs sm:text-sm border border-white/10 hover:bg-white/10 hover:text-white transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500/40 max-w-[120px] sm:max-w-none truncate"
    >
      {visibleModels.map((model) => (
        <option key={model.id} value={model.id} className="bg-gray-800 text-white">
          {model.name}
          {model.is_default ? ' (default)' : ''}
        </option>
      ))}
    </select>
  );
}
