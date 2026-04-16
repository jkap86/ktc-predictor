'use client';

import { useModel } from '../context/ModelContext';

export default function ModelSelector() {
  const { models, selectedModelId, setSelectedModelId, loading } = useModel();

  if (loading || models.length <= 1) return null;

  return (
    <select
      value={selectedModelId || ''}
      onChange={(e) => setSelectedModelId(e.target.value)}
      className="px-3 py-1.5 rounded-lg bg-white/10 text-white text-sm border border-white/20 hover:bg-white/20 transition-colors cursor-pointer focus:outline-none focus:ring-2 focus:ring-white/30"
    >
      {models.map((model) => (
        <option key={model.id} value={model.id} className="bg-gray-800 text-white">
          {model.name}
          {model.is_default ? ' (default)' : ''}
        </option>
      ))}
    </select>
  );
}
