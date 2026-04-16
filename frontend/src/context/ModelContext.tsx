'use client';

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { ModelInfo, ModelsListResponse } from '../types/player';

interface ModelContextType {
  models: ModelInfo[];
  selectedModelId: string | null;
  setSelectedModelId: (id: string) => void;
  loading: boolean;
}

const ModelContext = createContext<ModelContextType>({
  models: [],
  selectedModelId: null,
  setSelectedModelId: () => {},
  loading: true,
});

export function useModel() {
  return useContext(ModelContext);
}

const STORAGE_KEY = 'ktc-dev-selected-model';

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelIdState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error('Failed to fetch models');
        const data: ModelsListResponse = await response.json();
        setModels(data.models);

        // Restore from localStorage, or use default
        const stored = localStorage.getItem(STORAGE_KEY);
        const validIds = data.models.map((m) => m.id);
        if (stored && validIds.includes(stored)) {
          setSelectedModelIdState(stored);
        } else {
          setSelectedModelIdState(data.default_model);
        }
      } catch {
        // API not available yet
      } finally {
        setLoading(false);
      }
    }
    fetchModels();
  }, []);

  const setSelectedModelId = useCallback((id: string) => {
    setSelectedModelIdState(id);
    localStorage.setItem(STORAGE_KEY, id);
  }, []);

  return (
    <ModelContext.Provider value={{ models, selectedModelId, setSelectedModelId, loading }}>
      {children}
    </ModelContext.Provider>
  );
}
