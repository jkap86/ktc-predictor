'use client';

import { useEffect, useState } from 'react';
import { getModelMetricsByYear } from '../lib/api';
import type { MetricsByYearResponse } from '../types/player';

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState<MetricsByYearResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getModelMetricsByYear()
      .then(setMetrics)
      .catch(() => setError(true));
  }, []);

  if (error || !metrics) return null;

  return (
    <div className="bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center gap-6 text-xs overflow-x-auto">
        <span className="font-medium text-gray-600 dark:text-gray-400 whitespace-nowrap">
          Model Performance:
        </span>
        {metrics.by_year.map((y) => (
          <span
            key={y.year}
            className="text-gray-500 dark:text-gray-400 whitespace-nowrap"
          >
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {y.year}:
            </span>{' '}
            R²={y.r2.toFixed(2)} MAE={y.mae.toFixed(0)}
          </span>
        ))}
        <span className="text-gray-400 dark:text-gray-500 whitespace-nowrap">
          ({metrics.by_year.reduce((sum, y) => sum + y.n_samples, 0)} samples)
        </span>
      </div>
    </div>
  );
}
