import { Suspense } from 'react';
import CompareClient from './CompareClient';

export default function ComparePage() {
  return (
    <Suspense
      fallback={
        <div className="flex justify-center items-center py-12">
          <div className="w-8 h-8 border-2 border-gray-200 dark:border-gray-600 border-t-blue-600 rounded-full animate-spin" />
        </div>
      }
    >
      <CompareClient />
    </Suspense>
  );
}
