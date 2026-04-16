'use client';

import PlayerSearch from '../components/PlayerSearch';

export default function Home() {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
          KTC Value Predictor
        </h1>
        <p className="text-gray-600 dark:text-gray-300">
          Search for players to see their predicted end-of-season KTC values
        </p>
      </div>
      <PlayerSearch />
    </div>
  );
}
