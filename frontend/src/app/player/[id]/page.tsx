import PlayerPageClient from './PlayerPageClient';

// Required for Next.js static export — dynamic routes render client-side
export function generateStaticParams() {
  return [];
}

export default function PlayerPage() {
  return <PlayerPageClient />;
}
