import PlayerPageClient from './PlayerPageClient';

// Required for Next.js static export with dynamic routes.
// Returns a placeholder so the route template is generated;
// actual player ID comes from useParams at runtime.
export function generateStaticParams() {
  return [{ id: '_' }];
}

export const dynamicParams = true;

export default function PlayerPage() {
  return <PlayerPageClient />;
}
