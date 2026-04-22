/** @type {import('next').NextConfig} */
const isExport = process.env.NEXT_OUTPUT === 'export';

const nextConfig = {
  ...(isExport ? { output: 'export', trailingSlash: true, images: { unoptimized: true } } : {}),
  async rewrites() {
    if (isExport) return [];
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5002';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
