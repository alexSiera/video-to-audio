/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Монорепо: транслируем пакеты-источники (workspace) через Next.
  transpilePackages: ["@itr/ui", "@itr/shared"],
  experimental: {
    // Оптимизация импортов в монорепо.
    optimizePackageImports: ["lucide-react"],
  },
};

export default nextConfig;
