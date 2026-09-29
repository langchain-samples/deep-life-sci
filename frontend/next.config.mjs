/** @type {import('next').NextConfig} */
const nextConfig = {
  // This app is run by end users through `uv run scripts/dev.py`, so Next's dev
  // overlay button in the bottom-left corner is chrome for a toolchain they are not being
  // asked to think about. See frontend/CLAUDE.md.
  devIndicators: false,
  // Artifact components load /ui/* from the page origin, so this proxy is
  // what makes them render at all. See frontend/CLAUDE.md.
  async rewrites() {
    return [
      { source: "/ui/:path*", destination: `${process.env.LANGGRAPH_API_URL || "http://localhost:2024"}/ui/:path*` },
    ];
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
