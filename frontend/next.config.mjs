// Two builds from one app. `next dev` is the local one: a Node server beside `langgraph dev`,
// with the /ui/* rewrite and the API passthrough. DEEP_LIFE_SCI_STATIC=1 is the one the
// deployment image makes: plain files under /app that the agent server serves itself
// (deep_life_sci/webapp.py), so /ui/* is already same-origin and there is no Node server to
// run a rewrite or a route handler. See frontend/CLAUDE.md.
const STATIC = process.env.DEEP_LIFE_SCI_STATIC === "1";
export const BASE_PATH = "/app";

/** @type {import('next').NextConfig} */
const devConfig = {
  // This app is run by end users through `uv run scripts/dev.py`, so Next's dev
  // overlay button in the bottom-left corner is chrome for a toolchain they are not being
  // asked to think about. See frontend/CLAUDE.md.
  devIndicators: false,
  // `route.dev.ts` is the API passthrough, which only a Node server can run.
  pageExtensions: ["dev.ts", "tsx", "ts", "jsx", "js"],
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

/** @type {import('next').NextConfig} */
const staticConfig = {
  output: "export",
  basePath: BASE_PATH,
  // Every page as `<route>/index.html`, which is what a static file server finds.
  trailingSlash: true,
  images: { unoptimized: true },
  pageExtensions: ["tsx", "ts", "jsx", "js"],
  // Set here rather than read from .env.local, which setup writes for `next dev` and which
  // would otherwise point this build at localhost:2024. An empty API URL means "the server
  // that served this page".
  env: {
    NEXT_PUBLIC_BASE_PATH: BASE_PATH,
    NEXT_PUBLIC_SAME_ORIGIN: "1",
    NEXT_PUBLIC_ASSISTANT_ID: "agent",
    NEXT_PUBLIC_API_URL: "",
  },
};

export default STATIC ? staticConfig : devConfig;
