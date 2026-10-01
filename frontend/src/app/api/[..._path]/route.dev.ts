import { initApiPassthrough } from "langgraph-nextjs-api-passthrough";
import { NextRequest, NextResponse } from "next/server";

// The API passthrough `dev.py --remote` uses: it forwards /api/* to LANGGRAPH_API_URL and
// adds LANGSMITH_API_KEY to every request, so the browser never holds the key. Whoever can
// send this route a request can use that key, so only this dev server's own page may. The
// server listens on 127.0.0.1 only (package.json), which keeps other machines out, and a
// request from any other origin is refused here: another site open in the same browser, or
// a name rebound to 127.0.0.1. No CORS headers go out either, where upstream's handler
// answers every origin with `*`. See frontend/CLAUDE.md.
const passthrough = initApiPassthrough({
  apiUrl: process.env.LANGGRAPH_API_URL ?? "remove-me",
  apiKey: process.env.LANGSMITH_API_KEY ?? "remove-me",
  disableWarningLog: true,
});

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]"]);

/** Whether the request comes from a page this server served, as the browser tells it. */
function fromOwnPage(req: NextRequest): boolean {
  const host = req.headers.get("host") ?? "";
  if (!LOOPBACK.has(host.replace(/:\d+$/, ""))) return false;
  const site = req.headers.get("sec-fetch-site");
  if (site && site !== "same-origin" && site !== "none") return false;
  const origin = req.headers.get("origin");
  return !origin || origin === `http://${host}`;
}

type Handler = (req: NextRequest) => Promise<Response>;

function ownPageOnly(handler: Handler): Handler {
  return async (req) => {
    if (!fromOwnPage(req)) {
      return new NextResponse("This proxy serves only the chat UI on this machine.", {
        status: 403,
      });
    }
    const res = await handler(req);
    for (const name of Array.from(res.headers.keys())) {
      if (name.toLowerCase().startsWith("access-control-")) res.headers.delete(name);
    }
    return res;
  };
}

export const GET = ownPageOnly(passthrough.GET);
export const POST = ownPageOnly(passthrough.POST);
export const PUT = ownPageOnly(passthrough.PUT);
export const PATCH = ownPageOnly(passthrough.PATCH);
export const DELETE = ownPageOnly(passthrough.DELETE);

// Only a cross-origin request is preceded by a preflight, and none is let through.
export function OPTIONS() {
  return new NextResponse(null, { status: 403 });
}
