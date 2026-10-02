import { Client } from "@langchain/langgraph-sdk";
import { withAuthHeader } from "@/lib/auth";

export function createClient(apiUrl: string, authScheme: string | undefined) {
  return new Client({
    apiUrl,
    // Never an API key from the environment: the browser holds none (frontend/CLAUDE.md).
    apiKey: null,
    // Read per request, so a token renewed mid-session is the one sent, and added only for
    // a request to this page's own origin (lib/auth.ts).
    onRequest: async (url, init) => ({
      ...init,
      headers: await withAuthHeader(url, new Headers(init.headers)),
    }),
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
  });
}
