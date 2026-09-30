import { Client } from "@langchain/langgraph-sdk";
import { withAuthHeader } from "@/lib/auth";

export function createClient(apiUrl: string, authScheme: string | undefined) {
  return new Client({
    apiUrl,
    // Never an API key from the environment: the browser holds none (frontend/CLAUDE.md).
    apiKey: null,
    // Read per request, so a token renewed mid-session is the one sent.
    onRequest: (_url, init) => ({
      ...init,
      headers: withAuthHeader(new Headers(init.headers)),
    }),
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
  });
}
