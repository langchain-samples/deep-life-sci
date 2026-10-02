// Sign-in for a deployed app, through the institution's OpenID Connect provider.
//
// Only a build served by the agent server itself signs in: it sets NEXT_PUBLIC_SAME_ORIGIN,
// and the server answers `<basePath>/config.json` with the provider's settings at run time,
// so one build works with any provider. `next dev` has neither and runs without sign-in,
// against a local server that has no auth either.
//
// Tokens are kept in memory only (InMemoryWebStorage), never in localStorage or
// sessionStorage: a page reload signs in again, which with a live provider session is one
// redirect and no prompt. The server verifies what is sent (deep_life_sci/auth.py).
//
// The token goes to the agent server that served this page and nowhere else. That build
// also takes no server URL from the query string (`pinnedConnection`), where a link could
// otherwise point the app, and the signed-in user's token, at a server of its own.

import {
  InMemoryWebStorage,
  User,
  UserManager,
  WebStorageStateStore,
} from "oidc-client-ts";

export type AuthConfig = {
  auth: "oidc";
  issuer: string;
  clientId: string;
  scope: string;
  // Which token the server expects: the ID token (its audience is the client ID) or an
  // access token issued for an API.
  token: "id" | "access";
};

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const SAME_ORIGIN = process.env.NEXT_PUBLIC_SAME_ORIGIN === "1";
export const CALLBACK_PATH = `${BASE_PATH}/auth/callback`;

// Renew this long before the token requests carry expires. The server tolerates a minute
// of clock skew past `exp`; renewing a minute before it keeps a request in flight inside.
const RENEW_BEFORE_SECONDS = 60;
// After a renewal fails, one made on demand by a request waits this long before the next.
const RETRY_AFTER_MS = 30_000;
// Browsers fire a longer setTimeout at once (the delay overflows a signed 32-bit int).
const MAX_TIMER_MS = 2 ** 31 - 1;
// Set when the user signs out, so a reload or a new tab does not sign them straight back in
// through a provider session that is still alive. A flag, not a token.
const SIGNED_OUT_KEY = "deep-life-sci:signed-out";

let configPromise: Promise<AuthConfig | null> | null = null;
let manager: UserManager | null = null;
let current: User | null = null;
let tokenKind: AuthConfig["token"] = "id";
let renewal: Promise<User | null> | null = null;
let renewTimer: ReturnType<typeof setTimeout> | undefined;
let lastFailure = 0;
let ended = false;
const sessionListeners = new Set<(ended: boolean) => void>();

/** The server's sign-in settings, or null in a build that does not sign in.
 *
 * A build that does sign in fails closed: when the settings cannot be read it rejects, and
 * the page says so, rather than running unsigned-in with every request refused. A failure
 * is not cached, so the next call tries again. */
export function loadAuthConfig(): Promise<AuthConfig | null> {
  if (!SAME_ORIGIN) return Promise.resolve(null);
  configPromise ??= fetchAuthConfig().catch((error: Error) => {
    configPromise = null;
    throw error;
  });
  return configPromise;
}

async function fetchAuthConfig(): Promise<AuthConfig> {
  let problem = "no answer";
  // A server mid-rollout can answer 502/503 for a moment; a 4xx will not fix itself.
  for (const wait of [0, 1000, 3000]) {
    if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
    let res: Response;
    try {
      res = await fetch(`${BASE_PATH}/config.json`, { cache: "no-store" });
    } catch (error) {
      problem = (error as Error).message;
      continue;
    }
    if (res.ok) {
      const body = await res.json().catch(() => null);
      if (body?.auth === "oidc" && body.issuer && body.clientId) return body as AuthConfig;
      problem = "the server sent no sign-in settings";
      break;
    }
    problem = `HTTP ${res.status}`;
    if (res.status < 500) break;
  }
  throw new Error(`Could not load the sign-in settings (${problem}). Reload to try again.`);
}

/** In a build the agent server serves: that server and its graph, whatever the query string
 * says. Null in a build that lets the user choose (`next dev`). */
export function pinnedConnection(): { apiUrl: string; assistantId: string } | null {
  if (!SAME_ORIGIN || typeof window === "undefined") return null;
  return {
    apiUrl: window.location.origin,
    assistantId: process.env.NEXT_PUBLIC_ASSISTANT_ID || "agent",
  };
}

export function userManager(config: AuthConfig): UserManager {
  if (manager) return manager;
  const origin = window.location.origin;
  tokenKind = config.token;
  manager = new UserManager({
    authority: config.issuer,
    client_id: config.clientId,
    redirect_uri: `${origin}${CALLBACK_PATH}`,
    silent_redirect_uri: `${origin}${CALLBACK_PATH}`,
    post_logout_redirect_uri: `${origin}${BASE_PATH}/`,
    response_type: "code",
    scope: config.scope,
    // Renewal is scheduled here instead, off the token requests actually carry. The
    // library's own timers follow the access token's lifetime, and in the default `id`
    // mode it is the ID token that is sent, which can expire first (Entra: an hour,
    // against 60-90 minutes).
    automaticSilentRenew: false,
    userStore: new WebStorageStateStore({ store: new InMemoryWebStorage() }),
  });
  manager.events.addUserLoaded(setUser);
  manager.events.addUserUnloaded(() => {
    current = null;
    clearTimeout(renewTimer);
  });
  return manager;
}

export function setUser(user: User | null) {
  current = user;
  if (user && secondsLeft(user) > 0) setEnded(false);
  scheduleRenewal();
}

/** When the token requests carry stops being accepted, in epoch seconds. */
function expiresAt(user: User): number | undefined {
  if (tokenKind === "access") return user.expires_at;
  const exp = user.profile?.exp;
  return typeof exp === "number" ? exp : user.expires_at;
}

/** Seconds until the token requests carry expires: Infinity if it says no expiry. */
export function secondsLeft(user: User): number {
  const at = expiresAt(user);
  return at === undefined ? Infinity : at - Date.now() / 1000;
}

/** When to renew a token with `left` seconds to go: a minute before it expires, or halfway
 * there for one that lives less than two, so a short-lived token cannot renew in a loop. */
function renewIn(left: number): number {
  return Math.max(left - RENEW_BEFORE_SECONDS, left / 2) * 1000;
}

function scheduleRenewal() {
  clearTimeout(renewTimer);
  if (!current) return;
  const delay = renewIn(secondsLeft(current));
  if (Number.isFinite(delay)) renewAt(Date.now() + Math.max(delay, 0));
}

/** Renew at `deadline` (epoch ms). A deadline past the longest timer is reached in steps. */
function renewAt(deadline: number) {
  clearTimeout(renewTimer);
  renewTimer = setTimeout(
    () => (Date.now() < deadline - 1000 ? renewAt(deadline) : void renew()),
    Math.min(deadline - Date.now(), MAX_TIMER_MS),
  );
}

/** Renew silently, one renewal at a time: through the refresh token when the provider
 * issued one, otherwise through a hidden iframe. Resolves to the renewed user, or null.
 *
 * When a renewal fails, the session goes on with the token it has until that expires; then
 * it has ended, and the page offers to sign in again. It never navigates away by itself:
 * that would lose the composer's draft. A run in progress survives it (runs continue when
 * their page goes, and the page rejoins them). */
function renew(): Promise<User | null> {
  if (!manager) return Promise.resolve(null);
  const userManager = manager;
  renewal ??= (async () => {
    try {
      const before = current?.id_token;
      let user = await userManager.signinSilent();
      // A refresh response without a new ID token keeps the old one, which is the token
      // that is expiring; the iframe asks the provider for a fresh one.
      if (user && tokenKind === "id" && user.id_token === before) {
        user = await userManager.signinSilent({ forceIframeAuth: true });
      }
      if (!user || secondsLeft(user) <= 0) {
        throw new Error("the provider returned no fresh token");
      }
      return user;
    } catch {
      lastFailure = Date.now();
      const user = current;
      if (!user || secondsLeft(user) <= 0) setEnded(true);
      // Try once more as it expires; if that fails too, the session has ended.
      else renewAt(Date.now() + secondsLeft(user) * 1000);
      return null;
    } finally {
      renewal = null;
    }
  })();
  return renewal;
}

function setEnded(value: boolean) {
  if (ended === value) return;
  ended = value;
  for (const listener of sessionListeners) listener(value);
}

/** Told `true` when the session ends (renewal failed and the token expired), `false` when a
 * later renewal brings it back. Returns the unsubscribe. */
export function onSessionEnded(listener: (ended: boolean) => void): () => void {
  sessionListeners.add(listener);
  return () => sessionListeners.delete(listener);
}

/** The token requests carry, renewed first when it has expired; null before sign-in, in a
 * build without it, and once the session has ended. */
export async function authToken(): Promise<string | null> {
  let user = current;
  if (!user) return null;
  if (secondsLeft(user) <= 0) {
    // A timer can fire late (a laptop that slept), so a request renews on demand too.
    if (Date.now() - lastFailure < RETRY_AFTER_MS) return null;
    user = await renew();
    if (!user) return null;
  }
  return (tokenKind === "access" ? user.access_token : user.id_token) ?? null;
}

/** Whether `url` is on the origin this page was served from. */
function sameOrigin(url: string | URL): boolean {
  if (typeof window === "undefined") return false;
  try {
    return new URL(url, window.location.href).origin === window.location.origin;
  } catch {
    return false;
  }
}

/** `headers` with the signed-in user's token added, if `url` is this page's own origin.
 * Anywhere else gets no token, and no Authorization header at all. */
export async function withAuthHeader(url: string | URL, headers: Headers): Promise<Headers> {
  if (!sameOrigin(url)) {
    headers.delete("Authorization");
    return headers;
  }
  const token = await authToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

/** `fetch`, with the signed-in user's token when there is one and `input` is this origin. */
export async function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = await withAuthHeader(input, new Headers(init.headers));
  return fetch(input, { ...init, headers });
}

/** The agent server's URL: an explicit setting, else the server that served this page. */
export function defaultApiUrl(): string | undefined {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (SAME_ORIGIN && typeof window !== "undefined") return window.location.origin;
  return undefined;
}

export function wasSignedOut(): boolean {
  try {
    return window.localStorage.getItem(SIGNED_OUT_KEY) === "1";
  } catch {
    return false;
  }
}

export function setSignedOut(value: boolean) {
  try {
    if (value) window.localStorage.setItem(SIGNED_OUT_KEY, "1");
    else window.localStorage.removeItem(SIGNED_OUT_KEY);
  } catch {
    // Storage blocked: a reload may then sign in again through the provider's session.
  }
}
