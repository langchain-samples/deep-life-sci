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

let configPromise: Promise<AuthConfig | null> | null = null;
let manager: UserManager | null = null;
let current: User | null = null;
let tokenKind: AuthConfig["token"] = "id";

/** The server's sign-in settings, or null when this build does not sign in. */
export function loadAuthConfig(): Promise<AuthConfig | null> {
  if (!SAME_ORIGIN) return Promise.resolve(null);
  configPromise ??= fetch(`${BASE_PATH}/config.json`, { cache: "no-store" })
    .then(async (res) => {
      if (!res.ok) return null;
      const body = await res.json();
      return body?.auth === "oidc" ? (body as AuthConfig) : null;
    })
    .catch(() => null);
  return configPromise;
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
    automaticSilentRenew: true,
    userStore: new WebStorageStateStore({ store: new InMemoryWebStorage() }),
  });
  manager.events.addUserLoaded((user) => {
    current = user;
  });
  manager.events.addUserUnloaded(() => {
    current = null;
  });
  // Renewal failed (the provider session ended, or third-party cookies blocked the hidden
  // iframe): sign in again rather than let requests go out without a token.
  manager.events.addAccessTokenExpired(() => {
    void manager?.signinRedirect({ state: { returnTo: window.location.href } });
  });
  return manager;
}

export function setUser(user: User | null) {
  current = user;
}

/** The token requests carry, or null before sign-in and in a build without it. */
export function authToken(): string | null {
  if (!current || current.expired) return null;
  return (tokenKind === "access" ? current.access_token : current.id_token) ?? null;
}

export function withAuthHeader(headers: Headers): Headers {
  const token = authToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

/** `fetch`, with the signed-in user's token when there is one. */
export function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  return fetch(input, { ...init, headers: withAuthHeader(new Headers(init.headers)) });
}

/** The agent server's URL: an explicit setting, else the server that served this page. */
export function defaultApiUrl(): string | undefined {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (SAME_ORIGIN && typeof window !== "undefined") return window.location.origin;
  return undefined;
}
