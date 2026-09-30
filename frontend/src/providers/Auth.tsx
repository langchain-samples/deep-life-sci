"use client";

// Holds the app back until the user is signed in, when this build signs in at all
// (see lib/auth.ts). Without sign-in it renders its children straight away.

import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import { UserManager } from "oidc-client-ts";
import {
  AuthConfig,
  CALLBACK_PATH,
  loadAuthConfig,
  setUser,
  userManager,
} from "@/lib/auth";

type AuthState = { signOut: (() => void) | null; name: string | null };

const AuthContext = createContext<AuthState>({ signOut: null, name: null });

export const useAuth = () => useContext(AuthContext);

type Phase =
  | { kind: "loading" }
  | { kind: "open" }
  | { kind: "signed-in"; manager: UserManager; name: string | null }
  | { kind: "failed"; message: string };

async function signIn(config: AuthConfig): Promise<Phase> {
  const manager = userManager(config);
  let user = await manager.getUser();
  if (!user || user.expired) {
    // The provider still has a session after a reload, so this usually returns a user
    // without showing anything; failing that, the full redirect below does.
    user = await manager.signinSilent().catch(() => null);
  }
  if (!user) {
    await manager.signinRedirect({ state: { returnTo: window.location.href } });
    return { kind: "loading" };
  }
  setUser(user);
  const name = (user.profile.name || user.profile.email || null) as string | null;
  return { kind: "signed-in", manager, name };
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<Phase>({ kind: "loading" });

  useEffect(() => {
    if (window.location.pathname.startsWith(CALLBACK_PATH)) return;
    loadAuthConfig()
      .then((config) => (config ? signIn(config) : ({ kind: "open" } as Phase)))
      .then(setPhase)
      .catch((error: Error) => setPhase({ kind: "failed", message: error.message }));
  }, []);

  if (phase.kind === "loading") {
    return <p className="text-muted-foreground p-8 text-center">Signing in…</p>;
  }
  if (phase.kind === "failed") {
    return (
      <div className="mx-auto max-w-md p-8 text-center">
        <p className="font-medium">Sign-in did not complete.</p>
        <p className="text-muted-foreground mt-2 text-sm">{phase.message}</p>
      </div>
    );
  }
  const value: AuthState =
    phase.kind === "signed-in"
      ? { signOut: () => void phase.manager.signoutRedirect(), name: phase.name }
      : { signOut: null, name: null };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
