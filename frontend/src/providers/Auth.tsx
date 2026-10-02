"use client";

// Holds the app back until the user is signed in, when this build signs in at all
// (see lib/auth.ts). Without sign-in it renders its children straight away.

import { createContext, ReactNode, useContext, useEffect, useState } from "react";
import { UserManager } from "oidc-client-ts";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  AuthConfig,
  BASE_PATH,
  loadAuthConfig,
  onSessionEnded,
  secondsLeft,
  setSignedOut,
  setUser,
  userManager,
  wasSignedOut,
} from "@/lib/auth";

type AuthState = { signOut: (() => void) | null; name: string | null };

const AuthContext = createContext<AuthState>({ signOut: null, name: null });

export const useAuth = () => useContext(AuthContext);

type Phase =
  | { kind: "loading" }
  | { kind: "open" }
  | { kind: "signed-in"; manager: UserManager; name: string | null }
  // `local`: the provider has no sign-out endpoint, so only this app forgot the user.
  | { kind: "signed-out"; manager: UserManager; local: boolean }
  | { kind: "failed"; message: string };

const SESSION_TOAST = "session-ended";

function signInAgain(manager: UserManager) {
  void manager.signinRedirect({ state: { returnTo: window.location.href } });
}

async function signIn(config: AuthConfig): Promise<Phase> {
  const manager = userManager(config);
  // Signed out on purpose: wait to be asked, rather than signing straight back in through
  // the provider session that sign-out may have left alive.
  if (wasSignedOut()) return { kind: "signed-out", manager, local: false };
  let user = await manager.getUser();
  if (!user || secondsLeft(user) <= 0) {
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
    loadAuthConfig()
      .then((config) => (config ? signIn(config) : ({ kind: "open" } as Phase)))
      .then(setPhase)
      .catch((error: Error) => setPhase({ kind: "failed", message: error.message }));
  }, []);

  // When renewal fails and the token has expired, ask rather than navigate: leaving the
  // page would stop a run in progress, and lose whatever is in the composer.
  useEffect(() => {
    if (phase.kind !== "signed-in") return;
    const { manager } = phase;
    const unsubscribe = onSessionEnded((ended) => {
      if (!ended) {
        toast.dismiss(SESSION_TOAST);
        return;
      }
      toast.error("Your sign-in has expired", {
        id: SESSION_TOAST,
        duration: Infinity,
        description:
          "Sign in again to keep working. Signing in reloads the page, which stops a run that is still going.",
        action: { label: "Sign in", onClick: () => signInAgain(manager) },
      });
    });
    return () => {
      unsubscribe();
      toast.dismiss(SESSION_TOAST);
    };
  }, [phase]);

  if (phase.kind === "loading") {
    return <p className="text-muted-foreground p-8 text-center">Signing in…</p>;
  }
  if (phase.kind === "failed") {
    return (
      <div className="mx-auto max-w-md p-8 text-center">
        <p className="font-medium">Sign-in did not complete.</p>
        <p className="text-muted-foreground mt-2 text-sm">{phase.message}</p>
        <Button className="mt-4" onClick={() => window.location.reload()}>
          Try again
        </Button>
      </div>
    );
  }
  if (phase.kind === "signed-out") {
    const { manager } = phase;
    return (
      <div className="mx-auto max-w-md p-8 text-center">
        <p className="font-medium">You are signed out of Deep Life Sci.</p>
        {phase.local && (
          <p className="text-muted-foreground mt-2 text-sm">
            Your identity provider may still have you signed in. On a shared computer, sign out
            there too.
          </p>
        )}
        <Button
          className="mt-4"
          onClick={() => {
            setSignedOut(false);
            // `login` makes the provider ask who is signing in, rather than reusing a
            // session someone else may have left behind.
            void manager.signinRedirect({
              prompt: "login",
              state: { returnTo: `${window.location.origin}${BASE_PATH}/` },
            });
          }}
        >
          Sign in
        </Button>
      </div>
    );
  }
  const value: AuthState =
    phase.kind === "signed-in"
      ? {
          signOut: () => {
            const { manager } = phase;
            setSignedOut(true);
            // Navigates to the provider's sign-out page; providers without one
            // (no `end_session_endpoint`) reject, and only this app signs out.
            manager.signoutRedirect().catch(async () => {
              await manager.removeUser().catch(() => undefined);
              setPhase({ kind: "signed-out", manager, local: true });
            });
          },
          name: phase.name,
        }
      : { signOut: null, name: null };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
