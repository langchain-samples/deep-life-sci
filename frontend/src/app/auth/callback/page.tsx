"use client";

// Where the identity provider sends the browser back, for a full sign-in and for the
// hidden-iframe renewals oidc-client-ts runs before a token expires (lib/auth.ts).

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { BASE_PATH, loadAuthConfig, setUser, userManager } from "@/lib/auth";

export default function AuthCallback() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAuthConfig()
      .then(async (config) => {
        if (!config) return router.replace("/");
        const user = await userManager(config).signinCallback();
        // A renewal: the page that started it picks the new token up from the manager.
        if (window.parent !== window) return;
        if (user) setUser(user);
        // Client-side navigation, not a reload: the token lives in memory only.
        const state = (user?.state ?? {}) as { returnTo?: string };
        const target = state.returnTo ? new URL(state.returnTo) : null;
        const sameApp = target && target.origin === window.location.origin;
        const path = sameApp ? target.pathname.slice(BASE_PATH.length) || "/" : "/";
        router.replace(`${path}${sameApp ? target.search : ""}`);
      })
      .catch((e: Error) => setError(e.message));
  }, [router]);

  return (
    <p className="text-muted-foreground p-8 text-center">
      {error ? `Sign-in did not complete: ${error}` : "Signing in…"}
    </p>
  );
}
