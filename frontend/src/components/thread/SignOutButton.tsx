"use client";

import { LogOut } from "lucide-react";
import { useAuth } from "@/providers/Auth";
import { TooltipIconButton } from "./tooltip-icon-button";

/** Shown only when this build signs in (lib/auth.ts). */
export function SignOutButton() {
  const { signOut, name } = useAuth();
  if (!signOut) return null;
  return (
    <TooltipIconButton
      size="lg"
      className="p-4"
      tooltip={name ? `Sign out ${name}` : "Sign out"}
      variant="ghost"
      onClick={signOut}
    >
      <LogOut className="size-5" />
    </TooltipIconButton>
  );
}
