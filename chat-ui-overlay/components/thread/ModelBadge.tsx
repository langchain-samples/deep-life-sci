import { useEffect, useRef, useState } from "react";
import { useQueryState } from "nuqs";
import { Wrench } from "lucide-react";
import { getApiKey } from "@/lib/api-key";

/**
 * The main agent's model and effort, under the composer, with a wrench that lists every role.
 *
 * Read-only on purpose: models are chosen in `models.yaml` (or by env override) and take
 * effect when the server restarts, so this reports what the server resolved rather than
 * offering a choice it could not apply. The server's `GET /models` (`deep_life_sci/webapp.py`)
 * is the source, because only the running process knows which overrides won.
 *
 * A server without that route, or one that is not up yet, renders nothing. The badge is
 * information, and an error here would sit under every message the user types.
 */

type Role = {
  role: "root" | "subagent" | "search";
  model: string;
  label: string;
  provider: string;
  effort: string;
};

const ROLE_NAMES: Record<Role["role"], string> = {
  root: "Main agent",
  subagent: "Subagents",
  search: "Web search",
};

const PROVIDER_NAMES: Record<string, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
};

const EFFORT_NAMES: Record<string, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "Extra high",
  max: "Max",
};

// A prefixed id names its provider (`openai/...`, or a custom gateway provider's own name);
// a bare id is on the Anthropic-native path, which is what `provider` says.
function providerName({ model, provider }: Role): string {
  const name = model.includes("/") ? model.split("/")[0] : provider;
  return PROVIDER_NAMES[name] ?? name;
}

function effortName(effort: string): string {
  return EFFORT_NAMES[effort] ?? effort;
}

function useModelRoles(): Role[] | null {
  // Resolved the same way `providers/Stream.tsx` resolves them, so this asks the server the
  // chat is talking to.
  const [apiUrl] = useQueryState("apiUrl");
  const [authScheme] = useQueryState("authScheme");
  const url = apiUrl || process.env.NEXT_PUBLIC_API_URL;
  const scheme = authScheme || process.env.NEXT_PUBLIC_AUTH_SCHEME;
  const [roles, setRoles] = useState<Role[] | null>(null);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    const headers = new Headers();
    const apiKey = getApiKey();
    if (apiKey) headers.set("X-Api-Key", apiKey);
    if (scheme) headers.set("X-Auth-Scheme", scheme);
    fetch(`${url}/models`, { headers, signal: controller.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => setRoles(Array.isArray(body?.roles) ? body.roles : null))
      .catch(() => setRoles(null));
    return () => controller.abort();
  }, [url, scheme]);

  return roles;
}

export function ModelBadge() {
  const roles = useModelRoles();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const root = roles?.find((r) => r.role === "root");
  if (!roles || !root) return null;

  return (
    <div
      ref={ref}
      className="relative flex items-center gap-2 px-3.5 pb-2.5 text-xs text-gray-500"
    >
      <span className="font-medium text-gray-600" title={root.model}>
        {root.label}
      </span>
      {root.effort && <span className="ml-6">{effortName(root.effort)}</span>}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Models in use"
        aria-expanded={open}
        className="rounded p-1 hover:bg-gray-200 hover:text-gray-700"
      >
        <Wrench className="size-3.5" />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Models in use"
          className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-2 z-20 mb-2 w-[min(28rem,calc(100vw-2rem))] rounded-xl border bg-white p-4 text-sm text-gray-700 shadow-lg"
        >
          <table className="w-full text-left">
            <thead className="text-xs text-gray-500">
              <tr>
                <th className="pb-2 font-medium"></th>
                <th className="pb-2 font-medium">Provider</th>
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 font-medium">Effort</th>
              </tr>
            </thead>
            <tbody>
              {roles.map((r) => (
                <tr key={r.role} className="border-t">
                  <td className="py-2 pr-3 font-medium whitespace-nowrap">
                    {ROLE_NAMES[r.role] ?? r.role}
                  </td>
                  <td className="py-2 pr-3">{providerName(r)}</td>
                  <td className="py-2 pr-3 break-all" title={r.model}>
                    {r.label}
                  </td>
                  <td className="py-2">
                    {r.effort ? effortName(r.effort) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 border-t pt-3 text-xs leading-relaxed text-gray-500">
            Models and effort levels are set in <code>models.yaml</code> at the
            repository root. Providers and their credentials, including custom
            OpenAI-compatible endpoints, are configured in LangSmith under LLM
            Gateway. Restart the server after changing either.
          </p>
        </div>
      )}
    </div>
  );
}
