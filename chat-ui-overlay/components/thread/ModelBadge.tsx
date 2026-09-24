import { useEffect, useRef, useState } from "react";
import { useQueryState } from "nuqs";
import { Wrench } from "lucide-react";
import { getApiKey } from "@/lib/api-key";

/**
 * The main agent's model and effort, under the composer, with a wrench that lists every role.
 *
 * Read-only on purpose: models are chosen in `models.yaml` (or by env override), so this
 * reports what the server resolved rather than offering a choice here. The server's
 * `GET /models` (`deep_life_sci/webapp.py`) is the source, because only the running process
 * knows which overrides won. It re-reads models.yaml on each request, as each run does, and
 * the wrench asks again every time it opens, so an edit shows without reloading the page.
 *
 * A server without that route renders nothing. A models.yaml the server cannot use renders
 * as an error, with the server's message in the wrench, since the next message would fail.
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

type Models = { roles: Role[] } | { error: string };

function useModels(asked: number): Models | null {
  // Resolved the same way `providers/Stream.tsx` resolves them, so this asks the server the
  // chat is talking to.
  const [apiUrl] = useQueryState("apiUrl");
  const [authScheme] = useQueryState("authScheme");
  const url = apiUrl || process.env.NEXT_PUBLIC_API_URL;
  const scheme = authScheme || process.env.NEXT_PUBLIC_AUTH_SCHEME;
  const [models, setModels] = useState<Models | null>(null);

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    const headers = new Headers();
    const apiKey = getApiKey();
    if (apiKey) headers.set("X-Api-Key", apiKey);
    if (scheme) headers.set("X-Auth-Scheme", scheme);
    let timer: ReturnType<typeof setTimeout> | undefined;

    // `scripts/dev.py` opens the page once the UI answers, which can be well before the
    // agent server has loaded the graph: until then the connection is refused, and while
    // it starts a proxy in front of it may answer 502-504. Those are retried, with backoff,
    // for about two minutes. Any other answer is final: a 500 carries a models.yaml error
    // that asking again will not fix, and a server without this route says so with a 404.
    const retry = (attempt: number) => {
      if (controller.signal.aborted || attempt >= 12) return;
      timer = setTimeout(
        () => load(attempt + 1),
        Math.min(1000 * 2 ** attempt, 15000),
      );
    };
    const load = (attempt: number) => {
      fetch(`${url}/models`, { headers, signal: controller.signal })
        .then(async (res) => {
          if ([502, 503, 504].includes(res.status)) return retry(attempt);
          const body = await res.json().catch(() => null);
          if (Array.isArray(body?.roles)) setModels({ roles: body.roles });
          else if (typeof body?.error === "string")
            setModels({ error: body.error });
          else setModels(null);
        })
        .catch(() => retry(attempt));
    };
    load(0);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [url, scheme, asked]);

  return models;
}

export function ModelBadge() {
  const [open, setOpen] = useState(false);
  const [asked, setAsked] = useState(0);
  const models = useModels(asked);
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

  if (!models) return null;
  const error = "error" in models ? models.error : null;
  const roles = "roles" in models ? models.roles : [];
  const root = roles.find((r) => r.role === "root");
  if (!error && !root) return null;

  return (
    <div
      ref={ref}
      className="relative flex items-center gap-1.5 px-2.5 pb-2.5 text-xs text-gray-500"
    >
      {/* First, so it reads as configuring the whole line rather than the effort beside it. */}
      <button
        type="button"
        onClick={() => {
          // Asking again on open shows a models.yaml edit without reloading the page.
          if (!open) setAsked((n) => n + 1);
          setOpen((v) => !v);
        }}
        aria-label="Models in use"
        aria-expanded={open}
        className="rounded p-1 hover:bg-gray-200 hover:text-gray-700"
      >
        <Wrench className="size-3.5" />
      </button>
      {error ? (
        <span className="font-medium text-red-600">Model settings error</span>
      ) : (
        root && (
          <>
            <span
              className="font-medium text-gray-600"
              title={root.model}
            >
              {root.label}
            </span>
            {root.effort && (
              <span className="ml-2">{effortName(root.effort)}</span>
            )}
          </>
        )
      )}

      {open && (
        <div
          role="dialog"
          aria-label="Models in use"
          className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-2 z-20 mb-2 w-[min(28rem,calc(100vw-2rem))] rounded-xl border bg-white p-4 text-sm text-gray-700 shadow-lg"
        >
          {error ? (
            <p className="text-xs leading-relaxed break-words whitespace-pre-wrap text-red-700">
              {error}
            </p>
          ) : (
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
                  <tr
                    key={r.role}
                    className="border-t"
                  >
                    <td className="py-2 pr-3 font-medium whitespace-nowrap">
                      {ROLE_NAMES[r.role] ?? r.role}
                    </td>
                    <td className="py-2 pr-3">{providerName(r)}</td>
                    <td
                      className="py-2 pr-3 break-all"
                      title={r.model}
                    >
                      {r.label}
                    </td>
                    <td className="py-2">
                      {r.effort ? effortName(r.effort) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-3 border-t pt-3 text-xs leading-relaxed text-gray-500">
            Models and effort levels are set in <code>models.yaml</code> at the
            repository root, and edits apply to your next message. Provider API
            keys go in LangSmith under Settings → Integrations → Provider
            Secrets.
          </p>
        </div>
      )}
    </div>
  );
}
