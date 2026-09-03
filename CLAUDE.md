# CLAUDE.md

Guidance for Claude Code in this repository. **Keep this file at or below 200 lines**, and
keep it to what cannot be read off the code: commands, rules, and traps. Every module here
carries a docstring explaining itself — point at it rather than restating it.

## What this is

A Deep Agents demo: a PubMed/PMC research assistant for life scientists. It searches PubMed,
retrieves abstracts and PMC full text, queries ClinicalTrials.gov, searches the web for what
neither holds, fans out cheap subagents across papers, and computes/plots in a sandbox.

Read `docs/concept.md` (design rationale) and `docs/measurements.md` (the numbers behind
the architectural choices) before changing the shape of the agent; `docs/ctgov_concept.md`
and `docs/pmc_concept.md` cover those two surfaces. `README.md` is human setup only — keep
it to what a new user needs to run the thing, nothing more.

## Commands

```bash
uv run scripts/setup.py                # once per clone: .env, deps, snapshot, chat UI
uv run scripts/dev.py                  # the chat stack, both halves, Ctrl-C stops both
                                       # NO_BROWSER=1 skips opening the browser tab
uv run agent ["question"]              # bare = DEMO_QUESTION
ROOT_MODEL=claude-sonnet-5 uv run agent        # swap one role; SUBAGENT_/JUDGE_ too
uv run scripts/build_snapshot.py       # one-off: bake the scientific/bio Python stack
                                       # into the sandbox snapshot (~100s, rdkit is most of it)
uv run langgraph dev                   # just the graph, on :2024

uv run python -m evals.sync            # push evals/datasets/*.yaml to LangSmith
uv run python -m evals.run --structural --limit 3   # score, no judge model
uvx ruff check .                       # config lives in pyproject.toml
```

`setup.py` and `dev.py` are split so starting the agent never triggers an install: setup does
`.env`, `uv sync`, the snapshot and the frontend, each skipped when already done; dev
*verifies* them instead. **Setup is the only thing that writes `.env`**, so a new required
setting needs a prompt there plus a line in `.env.example`.

There is no test suite; `evals/` is the closest thing to a regression check (see
`evals/README.md`) and ruff is configured in `pyproject.toml` (`dev` group). Skipping
`build_snapshot.py` is slow, not broken: `sandbox.py` falls back to a ~95s runtime install
per sandbox when no snapshot matches `SANDBOX_SNAPSHOT_NAME`.

The chat UI is a gitignored clone of `langchain-ai/agent-chat-ui` at `.chat-ui/`, patched by
setup on every run, with our own components in `chat-ui-overlay/`. **See
`scripts/CLAUDE.md`** before touching any of that.

## Environment

`scripts/setup.py` writes `.env`. **One LangSmith key**, prompted for once and written to
both names that read it: `LANGSMITH_API_KEY` (tracing, sandboxes, fallback for models) and
`LANGSMITH_GATEWAY_API_KEY` (model calls; `models.py:gateway_key()` is the order). Same value
unless hand-edited to bill models elsewhere — setup only fills it when empty, so that edit
survives. The gateway takes a **LangSmith** key, never a provider key: an OpenAI/Anthropic
key lives workspace-side under Settings → Integrations → Provider Secrets, and one sent from
a client gets a 403 (verified).

Model env var names live in `models.py:ENV_VARS`, imported by `cli.py` and `evals/run.py`. A
new axis is preserved by adding it *there and nowhere else* — a hand-copied list is how a
`ROOT_MODEL=...` on the command line silently loses to `.env`.

Every host-side path is defined in `research_agent/paths.py`, anchored to the repo root rather
than to any module's location. Getting that anchor wrong silently starts a second empty cache
instead of failing. `RESEARCH_AGENT_DATA_DIR` overrides it.

## Architecture

```
research_agent/
├── agent.py cli.py graph.py runner.py    assembly + the three entry points
├── sandbox.py                            sandbox lifecycle + WebSocket retry
├── models.py paths.py                    gateway routing, host-side paths
├── prompts/     system.py subagents.py
├── sources/     pubmed.py pmc.py ctgov.py web.py cache_io.py _http.py
└── middleware/  artifacts.py uploads.py perf.py progress.py
evals/  scripts/  ui/  chat-ui-overlay/  docs/  data/
```

Three entry points build the **same** agent via `agent.py:build_agent(backend)`, which only
assembles — it owns no sandbox and no I/O, which is what lets all three share it:

- `cli.py` — one-shot CLI, sandbox for the life of a `with` block.
- `graph.py` — LangGraph server. Sandbox keyed to `thread_id` and reused across turns, so
  turn 2 sees turn 1's files. Studio inspection (no `thread_id`) gets `_UnboundSandbox`,
  which raises on any call — never let it reach a real run path.
- `runner.py` — `run_once(question) -> RunResult`. Exists for `evals/`: the trajectory,
  artifact names and `root_context_chars` aren't recoverable from the answer text, and are
  where regressions show up.

`evals/` is deliberately outside the package — nothing shipped at deploy time should carry a
test framework. Its entry points must run with `-m` from the repo root.

### The two code surfaces

The root model does **not** call the PubMed tools directly. It writes JavaScript in a QuickJS
interpreter (`CodeInterpreterMiddleware`) and reaches tools through programmatic tool calling,
so search → fetch → fan out → compute → collect happens inside one `eval`.

- `eval` (QuickJS) — orchestration only. No network, filesystem, or shell of its own.
- `execute` (sandbox shell) — real Python 3 in a LangSmith sandbox container, for stats
  and plots.

Everything in the middleware's `ptc=[...]` allowlist appears in JS camelCased (`pubmed_search`
→ `tools.pubmedSearch`). **Adding a tool means adding it to that allowlist and writing a
prompt segment for it in `prompts/system.py`** — the model has no other way to discover it.
The non-default `timeout=900`, `max_result_chars=40_000` and `max_ptc_calls=512` are all
load-bearing; the 5s default timeout kills every real fan-out.

`fetch_figures` and `fetch_supplementary` are the exception to module-level tools: `agent.py`
builds them per run from `make_sandbox_tools(backend)`, because an image has to exist in the
sandbox as a real file before `figure-analyst` can `read_file` it.

`CodeInterpreterMiddleware` and dynamic subagents are both **beta** — their APIs may move
between deepagents releases.

### What must never enter root context

PTC tool output is marshalled into the JS heap and never reaches the model's context. That is
the core economy of the design, and these are its rules:

- **Payloads go into subagent prompts, not the root transcript.** `pmc_locate` is the triage
  step before any full-text call — sections, captions, supplementary filenames and the body's
  character cost, *without* the body. `fetch_full_text` returns ~40k chars for a median paper
  and is meant for a `full-text-analyst`, never the root.
- **Host-side `data/` is invisible to the agent.** The sandbox starts empty; the agent writes
  what it needs with `tools.writeFile`. A within-run optimisation, not a corpus.
- **`/workspace/out/` is the user deliverables contract**, swept by `ArtifactMiddleware` and
  published through the `ui` state key (`ui/ui.tsx`). The prompt forbids `read_file` on
  anything in it — reading a PNG back cost more context than an entire run.
- **Web search is a PTC tool, never a spec bound to the root model.** Both providers ship it
  server-side, so binding it lands every retrieved page in root context outside `eval` — 27.9k
  input tokens for one question, measured. `sources/web.py` spends the search in a throwaway
  `search`-role call and returns a digest; a raw provider dict can't reach
  `create_deep_agent(tools=...)` anyway (`langchain_quickjs/_ptc.py:100` reads `.name`).
- **`/workspace/uploads/` is not under `out/`**, or a file the user gave us would come back as
  a deliverable of their own question. `UploadMiddleware` strips attachments out of the human
  message before the first model call and passes a manifest instead; the durable copy lives in
  the LangGraph store, per thread, because a container is reaped on `IDLE_TTL_SECONDS`.

`middleware/progress.py` is the run's only visible output while it works — everything happens
inside one `eval`, so the transcript shows a single unreturned tool call for minutes.

### Subagents

Four leaves in `prompts/subagents.py`, wired in `agent.py` by `analyst_leaf()`. Two deepagents
defaults bite here, and every leaf works around both:

- subagents **inherit the parent's tools**, so each leaf sets `tools: []` explicitly. Leaving
  the default toolset in place made models invent paths and doubled their latency.
- `middleware: []` does **not** mean "no middleware" — deepagents prepends its own filesystem,
  `execute` included. So each leaf also passes a `FilesystemMiddleware` narrowed to
  `read_file`, the floor it refuses to drop (and what `figure-analyst` needs to see an image).

The auto-added `general-purpose` subagent *does* inherit the PubMed tools and an unrestricted
filesystem including `execute`. Nothing routes work to it, but don't start.

### Model gateway

Four roles × three independent env vars, defaulting to twelve constants in `models.py`:

| | `_MODEL` | `_PROVIDER` | `_EFFORT` |
|---|---|---|---|
| `ROOT` | `openai/gpt-5.6-terra` | `openai` | `low` |
| `SUBAGENT` | `openai/gpt-5.6-luna` | `openai` | `low` |
| `SEARCH` | `openai/gpt-5.6-luna` | `openai` | `low` |
| `JUDGE` | `openai/gpt-5.6-luna` | `openai` | `low` |

Root runs the larger model, leaves and `SEARCH` the cheaper one; `SEARCH` carries the
provider's own web search (`WEB_SEARCH_SPECS`), so it must be an id that supports it or the
call 400s. `models.py`'s docstring has the rest — why three axes rather than named profiles,
why Anthropic models must go down the native `/anthropic` path or silently lose prompt
caching, and what an unset `_EFFORT` means on each path. Two things to know before a swap:
naming a `_PROVIDER` that the model id's form contradicts makes `_provider_for` raise rather
than send it down the wrong path, and **Haiku 4.5 has no effort scale**, so
`SUBAGENT_MODEL=claude-haiku-4-5-20251001` must also clear `SUBAGENT_EFFORT=` or it 400s.

## Invariants worth preserving

Each of these is a rule whose rationale lives in the file named beside it. Read that docstring
before changing the behaviour.

- **Subagents do no I/O.** NCBI allows 3 req/sec (10 with a key); N subagents each fetching
  would collect 429s. Batch-fetch up front is what makes a large fan-out safe.
- **Rate limits and API guards are per-source and load-bearing.** See
  `research_agent/sources/CLAUDE.md` before editing anything under `sources/`.
- **The host cache expires on the same idle window as the sandbox** — `IDLE_TTL_SECONDS` is in
  `paths.py` so `sandbox.py` and `sources/cache_io.py` cannot drift into a warm cache pointing
  at a dead container. `RESEARCH_AGENT_CACHE_TTL=off` restores the permanent cache, which is
  what `evals/run.py` sets so refetching doesn't depend on wall clock.
- **Blocking calls must go through `asyncio.to_thread`** (`sources/cache_io.py`). Under
  `langgraph dev`, blockbuster turns a blocking `read_text()` in a coroutine into a
  `BlockingError` that kills the run; in production it stalls every other run in the process.
- **`ResilientSandbox` retries assume idempotence** (`sandbox.py`). It retries transient
  WebSocket failures below the tool boundary, because one HTTP 502 once destroyed a completed
  18-way fan-out. A tool whose sandbox command must run exactly once has to be routed around
  it.
- **Artifact components only render same-origin.** `/ui/{graph}` returns a script tag with a
  *host-relative* `src`, so a cross-origin frontend (hosted Agent Chat, Studio) 404s it and
  `LoadExternalComponent` paints an empty div — no console error, the chart just absent. Any
  frontend needs `/ui/*` proxied. Check the network tab for `/ui/<graph>/entrypoint.js`
  before suspecting `middleware/artifacts.py` or `ui/ui.tsx`.
- **Graph state is downloaded whole by any client listing threads.** The QuickJS snapshot is
  the heavy one, up to 11 MB of base64 per thread: the sidebar asks for `select`/`extract`,
  not `values` (`patch_thread_search`), and reads get an `_UnboundSandbox` (`graph.py`).
- **Prompt changes are the main tuning lever.** One line telling the model to print numbers
  instead of reading its plot back cut root context from 115k to 31k chars. Treat `prompts/`
  as production code.

## Writing prompts

Agent prompts should be as concise and general as possible. Avoid unnecessary detail and
exposition. Unless critical, avoid giving the agent explicit instructions or step-by-step
procedures; instead, simply document potential pitfalls or things it should know.
