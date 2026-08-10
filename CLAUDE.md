# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Deep Agents demo: a PubMed/PMC research assistant for life scientists. The agent
searches PubMed, retrieves abstracts and PMC full text, fans out cheap subagents across
many papers, and computes/plots in a sandboxed Python container.

`concept.md` holds the design rationale; `README.md` holds the measured numbers behind
the architectural choices. Both are worth reading before changing the shape of the agent.

## Commands

```bash
uv run agent.py                        # one-shot CLI, default `anthropic` profile
MODEL_PROFILE=openai uv run agent.py   # switch model pair (see models.py PROFILES)
uv run build_snapshot.py               # one-off: bake numpy/pandas/scipy/matplotlib
                                       # into the sandbox snapshot (~35s)
./dev.sh                               # both halves of the chat stack, Ctrl-C stops both
uv run langgraph dev                   # just the graph, on :2024
cd ../agent-chat-ui && pnpm dev        # just the UI, on :3000 -> http://localhost:2024
```

`dev.sh` is the normal way in: it starts the graph server and the frontend together,
prefixes their logs, and reuses either one that is already listening rather than
fighting it for the port. `AGENT_CHAT_UI=<path>` overrides where it looks for the
frontend.

The chat UI is a local clone of `langchain-ai/agent-chat-ui` living *outside* this repo
(`../agent-chat-ui`), with **one required patch**: a `/ui/:path*` rewrite in
`next.config.mjs` pointing at `:2024`. Without it the artifact components silently never
render — see the invariant below.

There is no test suite and no lint config in `pyproject.toml`. Ruff has been used ad hoc
(`uvx ruff check .`) but isn't a declared dependency.

Skip `build_snapshot.py` and runs still work — they just pay a ~30s `pip install` each
time, per sandbox.

## Environment

Copy `.env.example` to `.env`. The one thing that trips people up: **`OPENAI_API_KEY` is
the LangSmith gateway service key (`lsv2_sk_...`), not an OpenAI key.** All model calls
go through the LangSmith LLM gateway. `LANGSMITH_API_KEY` is for tracing and sandbox
provisioning.

`agent.py` calls `load_dotenv(override=True)` on purpose (an exported `LANGSMITH_PROJECT`
would otherwise capture traces), but captures CLI-passed `MODEL_PROFILE` first and
restores it afterwards. Adding another per-run env override means adding it to
`_CLI_OVERRIDES` in `agent.py`.

## Architecture

Two entry points build the **same** agent via `agent.py:build_agent(backend)`:

- `agent.py` — CLI. Owns one sandbox for the life of a `with` block, streams the answer.
- `graph.py` — LangGraph server (`make_graph`). Sandbox is keyed to `thread_id` and
  reused across turns, so turn 2 sees turn 1's files. Cleanup is `idle_ttl_seconds`
  only; nothing is explicitly deleted. Studio inspection (no `thread_id`) gets
  `_UnboundSandbox`, which raises on any call — never let it reach a real run path.

### The two code surfaces

The root model does **not** call the PubMed tools directly. It writes JavaScript in a
QuickJS interpreter (`CodeInterpreterMiddleware`) and reaches tools through programmatic
tool calling, so search → fetch → fan out → compute → collect happens inside one `eval`.

- `eval` (QuickJS) — orchestration only. No network, filesystem, or shell of its own.
- `execute` (sandbox shell) — real Python 3 in a LangSmith sandbox container, for stats
  and plots.

Everything in the middleware's `ptc=[...]` allowlist appears in JS camelCased
(`pubmed_search` → `tools.pubmedSearch`). **Adding a tool means adding it to that
allowlist and writing a prompt segment for it in `prompts.py`** — the model has no other
way to discover it. The non-default `timeout=900`, `max_result_chars=40_000` and
`max_ptc_calls=512` are all load-bearing; the 5s default timeout kills every real fan-out.

### Data flow, and what must never enter root context

PTC tool output is marshalled into the JS heap and never reaches the model's context.
That's the core economy of the design:

- Host-side `data/` (abstracts, searches, PMC) is the durable cache owned by
  `pubmed.py`/`pmc.py`. **The agent cannot see it** — the sandbox starts empty and the
  agent writes what it needs there with `tools.writeFile`.
- Abstract/full-text payloads go into *subagent prompts*, not the root transcript.
- `/workspace/out/` is the user deliverables folder; `ArtifactMiddleware` sweeps it after
  every tool call and pushes bytes through the `ui` state key (components in `ui.tsx`).
  The prompt forbids `read_file` on anything in `out/` — reading a PNG back cost more
  context than an entire run.

### Subagents

Three leaves in `prompts.py`, wired in `agent.py`. deepagents' default is that subagents
**inherit the parent's tools**, so every leaf sets `tools: []` explicitly:

- `abstract-analyst`, `full-text-analyst` — `text_leaf()`: no tools, no middleware. Their
  prompts claim they have no tools; giving them `read_file` anyway made models invent
  paths and doubled their latency.
- `figure-analyst` — `image_leaf()`: `FilesystemMiddleware` restricted to `read_file`,
  because it's handed a sandbox path and must open the image to see it.

The auto-added `general-purpose` subagent *does* inherit the PubMed tools and an
unrestricted filesystem including `execute`. Nothing routes work to it, but don't start.

Root runs the larger model, leaves the cheaper one (`models.py:root_model` /
`subagent_model`). Under the `anthropic` profile, subagent prompt caching is a net loss —
each leaf is a fresh single-turn agent with a unique payload, so it pays cache-write
premium for reads that never happen.

### Model gateway

`models.py` picks the gateway path per profile, and the difference is not cosmetic:
Anthropic models must use the **native** `/anthropic/v1/messages` path or prompt caching
silently stops working (the OpenAI-compatible shim drops `cache_control`). On the native
path the base URL must **not** end in `/v1` (the SDK appends it) and model ids are bare
(`claude-sonnet-4-6`, not `anthropic/...`). OpenAI models use `/v1` and cache
server-side automatically.

## Invariants worth preserving

- **Subagents do no I/O.** NCBI allows 3 req/sec (10 with a key); N subagents each
  fetching would collect 429s. Batch-fetch up front is what makes a large fan-out safe.
- **The guards in `pubmed.py` and `pmc.py` are not boilerplate.** Each corresponds to a
  verified API failure mode that returns a *wrong answer rather than an error* — PMID
  tokenization, silent query rewriting, esummary's 500-UID cap returning HTTP 200,
  unguessable PMC object version suffixes. Don't simplify them away; `pubmed_api_notes/`
  and `pmc_api_notes/` (both gitignored) have the probe results.
- **Blocking calls must go through `asyncio.to_thread`** (`cache_io.py`). Under
  `langgraph dev`, blockbuster turns a blocking `read_text()` in a coroutine into a
  `BlockingError` that kills the run; in production it stalls every other run in the
  process.
- **`ResilientSandbox` retries assume idempotence** (`resilience.py`). It retries
  transient WebSocket failures below the tool boundary, because one HTTP 502 once
  destroyed a completed 18-way fan-out. A tool whose sandbox command must run exactly
  once has to be routed around it.
- **Artifact components only render same-origin.** `/ui/{graph}` returns a script tag
  with a *host-relative* `src`, so the browser resolves it against the page's origin, not
  the API's. On a cross-origin frontend (hosted Agent Chat, Studio at
  `smith.langchain.com`) that 404s, the `onload` never fires, and `LoadExternalComponent`
  paints an empty div — no console error in the transcript, the run looks fine, the chart
  is just absent. Any frontend needs `/ui/*` proxied to the agent server. This is a
  frontend wiring constraint, not a bug in `artifacts.py` or `ui.tsx`; check the network
  tab for `/ui/<graph>/entrypoint.js` before touching either.
- **Prompt changes are the main tuning lever.** One line telling the model to print
  numbers instead of reading its plot back cut root context from 115k to 31k chars. Treat
  `prompts.py` as production code.


## Writing prompts

Agent prompts should be as concise and general as possible. Avoid 