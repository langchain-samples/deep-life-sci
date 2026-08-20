# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Deep Agents demo: a PubMed/PMC research assistant for life scientists. The agent
searches PubMed, retrieves abstracts and PMC full text, queries the ClinicalTrials.gov
registry, fans out cheap subagents across many papers, and computes/plots in a sandboxed
Python container.

`docs/concept.md` holds the design rationale; `docs/measurements.md` holds the measured
numbers behind the architectural choices. Both are worth reading before changing the
shape of the agent. `README.md` is setup only — human onboarding, not a reference.

## Commands

```bash
uv run agent                           # one-shot CLI, default `anthropic` profile
MODEL_PROFILE=mixed uv run agent       # switch model pair (see models.py PROFILES)
uv run scripts/build_snapshot.py       # one-off: bake the scientific/bio Python stack
                                       # into the sandbox snapshot (~100s, rdkit is most of it)
./scripts/dev.sh                       # both halves of the chat stack, Ctrl-C stops both
uv run langgraph dev                   # just the graph, on :2024
cd ../agent-chat-ui && pnpm dev        # just the UI, on :3000 -> http://localhost:2024

uv run python -m evals.sync            # push evals/datasets/*.yaml to LangSmith
uv run python -m evals.run --structural --limit 3   # score, no judge model
uvx ruff check .                       # config lives in pyproject.toml
```

`scripts/dev.sh` is the normal way in: it starts the graph server and the frontend together,
prefixes their logs, and reuses either one that is already listening rather than
fighting it for the port. `AGENT_CHAT_UI=<path>` overrides where it looks for the
frontend.

The chat UI is a local clone of `langchain-ai/agent-chat-ui` living *outside* this repo
(`../agent-chat-ui`), with **two local patches**:

1. A `/ui/:path*` rewrite in `next.config.mjs` pointing at `:2024`. Required — without
   it the artifact components silently never render (see the invariant below).
2. An empty-message early return in `src/components/thread/messages/ai.tsx`. Upstream
   only skips *tool results* when "Hide Tool Calls" is on; an AI turn that is pure
   `thinking` + `tool_use` still renders its `opacity-0` hover CommandBar, costing 24px
   plus the parent `gap-4`. Our runs are dozens of such turns, so unpatched the first
   visible output sits ~900px of whitespace below the question.

There is no test suite. Ruff is configured in `pyproject.toml` and available in the
`dev` dependency group; `evals/` is the closest thing to a regression check.

Skip `scripts/build_snapshot.py` and runs still work — they just pay a ~95s install each
time, per sandbox. `sandbox.py` looks for the snapshot named by `SANDBOX_SNAPSHOT_NAME`
(default `pubmed-py-bio`) and falls back to installing at runtime when nothing matches, so a
missing snapshot is slow rather than broken.

## Environment

Copy `.env.example` to `.env`. The one thing that trips people up: **`OPENAI_API_KEY` is
the LangSmith gateway service key (`lsv2_sk_...`), not an OpenAI key.** All model calls
go through the LangSmith LLM gateway. `LANGSMITH_API_KEY` is for tracing and sandbox
provisioning.

`cli.py` calls `load_dotenv(override=True)` on purpose (an exported `LANGSMITH_PROJECT`
would otherwise capture traces), but captures CLI-passed `MODEL_PROFILE` first and
restores it afterwards. Adding another per-run env override means adding it to
`_CLI_OVERRIDES` in `cli.py`. That ordering is why `cli.py` imports `sandbox.py` *after*
`load_dotenv` — `sandbox.py` reads `SANDBOX_SNAPSHOT_NAME` at import time.

Every host-side path is defined in `research_agent/paths.py`, anchored to the repo root
rather than to any module's location. `RESEARCH_AGENT_DATA_DIR` overrides where the cache
lives; getting that anchor wrong silently starts a second empty cache instead of failing.

## Architecture

```
research_agent/
├── agent.py cli.py graph.py runner.py    assembly + the three entry points
├── sandbox.py                            sandbox lifecycle + WebSocket retry
├── models.py paths.py                    gateway routing, host-side paths
├── prompts/     system.py subagents.py
├── sources/     pubmed.py pmc.py ctgov.py cache_io.py
└── middleware/  artifacts.py perf.py
evals/  scripts/  ui/  docs/  data/
```

`evals/` is deliberately outside the package — it measures the agent, it isn't part of it,
and nothing shipped at deploy time should carry a test framework. Its entry points are run
with `-m` from the repo root (`uv run python -m evals.run`), which is what puts the root on
`sys.path` so `evals.evaluators` resolves. Dataset seeds live in git as JSONL and LangSmith
is the mirror; `sync.py` matches on `seed_id` and never deletes.

Three entry points build the **same** agent via `agent.py:build_agent(backend)`. It only
assembles — it owns no sandbox and no I/O, which is what lets all three share it:

- `cli.py` — one-shot CLI. Takes a `sandbox_session()` for the life of a `with` block and
  streams the answer. The sandbox still gets an `idle_ttl_seconds` as a server-side
  backstop: a `finally` does not survive `kill -9`.
- `graph.py` — LangGraph server (`make_graph`). Sandbox is keyed to `thread_id` and
  reused across turns, so turn 2 sees turn 1's files. Cleanup is `idle_ttl_seconds`
  only; nothing is explicitly deleted. Studio inspection (no `thread_id`) gets
  `_UnboundSandbox`, which raises on any call — never let it reach a real run path.
- `runner.py` — `run_once(question) -> RunResult`. One question, a disposable sandbox, and
  a return value carrying the trajectory, artifact names and root-context size alongside
  the answer. This exists for `evals/`: those three numbers are not recoverable from the
  answer text and are where regressions actually show up.

### The two code surfaces

The root model does **not** call the PubMed tools directly. It writes JavaScript in a
QuickJS interpreter (`CodeInterpreterMiddleware`) and reaches tools through programmatic
tool calling, so search → fetch → fan out → compute → collect happens inside one `eval`.

- `eval` (QuickJS) — orchestration only. No network, filesystem, or shell of its own.
- `execute` (sandbox shell) — real Python 3 in a LangSmith sandbox container, for stats
  and plots.

Everything in the middleware's `ptc=[...]` allowlist appears in JS camelCased
(`pubmed_search` → `tools.pubmedSearch`). **Adding a tool means adding it to that
allowlist and writing a prompt segment for it in `prompts/system.py`** — the model has no other
way to discover it. The non-default `timeout=900`, `max_result_chars=40_000` and
`max_ptc_calls=512` are all load-bearing; the 5s default timeout kills every real fan-out.

`fetch_figures` and `fetch_supplementary` are the exception to module-level tools:
`agent.py` builds them per run from `make_sandbox_tools(backend)`, because an image has to
exist in the sandbox as a real file before `figure-analyst` can `read_file` it.

`CodeInterpreterMiddleware` and dynamic subagents are both **beta** — their APIs may move
between deepagents releases.

### Data flow, and what must never enter root context

PTC tool output is marshalled into the JS heap and never reaches the model's context.
That's the core economy of the design:

- Host-side `data/` (abstracts, PMC) is the cache owned by
  `sources/pubmed.py`/`sources/pmc.py`. **The agent cannot see it** — the sandbox starts empty and the
  agent writes what it needs there with `tools.writeFile`. It is a within-run
  optimisation, not a corpus: entries expire on an idle TTL (see below).
- Abstract/full-text payloads go into *subagent prompts*, not the root transcript.
- `pmc_locate` is the triage step before any full-text call: it reports section titles,
  figure and table captions, supplementary filenames and the body's character cost
  *without* the body. `fetch_full_text` returns ~40k chars for a median paper and is meant
  for a `full-text-analyst`, never for the root.
- `/workspace/out/` is the user deliverables folder; `ArtifactMiddleware` sweeps it after
  every tool call and pushes bytes through the `ui` state key (components in `ui/ui.tsx`).
  The prompt forbids `read_file` on anything in `out/` — reading a PNG back cost more
  context than an entire run.

### Subagents

Four leaves in `prompts/subagents.py`, wired in `agent.py` by `analyst_leaf()`.
deepagents' default is that subagents **inherit the parent's tools**, so every leaf sets
`tools: []` explicitly — and because `middleware: []` does *not* mean "no middleware"
(deepagents prepends its own default filesystem, `execute` included), every leaf also
passes a `FilesystemMiddleware` narrowed to `read_file`, which is the floor
`FilesystemMiddleware` refuses to drop. The two kinds reach that same restriction from
opposite directions:

- `abstract-analyst`, `full-text-analyst` — payload arrives in the prompt, nothing needed
  on disk. Their prompts claim they have no tools; leaving the default toolset in place
  made models invent paths and doubled their latency.
- `figure-analyst` — handed a sandbox path, and `read_file` is what turns it into an image
  the model can see.

The auto-added `general-purpose` subagent *does* inherit the PubMed tools and an
unrestricted filesystem including `execute`. Nothing routes work to it, but don't start.

Root runs the larger model, leaves the cheaper one (`models.py:root_model` /
`subagent_model`). Whenever the leaves are Anthropic (the default `anthropic`, and
`mixed`), subagent prompt caching is a net loss — each leaf is a fresh single-turn agent
with a unique payload, so it pays cache-write premium for reads that never happen. Turn it
off on the subagent model, keep it on the root. Nothing needs configuring at the root under
`mixed` or `openai`: an OpenAI root caches server-side on `/v1`, and
`AnthropicPromptCachingMiddleware` no-ops for it (deepagents constructs it with
`unsupported_model_behavior="ignore"`).

### Model gateway

Three profiles in `models.py:PROFILES`, chosen by `MODEL_PROFILE`: `anthropic` (Sonnet 4.6
root over Haiku 4.5 leaves, the default), `mixed` (GPT-5.6 terra root over Haiku 4.5
leaves), and `openai` (terra over GPT-5.6 luna).

The gateway path is picked per **model id**, not per profile — that's what lets a single
profile mix providers, as `mixed` does. The difference is not cosmetic:
Anthropic models must use the **native** `/anthropic/v1/messages` path or prompt caching
silently stops working (the OpenAI-compatible shim drops `cache_control`). On the native
path the base URL must **not** end in `/v1` (the SDK appends it) and model ids are bare
(`claude-sonnet-4-6`, not `anthropic/...`). OpenAI models use `/v1` and cache
server-side automatically. `_provider_for` reads the path off the id form — bare means
native, prefixed means `/v1` — and rejects anything matching neither rather than sending
it down the wrong path, which the gateway answers with a 501 that looks like an outage.

## Invariants worth preserving

- **Subagents do no I/O.** NCBI allows 3 req/sec (10 with a key); N subagents each
  fetching would collect 429s. Batch-fetch up front is what makes a large fan-out safe.
- **ClinicalTrials.gov is the tightest rate limit in the repo, and it is undocumented.**
  Measured at roughly a 10-token bucket refilling at ~1 req/sec — 12 concurrent requests
  returned ten 429s — and **the 429 carries no `Retry-After`**, so client-side backoff is
  the only thing between a fan-out and a dead run. There is no API key that raises it.
  `sources/ctgov.py:_throttle` serialises every request in the process, which is why that
  module needs no separate concurrency semaphore the way `pmc.py` does. The design
  response is to make per-trial fetching unnecessary rather than merely discouraged:
  `pageSize` reaches 1,000 and `filter.ids` takes 300, so a 1,000-trial corpus is four
  requests.
- **`sources/ctgov.py` inverts `pubmed.py`'s validation story on purpose.** This API
  answers a bad field, enum, area, sort or id with an HTTP 400 naming the token, so 4xx
  bodies are surfaced verbatim and there is no local `check_field_tags()` analog. Do not
  add one. Three behaviours still return a wrong answer rather than an error and each has
  a guard: `pageSize` clamps silently at 1,000, unknown ids vanish from a `filter.ids`
  batch with no missing list, and `countTotal` is opt-in and first-page-only. A fourth is
  a footgun rather than a bug — an unfiltered `/studies` is legal and returns all ~600k
  studies, which is why `ctgov_search` refuses an empty query.
- **Registry reference types are not interchangeable.** `referencesModule` mixes RESULT
  (sponsor-designated, and sparse — 1 of 126 references across one measured phase 3 set),
  DERIVED (NLM's automatic back-link from PubMed's `[si]` field, where the coverage
  actually is) and BACKGROUND (prior literature the sponsor cited — *other people's
  papers*). `_study_to_record` splits them into `result_pmids`, `trial_pmids` and
  `background_pmids` so the model never has to filter on `type`; collapsing them back into
  one list would answer "what has this trial published" with a reading list.
- **The guards in `sources/pubmed.py` and `sources/pmc.py` are not boilerplate.** Each corresponds to a
  verified API failure mode that returns a *wrong answer rather than an error* — PMID
  tokenization, silent query rewriting, esummary's 500-UID cap returning HTTP 200,
  unguessable PMC object version suffixes. Field tags in particular are validated locally
  against PubMed's documented set, so supporting a new one means extending that list.
  Don't simplify them away;
  `docs/pubmed_api_notes/`, `docs/pmc_api_notes/` and `docs/ctgov_api_notes/` (all
  gitignored) have the probe results; the ctgov notes ship a `probe.py` that reproduces
  every measurement, sleeps included.
- **The host cache expires on the same window as the sandbox.** `IDLE_TTL_SECONDS`
  lives in `paths.py` precisely so `sandbox.py` and `sources/cache_io.py` cannot drift
  apart: past that window a returning thread finds neither its container nor its corpus,
  rather than a warm cache pointing into a container that is gone. The TTL is *idle* — a
  hit calls `cache_io.touch`, so a long run keeps its corpus for as long as it is
  working. Expiry only stops a stale entry being used; `cache_io.sweep_if_due()` is what
  bounds the disk, called at run start by each of the three entry points and self-gated
  to once per TTL window (~11ms for 1800 files). `RESEARCH_AGENT_CACHE_TTL=off` restores
  the old permanent cache, which is what `evals/run.py` sets — otherwise whether an
  example refetches would depend on wall clock, and both latency and `from_cache` would
  move for reasons unrelated to the agent.
- **Blocking calls must go through `asyncio.to_thread`** (`sources/cache_io.py`). Under
  `langgraph dev`, blockbuster turns a blocking `read_text()` in a coroutine into a
  `BlockingError` that kills the run; in production it stalls every other run in the
  process.
- **`ResilientSandbox` retries assume idempotence** (`sandbox.py`). It retries
  transient WebSocket failures below the tool boundary, because one HTTP 502 once
  destroyed a completed 18-way fan-out. A tool whose sandbox command must run exactly
  once has to be routed around it.
- **Artifact components only render same-origin.** `/ui/{graph}` returns a script tag
  with a *host-relative* `src`, so the browser resolves it against the page's origin, not
  the API's. On a cross-origin frontend (hosted Agent Chat, Studio at
  `smith.langchain.com`) that 404s, the `onload` never fires, and `LoadExternalComponent`
  paints an empty div — no console error in the transcript, the run looks fine, the chart
  is just absent. Any frontend needs `/ui/*` proxied to the agent server. This is a
  frontend wiring constraint, not a bug in `middleware/artifacts.py` or `ui/ui.tsx`; check the network
  tab for `/ui/<graph>/entrypoint.js` before touching either.
- **Prompt changes are the main tuning lever.** One line telling the model to print
  numbers instead of reading its plot back cut root context from 115k to 31k chars. Treat
  `prompts/` as production code.


## Writing prompts

Agent prompts should be as concise and general as possible. Avoid unnecessary detail and exposition. Unless critical, avoid giving the agent explicit instructions or step-by-step procedures; instead, simply document potential pitfalls or things it should know.