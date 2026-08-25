# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Keep this file at or below 250 lines--if it
drifts longer, prune where possible.

## What this is

A Deep Agents demo: a PubMed/PMC research assistant for life scientists. The agent
searches PubMed, retrieves abstracts and PMC full text, queries the ClinicalTrials.gov
registry, fans out cheap subagents across many papers, and computes/plots in a sandboxed
Python container.

`docs/concept.md` holds the design rationale; `docs/measurements.md` holds the measured
numbers behind the architectural choices. Both are worth reading before changing the
shape of the agent. `README.md` is setup only — human onboarding, not a reference. Keep `README.md` as succinct and simple as possible--leave only the level of detail necessary
for the user to set up the repo and understand what it is and how to use it.

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
cd .chat-ui && pnpm dev                # just the UI, on :3000 -> http://localhost:2024

uv run python -m evals.sync            # push evals/datasets/*.yaml to LangSmith
uv run python -m evals.run --structural --limit 3   # score, no judge model
uvx ruff check .                       # config lives in pyproject.toml
```

`scripts/setup.py` and `scripts/dev.py` are the front door, split so starting the agent never
triggers an install: setup does `.env`, `uv sync`, the snapshot and the frontend, each skipped
when already done; dev *verifies* them instead. Setup is the only thing that writes `.env`, so
a new required setting needs a prompt there plus a line in `.env.example`.

Python rather than shell, so every platform runs the same command: a port check, a browser and
killing a process tree have no portable shell spelling. Two bash shims used to wrap these for
macOS/Linux; both are gone, because the only thing neither Python nor the user could do was
install uv, and installing uv is now a documented step in `README.md`. Preflight lives in
`_common.require_setup` and `cli.py:run`, so it applies on every platform.

`dev.py` reuses either port already listening and opens `:3000?hideToolCalls=true` once it
answers — polled, because a tab arriving before `next dev` binds shows a connection error.
The query param is nuqs state, so it is a default without a patch and the in-app toggle
still turns it off. Each server gets its own process group so teardown kills the whole tree, and **all four of SIGINT, SIGTERM, SIGHUP
and SIGQUIT are installed explicitly** (by `getattr`, since the last two are absent on
Windows). Any one left out skips the teardown silently and leaves both ports held — and
because reuse only asks whether *something* is listening, the next launch then serves the run
off that stale stack. SIGINT because its default handler may not be installed at all (a
background process in a non-interactive shell inherits `SIG_IGN` and Python keeps that
disposition); SIGHUP because closing the terminal signals only the foreground group, and the
servers sit in sessions of their own — so it reaches just the process meant to kill them.
The chat UI is not optional and has no flag; Node is the one thing setup will not install
unasked, so it comes *last* — a machine without Node still gets a working `uv run agent`.

The UI is a clone of `langchain-ai/agent-chat-ui` vendored at `.chat-ui/`, inside the repo so
nothing needs a writable directory beside it — which is what `.dockerignore` is for, since
`langgraph build`'s context is this directory and an unignored `.chat-ui` ships a dev-only
Next app in the deploy image (its comments list what must *not* be excluded).
`AGENT_CHAT_UI=<path>` points at a checkout elsewhere. Setup also runs `npm ci` in `ui/`,
whose components the *graph server* bundles: without those deps the bundler logs
`Could not resolve "xlsx"`, answers `/ui/<graph>/entrypoint.js` with a 200 anyway, and the
component is silently absent. Eleven patches, all reapplied by setup on every run so the
README's steps alone produce a working app, and each anchored on an exact upstream string
that it prints rather than guesses past — a moved anchor must not clobber an upstream fix.
Each leaves a mark behind, named in `setup.py:PATCH_MARKS`, which is also what its own
early-out tests; `dev.py` checks those marks and re-applies on every launch, because the
clone is gitignored and an un-patched one otherwise sits there with the feature simply
absent and nothing saying so:

1. A `/ui/:path*` rewrite in `next.config.mjs` pointing at `:2024`. Required — without it the
   artifact components silently never render (see the invariant below). Bails out if upstream
   ever grows a `rewrites` key, since a second one would silently shadow the first.
2. An empty-message early return in `src/components/thread/messages/ai.tsx`: an AI turn that
   is pure `thinking` + `tool_use` still renders its hover CommandBar, and our runs are dozens
   of such turns, so unpatched the first visible output sits ~900px of whitespace below the
   question.
3. `clip-path` → `clipPath` in `src/components/icons/langgraph.tsx`. Valid SVG, invalid JSX,
   so upstream's logo logs `Invalid DOM property` on every render and parks the dev overlay's
   error badge on a healthy app.
4. A spreadsheet-only upload allowlist across `src/hooks/use-file-upload.tsx`,
   `src/lib/multimodal-utils.ts` and `MultimodalPreview.tsx` — CSV/TSV/xlsx/xlsm ride in as
   `type: "file"` blocks and `middleware/uploads.py` takes them back out; images and PDFs stay
   refused, since an attachment the graph does not intercept is model context and nothing
   else. Every call site tests `isSupportedUpload`/`isSpreadsheetUpload` rather than the MIME
   list, because Windows with Excel installed reports a `.csv` as `application/vnd.ms-excel`
   and some browsers report `""` — extension first, MIME second. `accept="*/*"` stays on the
   composer input so a `.xls` reaches the toast telling the user to re-save it rather than
   being greyed out of the picker. `patch_uploads` accepts two baselines, upstream's and the
   earlier "nothing is accepted yet" patch, so an existing clone upgrades in place.
5. A `RunProgressContext` in `src/providers/Stream.tsx`, fed from the `onCustomEvent` hook
   that already handles UI messages, publishing the latest `{type: "progress"}` event.
6. `<RunStatus />` in `src/components/thread/index.tsx`, replacing the typing dots — which
   upstream stops showing once any AI message arrives, i.e. seconds into a run that lasts
   minutes. The component itself is ours; see the overlay below. This is the one patch that
   *deletes* upstream code: `firstTokenReceived` and `prevMessageLength` existed only to hide
   those dots, so with them gone both are written on three paths and read on none — left in,
   they read as if they still govern the loading indicator.
7. `devIndicators: false` in `next.config.mjs`, hiding Next's dev-overlay button in the
   bottom-left corner: this app is run by end users through `dev.py`, not by anyone working
   on the frontend. Its own patch rather than a second line in the rewrite above, so a clone
   that already has the rewrite still picks it up.
8. The header's link to `langchain-ai/agent-chat-ui` removed from
   `src/components/thread/index.tsx` — it points at the chat client, so to a user here it is
   a link to someone else's project. Takes the `OpenGitHubRepo` component and its now-unused
   imports with it, since a component with no call site is a lint error rather than harmless
   dead code; `src/components/icons/github.tsx` is left alone as untouched upstream.
9. `Agent Chat` → `setup.py:APP_NAME` across `layout.tsx`, `Stream.tsx` and
   `thread/index.tsx`. The header and empty-state heading are what a user reads, but a
   browser tab still saying `Agent Chat` is how a rename looks half-done, so all three
   move together. The one patch that is a bare rename rather than an anchored edit — the
   string is upstream's own product name and may move around within those files. It
   replaces `setup.py:PRIOR_APP_NAMES` too: a clone patched under an earlier name has no
   `Agent Chat` left to match, and the clone is gitignored, so that list is the only record
   the rename has. Changing `APP_NAME` means appending the old one there.
10. The empty state's logo and heading stacked rather than side by side, in
    `thread/index.tsx`. Its own patch rather than another edit inside the rename above,
    so a clone already carrying the rename still picks it up.
11. A 🧪 (U+1F9EA TEST TUBE) beside the name, in `thread/index.tsx`. Both places the name
    renders as a heading, so the two do not disagree; not the browser tab or the setup
    form, where it would read as decoration in a sentence. U+2697 ALEMBIC is the literal
    beaker but is text-presentation, so it renders monochrome and tiny on some platforms.

`chat-ui-overlay/` is the other half of that split, and the answer to the question the patch
list keeps raising. The eleven patches above are upstream-shaped — small, anchored, plausibly
things upstream would take. Product surface is the opposite: no upstream counterpart, never
converging, and the worst possible fit for a search-and-replace living inside a Python string.
So it is ordinary `.tsx` files in this repo, tracked in git and reviewable in a diff, which
`setup.py:ensure_overlay` copies into the clone's `src/` (the directory mirrors it, so a
file's path here is where it lands) — leaving the anchored patch as the one line that mounts
them. The copy is one-way: the clone is gitignored, so an edit made over there is a lost edit
whatever happens, and losing it on the next launch beats keeping it and diverging silently.
Fork upstream if an overlay component ever needs to *replace* `index.tsx` wholesale, or if
the app's identity has to change; `UI_REPO` in `setup.py` is then the only line to move.

There is no test suite; `evals/` is the closest thing to a regression check, and ruff is
configured in `pyproject.toml` (`dev` group). Skip `scripts/build_snapshot.py` and runs still
work — they just pay a ~95s install each time, per sandbox: `sandbox.py` looks for the
snapshot named by `SANDBOX_SNAPSHOT_NAME` (default `pubmed-py-bio`) and falls back to
installing at runtime when nothing matches, so a missing snapshot is slow rather than broken.

## Environment

`scripts/setup.py` writes `.env`; the trip-up it exists to prevent is that
**`OPENAI_API_KEY` is the LangSmith gateway service key (`lsv2_sk_...`), not an OpenAI
key** — all model calls go through the LangSmith LLM gateway. `LANGSMITH_API_KEY` is for
tracing and sandbox provisioning.

`cli.py` calls `load_dotenv(override=True)` on purpose (an exported `LANGSMITH_PROJECT`
would otherwise capture traces), but captures the CLI-passed model vars first and
restores them afterwards. The names come from `models.py:ENV_VARS`, which `cli.py` and
`evals/run.py` both import, so a new axis is preserved by adding it there and nowhere
else — a hand-copied list is how a `ROOT_MODEL=...` on the command line silently loses to
`.env`. That ordering is why `cli.py` imports `sandbox.py` *after* `load_dotenv` —
`sandbox.py` reads `SANDBOX_SNAPSHOT_NAME` at import time, whereas `models.py` reads the
environment only inside functions and is therefore safe to import before it.

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
├── sources/     pubmed.py pmc.py ctgov.py cache_io.py _http.py
└── middleware/  artifacts.py uploads.py perf.py progress.py
evals/  scripts/  ui/  chat-ui-overlay/  docs/  data/
```

`evals/` is deliberately outside the package — it measures the agent, it isn't part of it, and
nothing shipped at deploy time should carry a test framework. Its entry points run with `-m`
from the repo root (`uv run python -m evals.run`), which puts the root on `sys.path` so
`evals.evaluators` resolves. Seeds live in git as YAML and LangSmith is the mirror; `sync.py`
matches each seed's `id` (carried into example metadata as `seed_id`) and never deletes.

Three entry points build the **same** agent via `agent.py:build_agent(backend)`. It only
assembles — it owns no sandbox and no I/O, which is what lets all three share it:

- `cli.py` — one-shot CLI. Takes a `sandbox_session()` for the life of a `with` block and
  streams the answer. The sandbox still gets an `idle_ttl_seconds` as a server-side backstop:
  a `finally` does not survive `kill -9`.
- `graph.py` — LangGraph server (`make_graph`). Sandbox is keyed to `thread_id` and reused
  across turns, so turn 2 sees turn 1's files. Cleanup is `idle_ttl_seconds` only; nothing is
  explicitly deleted. Studio inspection (no `thread_id`) gets `_UnboundSandbox`, which raises
  on any call — never let it reach a real run path.
- `runner.py` — `run_once(question) -> RunResult`. One question, a disposable sandbox, and a
  return value carrying the trajectory, artifact names and root-context size alongside the
  answer. This exists for `evals/`: those three numbers are not recoverable from the answer
  text and are where regressions actually show up.

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

- Host-side `data/` (abstracts, PMC) is the cache owned by `sources/pubmed.py`/`sources/pmc.py`.
  **The agent cannot see it** — the sandbox starts empty and the agent writes what it needs
  there with `tools.writeFile`. A within-run optimisation, not a corpus: entries expire on an
  idle TTL (see below).
- Abstract/full-text payloads go into *subagent prompts*, not the root transcript.
- `pmc_locate` is the triage step before any full-text call: it reports section titles,
  figure and table captions, supplementary filenames and the body's character cost
  *without* the body. `fetch_full_text` returns ~40k chars for a median paper and is meant
  for a `full-text-analyst`, never for the root.
- `/workspace/out/` is the user deliverables folder; `ArtifactMiddleware` sweeps it after
  every tool call and pushes bytes through the `ui` state key (components in `ui/ui.tsx`).
  The prompt forbids `read_file` on anything in `out/` — reading a PNG back cost more
  context than an entire run.
- `/workspace/uploads/` (`paths.UPLOAD_DIR`) is the reverse trade: `UploadMiddleware` strips
  each attachment out of the human message before the first model call, so the bytes sit in one
  checkpoint and in no model request, and the prompt gets a manifest — names, sizes, row counts,
  column names. Not under `out/`, which is swept and published: a file the user gave us would
  come back as a deliverable of their own question.

`middleware/progress.py` is the run's only visible output while it works. Everything happens
inside one `eval`, so the transcript shows a single unreturned tool call for minutes; the
wrapped source tools emit a line each over LangGraph's `custom` stream mode, which the chat UI
renders beside the typing dots. Wrappers rather than a middleware hook because the calls are
made inside QuickJS, below the tool node, where no hook can see them.

`middleware/uploads.py` is the one place where **the sandbox is not the storage**. A container is
reaped after `IDLE_TTL_SECONDS` and the thread gets an empty replacement, so the durable copy
lives in the LangGraph store, per thread, and `before_agent` re-materialises it every turn —
reconciled by name and size, so a warm container costs one `ls`. Not graph state (re-serialised
into every checkpoint, the cost `MAX_INLINE_BYTES` bounds) and not model context (a 2k-row CSV
re-sent every turn, unusable without retyping it into `writeFile`). Without a store — the CLI —
turn 2 says the file is gone rather than pretending. A declined attachment (`.xls`, oversize) is
stripped too and replaced by the reason: left in, the block 400s the whole run at a provider
with no such document type.

### Subagents

Four leaves in `prompts/subagents.py`, wired in `agent.py` by `analyst_leaf()`. deepagents'
default is that subagents **inherit the parent's tools**, so every leaf sets `tools: []`
explicitly — and because `middleware: []` does *not* mean "no middleware" (deepagents prepends
its own default filesystem, `execute` included), every leaf also passes a
`FilesystemMiddleware` narrowed to `read_file`, the floor it refuses to drop. The two kinds
reach that same restriction from opposite directions:

- `abstract-analyst`, `full-text-analyst` — payload arrives in the prompt, nothing needed on
  disk. Their prompts claim they have no tools; leaving the default toolset in place made
  models invent paths and doubled their latency.
- `figure-analyst` — handed a sandbox path, and `read_file` is what turns it into an image the
  model can see.

The auto-added `general-purpose` subagent *does* inherit the PubMed tools and an
unrestricted filesystem including `execute`. Nothing routes work to it, but don't start.

Root runs the larger model, leaves the cheaper one (`models.py:root_model` /
`subagent_model`). Both defaults are now OpenAI, so nothing needs configuring for caching:
both paths cache server-side on `/v1`, and `AnthropicPromptCachingMiddleware` no-ops for
them (deepagents constructs it with `unsupported_model_behavior="ignore"`).

That stops being true the moment you point either role at an Anthropic model. Whenever the
*leaves* are Anthropic, subagent prompt caching is a net loss: each leaf is a fresh
single-turn agent with a unique payload, so it pays cache-write premium for reads that never
happen. Turn it off on the subagent model, keep it on the root.

### Model gateway

Three roles — `ROOT`, `SUBAGENT`, `JUDGE` — each configured by three independent env
vars, defaulting to the constants in `models.py`:

| | `_MODEL` | `_PROVIDER` | `_EFFORT` |
|---|---|---|---|
| `ROOT` | `openai/gpt-5.6-terra` | `openai` | `low` |
| `SUBAGENT` | `openai/gpt-5.6-luna` | `openai` | `low` |
| `JUDGE` | `openai/gpt-5.6-luna` | `openai` | `low` |

All nine sit in one block at the top of `models.py` rather than beside the notes that
justify them, so the configuration is readable without reading the rationale.

Three axes rather than one named profile because they vary independently: a root swap, a
leaf swap and a thinking level are three different experiments, and a name covering
combinations needs an entry per combination. `ROOT_EFFORT` always sat outside the naming
for that reason.

`_EFFORT` unset means different things per path. On an Anthropic model it is **not**
`effort=high`, it is no thinking at all — langchain-anthropic defaults `thinking` to adaptive
whenever effort is set, so setting it turns thinking on and `root_context_chars` moves with
it; on an OpenAI model, unset just takes the provider's own default. Both defaults now set
an effort explicitly, so this matters on a swap rather than out of the box — and one swap in
particular: Haiku 4.5 has no effort scale, so `SUBAGENT_MODEL=claude-haiku-4-5-20251001`
must also clear `SUBAGENT_EFFORT=` or the gateway answers with a 400.

The gateway path is picked per **model**, not globally — that's what lets one run mix
providers across roles. The difference is not cosmetic: Anthropic models must use the
**native** `/anthropic/v1/messages` path or prompt caching silently stops working (the
OpenAI-compatible shim drops `cache_control`). On the native path the base URL must **not**
end in `/v1` (the SDK appends it) and model ids are bare (`claude-sonnet-5`, not
`anthropic/...`). OpenAI models use `/v1` and cache server-side automatically.

Each role's path is stated outright rather than inferred, and a default path belongs to
the default model beside it — replace only `ROOT_MODEL` and the path comes from the new
id's form instead (bare means native, prefixed means `/v1`), which is what keeps
`ROOT_MODEL=claude-sonnet-5` a one-variable swap (bare, so it goes native and keeps
caching, without `ROOT_PROVIDER` being named). Naming `_PROVIDER` explicitly is
the escape hatch for an id in neither form, so a new model needs no code edit; name one
the id's form contradicts and `_provider_for` raises rather than sending it down the wrong
path, which the gateway answers with a 501 that looks like an outage.

## Invariants worth preserving

- **Subagents do no I/O.** NCBI allows 3 req/sec (10 with a key); N subagents each
  fetching would collect 429s. Batch-fetch up front is what makes a large fan-out safe.
- **ClinicalTrials.gov is the tightest rate limit in the repo, and it is undocumented.**
  Measured at roughly a 10-token bucket refilling at ~1 req/sec — 12 concurrent requests
  returned ten 429s — and **the 429 carries no `Retry-After`**, so client-side backoff is the
  only thing between a fan-out and a dead run. No API key raises it.
  `ctgov.py`'s `Throttle` (from `sources/_http.py`) serialises every request in the
  process, which is why that
  module needs no concurrency semaphore the way `pmc.py` does. The design response is to make
  per-trial fetching unnecessary rather than merely discouraged: `pageSize` reaches 1,000 and
  `filter.ids` takes 300, so a 1,000-trial corpus is four requests.
- **`sources/_http.py` holds the shared pacing/backoff mechanism, but each caller keeps
  its own `Throttle`.** The two APIs meter independently, so a shared `_last_call` would
  make a PubMed search delay a registry fetch for nothing. Each module also keeps its own
  `_request` — the POST branch, 4xx handling and exception types genuinely differ.
  `pmc.py` uses none of it; S3 caps concurrency with a semaphore instead.
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
- **The guards in `sources/pubmed.py` and `sources/pmc.py` are not boilerplate.** Each matches a
  verified API failure mode that returns a *wrong answer rather than an error* — PMID
  tokenization, silent query rewriting, esummary's 500-UID cap returning HTTP 200, unguessable
  PMC object version suffixes. Field tags in particular are validated locally against
  PubMed's documented set, so supporting a new one means extending that list. Don't simplify
  them away: `docs/pubmed_api_notes/`, `docs/pmc_api_notes/` and `docs/ctgov_api_notes/` (all
  gitignored) have the probe results, and the ctgov notes ship a `probe.py` that reproduces
  every measurement, sleeps included.
- **The host cache expires on the same window as the sandbox.** `IDLE_TTL_SECONDS` lives in
  `paths.py` precisely so `sandbox.py` and `sources/cache_io.py` cannot drift apart: past that
  window a returning thread finds neither its container nor its corpus, rather than a warm
  cache pointing into a container that is gone. The TTL is *idle* — a hit calls
  `cache_io.touch`, so a long run keeps its corpus as long as it is working. Expiry only stops
  a stale entry being used; `cache_io.sweep_if_due()` bounds the disk, called at run start by
  each of the three entry points and self-gated to once per TTL window (~11ms for 1800 files).
  `RESEARCH_AGENT_CACHE_TTL=off` restores the old permanent cache, which is what `evals/run.py`
  sets — otherwise whether an example refetches would depend on wall clock, and both latency
  and `from_cache` would move for reasons unrelated to the agent.
- **Blocking calls must go through `asyncio.to_thread`** (`sources/cache_io.py`). Under
  `langgraph dev`, blockbuster turns a blocking `read_text()` in a coroutine into a
  `BlockingError` that kills the run; in production it stalls every other run in the
  process.
- **`ResilientSandbox` retries assume idempotence** (`sandbox.py`). It retries
  transient WebSocket failures below the tool boundary, because one HTTP 502 once
  destroyed a completed 18-way fan-out. A tool whose sandbox command must run exactly
  once has to be routed around it.
- **Artifact components only render same-origin.** `/ui/{graph}` returns a script tag with a
  *host-relative* `src`, so the browser resolves it against the page's origin, not the API's.
  On a cross-origin frontend (hosted Agent Chat, Studio at `smith.langchain.com`) that 404s,
  the `onload` never fires, and `LoadExternalComponent` paints an empty div — no console
  error, the run looks fine, the chart is just absent. Any frontend needs `/ui/*` proxied to
  the agent server. This is frontend wiring, not a bug in `middleware/artifacts.py` or
  `ui/ui.tsx`; check the network tab for `/ui/<graph>/entrypoint.js` before touching either.
- **Prompt changes are the main tuning lever.** One line telling the model to print
  numbers instead of reading its plot back cut root context from 115k to 31k chars. Treat
  `prompts/` as production code.

## Writing prompts

Agent prompts should be as concise and general as possible. Avoid unnecessary detail and exposition. Unless critical, avoid giving the agent explicit instructions or step-by-step procedures; instead, simply document potential pitfalls or things it should know.