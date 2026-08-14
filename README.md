# PubMed research assistant

A [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) demo for life
sciences: searches PubMed, retrieves abstracts and PMC full text, fans out subagents to
answer a question across many papers at once, and computes and plots in a sandboxed
Python container. See `docs/concept.md` for the design rationale.

## Setup

1. Fill in `.env` (see `.env.example` for the shape):

   ```
   OPENAI_API_KEY=lsv2_sk_...     # LangSmith gateway service key, not an OpenAI key
   LANGSMITH_GATEWAY_ANTHROPIC_URL=https://gateway.smith.langchain.com/anthropic

   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=science-agent

   NCBI_API_KEY=          # optional: raises the rate limit from 3 to 10 req/sec
   NCBI_TOOL=pubmed_agent
   NCBI_EMAIL=you@example.com
   ```

   Models are called through the **LangSmith LLM gateway**. Gateway compute is
   authenticated by `OPENAI_API_KEY` (the `lsv2_sk_...` service key); `LANGSMITH_API_KEY`
   is left to tracing.

   **Switching model profiles** — `MODEL_PROFILE` selects the pair:

   ```bash
   uv run agent                        # anthropic: sonnet-4.6 + haiku-4.5 (default)
   MODEL_PROFILE=mixed uv run agent    # mixed: gpt-5.6-terra + haiku-4.5
   MODEL_PROFILE=openai uv run agent   # openai: gpt-5.6-terra + gpt-5.6-luna
   ```

   Add profiles in `research_agent/models.py`. The gateway path is chosen per **model id**, not per
   profile, so a profile may mix providers — which is what `mixed` does: an OpenAI root
   over Anthropic leaves. Bare ids (`claude-sonnet-4-6`) take the Anthropic-native path,
   provider-prefixed ids (`openai/gpt-5.6-terra`) take the OpenAI-compatible one.

   For Anthropic models `research_agent/models.py` uses the gateway's **Anthropic-native** path, not
   its OpenAI-compatible one, because **their prompt caching only survives on the native
   path** — verified against the live gateway:

   | path | Anthropic model, repeated 14k-token prefix |
   |---|---|
   | `/v1/chat/completions` (OpenAI-compatible) | `cached_tokens: 0`, with *and* without an explicit `cache_control` block — the gateway drops it |
   | `/anthropic/v1/messages` (native) | `cache_creation_input_tokens: 14413`, then `cache_read_input_tokens: 14413` |

   Two quirks of the native path: the base URL must **not** include `/v1` (the Anthropic
   SDK appends it; `/anthropic/v1/v1/messages` 501s), and model ids are **bare**
   (`claude-sonnet-4-6`) rather than provider-prefixed.

   This only affects Anthropic models routed through the OpenAI-compatible shim. Native
   OpenAI models on `/v1` cache automatically and server-side — the `openai` profile
   measured 103k cache reads with no configuration at all.

   `research_agent/cli.py` calls `load_dotenv(override=True)` deliberately — without it, a
   `LANGSMITH_PROJECT` already exported in your shell silently captures this project's
   traces.

2. Run it:

   ```bash
   uv run agent            # one-shot CLI
   ./scripts/dev.sh        # or the chat stack: graph server + UI
   ```

## How it works

The agent does **not** call the PubMed tools directly. It writes JavaScript in a QuickJS
interpreter and reaches them through programmatic tool calling, so a whole workflow —
search, batch-fetch, fan out across 30 abstracts, compute, collect — happens inside one
`eval` call instead of dozens of round trips.

```
user question
   -> eval (JS)
        tools.pubmedSearch()      esearch + esummary
        tools.fetchAbstracts()    efetch, batched, host-side disk cache
        tools.writeFile()         materialise records into the sandbox
        Promise.all(task(...))    one abstract-analyst subagent per paper
        tools.execute()           Python in the sandbox: stats, tables, plots
   -> synthesis with PMID citations
```

Subagents receive abstract text in their prompt and do no I/O of their own. That's
deliberate: NCBI allows 3 requests/sec, so N subagents each fetching their own abstract
would collect HTTP 429s. Fetching in batches up front is what makes the fan-out safe.

### Two code surfaces

`eval` (QuickJS) is orchestration only — no network, filesystem, or shell of its own.
For quantitative work the agent needs real Python, so the agent's **backend** is a
LangSmith sandbox: an isolated Linux container with numpy, pandas, scipy and matplotlib,
exposed as an `execute` shell tool. Both surfaces are in the PTC allowlist, so one `eval`
call can run search → fetch → write → compute → collect.

The sandbox is created per run and deleted when the run ends (with a 10-minute idle TTL
as a server-side backstop, since a `finally` doesn't survive `kill -9`). It starts empty:
**PubMed data does not appear in it by itself — the agent writes it there** with
`tools.writeFile`. That's free in tokens, because PTC tool output is marshalled into the
JS heap and never enters the model's context.

Booting a bare sandbox and `pip install`ing the stack costs ~30s per run. Bake it into a
snapshot once instead:

```bash
uv run scripts/build_snapshot.py     # ~35s, once
```

That freezes a sandbox with the libraries and `/workspace/out/` already in place; runs
then start in ~1-3s. `research_agent/sandbox.py` looks for the snapshot named by
`SANDBOX_SNAPSHOT_NAME`
(default `pubmed-py`) and falls back to installing at runtime if it isn't there, so a
fresh clone works without the build step — just slower.

## Files

```
research_agent/          the agent itself
├── agent.py             assembles it: tools, subagents, middleware
├── cli.py               one-shot CLI entry point (`uv run agent`)
├── graph.py             LangGraph server entry point, sandbox keyed to thread_id
├── runner.py            one question -> one RunResult; the seam evals score against
├── sandbox.py           sandbox lifecycle + WebSocket retry wrapper
├── models.py            gateway-backed model construction, per-id provider routing
├── paths.py             every host-side path, anchored to the repo root
├── prompts/             system.py (root prompt + JS snippets), subagents.py (the leaves)
├── sources/             pubmed.py, pmc.py (API clients + tools), cache_io.py
└── middleware/          artifacts.py (sweeps /workspace/out), perf.py (event-loop lag)

evals/                   LangSmith datasets + evaluators
scripts/                 dev.sh, build_snapshot.py
ui/                      ui.tsx artifact components, rendered by the chat frontend
docs/                    design notes, demo questions, per-endpoint API notes
data/                    abstract/search/PMC cache (gitignored)
```

`data/` is **host-side only — the agent never sees it.** The sandbox starts empty and the
agent writes what it needs there with `tools.writeFile`. Override its location with
`RESEARCH_AGENT_DATA_DIR`.

Root model is Sonnet; the per-abstract analyst subagent runs on Haiku. The fan-out is
where the token volume is, so that split is most of the cost story.

## Measured on a real run

**These numbers predate the sandbox.** They were measured with a host-rooted filesystem
backend and no `execute` tool, so the prompt was shorter and no run spent time booting a
container. The profile comparison still holds directionally; the absolute figures will
have moved. For sandbox-era figures see [After the sandbox](#after-the-sandbox) below.

The demo question: *"recent papers on base editing in the liver — which used in vivo
mouse models?"*, one run per profile.

| | `anthropic` | `openai` |
|---|---|---|
| root / subagent | sonnet-4.6 / haiku-4.5 | gpt-5.6-terra / gpt-5.6-luna |
| papers analysed | 22 | **49** |
| wall clock | 107s | **46s** |
| cost | $0.32 | **$0.083** |
| **per paper** | $0.0146 · 4.9s | **$0.0017 · 0.9s** |
| root turns | 6 | **2** |
| subagent cache reads | 0 of 96k input | **103k of 129k input** |

The `openai` profile did **more than twice the work for a quarter of the cost in half
the time**. Per paper that's ~8× cheaper and ~5× faster. Two things drive it:

1. **Root turns: 2 vs 6.** terra planned, searched, fetched, fanned out and synthesised
   in two turns. Sonnet took six, and root occupancy is where the latency lives — 54% of
   wall for terra vs 79% for Sonnet.
2. **Subagent caching actually works.** luna read 103k of its 129k input from cache.
   Haiku read *zero* and wrote cache on nearly every token — see the caching note below.

Caveats: one run each, and the agent chose its own `retmax`, so the paper counts differ
(49 vs 22) — this is a directional comparison, not a controlled benchmark. Both runs hit
a warm abstract cache, so PubMed time was negligible in both. Output quality wasn't
graded, though the GPT-5.6 run did separate out rat/organoid/human studies rather than
lumping them in.

### Where the time and cost go

Both profiles show the same shape: **the root agent dominates, not the fan-out.**

| | root share of wall | root share of cost |
|---|---|---|
| anthropic | 79% | 58% |
| openai | 54% | 80% |

Under `anthropic`, 22 Haiku subagents cost $0.134 against the root's $0.187 — and 40.6s
of subagent compute compressed into 14.2s of wall time. Under `openai`, 49 luna
subagents cost $0.017 total. The cheap-model-on-leaves split works in both; the expense
is the orchestrator re-reading a growing transcript.

Input dominates output roughly 3:1 overall.

### A caching trap worth knowing

`AnthropicPromptCachingMiddleware` applies `cache_control` to subagent prompts too, but
each subagent is a fresh single-turn agent holding a unique abstract — there is nothing
for a later call to reuse. Measured: Haiku wrote 96,019 cache tokens and read **0**,
paying the cache-write premium for a cache that never hits. The root, by contrast, got
76% of its input from cache and that's the whole saving.

This is a property of the *leaves*, not the profile, so it applies to any profile with
Anthropic subagents — `anthropic` and `mixed` both. On either, disable caching on the
subagent model and keep it on the root. The `openai` profile sidesteps it entirely: its
caching is automatic and server-side, so the leaves benefit instead of being penalised.

Under `mixed` the root escapes the trap for a different reason — an OpenAI root gets
server-side caching on `/v1` and `AnthropicPromptCachingMiddleware` no-ops for it
(deepagents constructs it with `unsupported_model_behavior="ignore"`), so nothing needs
configuring at the root either.

The one-search/one-fetch count holds in both: subagents never touch the network, so a
run costs two HTTP requests regardless of how many papers are analysed.

### After the sandbox

Three runs on the `anthropic` profile with the computational demo question (*"...then use
Python to plot the distribution of publication years and report the median year"*),
measured off the returned root message list rather than the trace UI:

| | run 1 | run 2 | run 3 (after prompt fix) |
|---|---|---|---|
| wall clock | 214s | 162s | **145s** |
| root messages | 34 | 24 | **16** |
| root AI turns | 17 | 12 | **8** |
| root context | — | 115,097 chars | **31,034 chars** |
| largest single message | 121,047 | 91,587 | **12,941** |

Runs 1 and 2 each had one enormous message, and both were the same thing: the agent
calling `read_file` on the PNG it had just drawn, which returns base64 and cost more
context than the entire rest of the run. The system prompt now tells it to report the
path and print the underlying numbers instead — that one line is the whole difference
between run 2 and run 3.

Two things verified directly rather than assumed:

- **No abstract text reaches the root.** Sampling 40 verbatim mid-abstract fragments
  from the on-disk cache and searching the root message list finds **0**. The fan-out
  isolation the design depends on holds under the sandbox.
- **`abstract-analyst` really has no tools.** Its resolved spec is `tools: []` with a
  `FilesystemMiddleware` restricted to `read_file` — no `execute`, no PubMed tools.
  Without that, deepagents' inherit-parent-tools default (`graph.py`) would hand every
  analyst a shell into the shared container.

One caveat worth knowing: the auto-added `general-purpose` subagent *does* inherit the
PubMed tools and an unrestricted filesystem including `execute`. That's stock deepagents
behaviour and nothing in the prompt routes work to it, but it's a path to a shell if you
start dispatching to it.

Sandbox boot from the snapshot was 2.3-2.9s across these runs, against ~30s for a bare
sandbox plus `pip install`.

## The tools

**`pubmed_search(term, retmax, sort, mindate, maxdate)`** — esearch then esummary.
Returns records with `pmid/title/first_author/last_author/year/journal/doi`, plus
`query_translation` and `warnings`. Large result sets are also dumped to
`data/searches/`; `saved_to_host` names that file, which is an operator-side archive the
agent's sandbox cannot open.

**`fetch_abstracts(pmids)`** — batched efetch, XML parsed, one JSON per PMID cached
under `data/abstracts/`. Returns `records/missing/invalid/from_cache`. Structured
abstracts keep their section labels; retracted papers are flagged.

**`pmc_locate(pmcids)`** — the cheap triage step, and the one to call before any full-text
retrieval. Reports which papers are in PMC's open-access bucket and what each contains —
section titles, figure labels and captions, table captions, supplementary file names —
plus the character cost of the body text, *without* returning the body text.

**`fetch_full_text(pmcids, sections, include_captions, include_tables)`** — the body text,
optionally narrowed to named sections. The median paper is ~40k characters, so this is
meant to be handed to a `full-text-analyst` subagent, never read by the root model.

**`fetch_figures(pmcid, files)`** / **`fetch_supplementary(pmcid, files)`** — download into
the sandbox and return paths. These are built per run against the live backend, because
an image has to exist as a real file before `figure-analyst` can `read_file` it and see it.

## A note on the guards in `sources/pubmed.py`

The defensive code there isn't boilerplate — the PubMed API has three failure modes that
produce **wrong answers rather than errors**, all verified against the live API:

1. **Malformed PMIDs return unrelated papers.** efetch tokenizes garbage and returns
   whatever numbers fall out: `42.9` yields PMIDs 42 *and* 9, `-5` yields PMID 5, all
   HTTP 200. Ids are validated against `^\d+$` before any request goes out.
2. **Broken queries are silently rewritten.** `cancer[nosuchfield]` returns 5,675,880
   hits — the unknown tag is dropped and the search runs across every field. Nothing in
   the response reports this; `errorlist.fieldsnotfound` is empty. So field tags are
   validated locally against PubMed's documented set before searching.
3. **esummary's 500-UID cap returns HTTP 200 with no results.** The body carries an
   `error` key and no `result` block, so a naive read looks like zero hits. Requests are
   chunked and the error key is checked first.

`docs/pubmed_api_notes/` has the full writeup including the probe results behind each claim.

## Not included

No web search beyond PubMed, and no literature source beyond it — full text comes from
PMC's open-access subset only, so paywalled papers stop at the abstract. No skills.

The sandbox is Python plus the standard scientific stack only — no genomics binaries
(PLINK, bcftools) and no data sources beyond PubMed. Nothing persists between runs.

The interpreter and dynamic subagents are both **beta**; APIs may change between
releases.
