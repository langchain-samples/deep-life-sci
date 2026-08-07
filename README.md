# PubMed research assistant

A [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) demo for life
sciences: searches PubMed, retrieves abstracts, and fans out subagents to answer a
question across many papers at once. See `concept.md` for the design rationale.

## Setup

1. Fill in `.env` (see `.env.example` for the shape):

   ```
   OPENAI_API_KEY=lsv2_sk_...     # LangSmith gateway service key, not an OpenAI key
   LANGSMITH_GATEWAY_ANTHROPIC_URL=https://gateway.smith.langchain.com/anthropic

   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=deepagents_testing

   NCBI_API_KEY=          # optional: raises the rate limit from 3 to 10 req/sec
   NCBI_TOOL=deepagents_demo
   NCBI_EMAIL=you@example.com
   ```

   Models are called through the **LangSmith LLM gateway**. Gateway compute is
   authenticated by `OPENAI_API_KEY` (the `lsv2_sk_...` service key); `LANGSMITH_API_KEY`
   is left to tracing.

   **Switching model profiles** — `MODEL_PROFILE` selects the pair:

   ```bash
   uv run agent.py                        # anthropic: sonnet-4.6 + haiku-4.5 (default)
   MODEL_PROFILE=openai uv run agent.py   # openai: gpt-5.6-terra + gpt-5.6-luna
   ```

   Add profiles in `models.py`. Each picks its own gateway path automatically.

   For Anthropic models `models.py` uses the gateway's **Anthropic-native** path, not
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

   `agent.py` calls `load_dotenv(override=True)` deliberately — without it, a
   `LANGSMITH_PROJECT` already exported in your shell silently captures this project's
   traces.

2. Run it:

   ```bash
   uv run agent.py
   ```

## How it works

The agent does **not** call the PubMed tools directly. It writes JavaScript in a QuickJS
interpreter and reaches them through programmatic tool calling, so a whole workflow —
search, batch-fetch, fan out across 30 abstracts, collect — happens inside one `eval`
call instead of dozens of round trips.

```
user question
   -> eval (JS)
        tools.pubmedSearch()      esearch + esummary
        tools.fetchAbstracts()    efetch, batched, disk-cached
        Promise.all(task(...))    one abstract-analyst subagent per paper
   -> synthesis with PMID citations
```

Subagents receive abstract text in their prompt and do no I/O of their own. That's
deliberate: NCBI allows 3 requests/sec, so N subagents each fetching their own abstract
would collect HTTP 429s. Fetching in batches up front is what makes the fan-out safe.

## Files

| | |
|---|---|
| `agent.py` | assembles the agent; `__main__` runs a demo question |
| `pubmed.py` | E-utilities client + the two tools |
| `prompts.py` | system prompt with reference JS snippets, and the subagent definition |
| `models.py` | gateway-backed model construction |
| `pubmed_api_notes/` | per-endpoint API notes (gitignored) |
| `data/` | abstract cache and search dumps (gitignored) |

Root model is Sonnet; the per-abstract analyst subagent runs on Haiku. The fan-out is
where the token volume is, so that split is most of the cost story.

## Measured on a real run

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

If you stay on `anthropic`, disable caching on the subagent model and keep it on the
root. The `openai` profile sidesteps this — its caching is automatic and server-side, so
the leaves benefit instead of being penalised.

The one-search/one-fetch count holds in both: subagents never touch the network, so a
run costs two HTTP requests regardless of how many papers are analysed.

## The tools

**`pubmed_search(term, retmax, sort, mindate, maxdate)`** — esearch then esummary.
Returns records with `pmid/title/first_author/last_author/year/journal/doi`, plus
`query_translation` and `warnings`.

**`fetch_abstracts(pmids)`** — batched efetch, XML parsed, one JSON per PMID cached
under `data/abstracts/`. Returns `records/missing/invalid/from_cache`. Structured
abstracts keep their section labels; retracted papers are flagged.

## A note on the guards in `pubmed.py`

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

`pubmed_api_notes/` has the full writeup including the probe results behind each claim.

## Not included

Abstracts only — no full text or PMC retrieval. No skills (`skills/example-skill/` is
still the placeholder). No web search beyond PubMed. No UI.

The interpreter and dynamic subagents are both **beta**; APIs may change between
releases.
