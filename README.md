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

   `models.py` uses the gateway's **Anthropic-native** path, not its OpenAI-compatible
   one, because **prompt caching only survives on the native path** — verified against
   the live gateway:

   | path | repeated 14k-token prefix |
   |---|---|
   | `/v1/chat/completions` (OpenAI-compatible) | `cached_tokens: 0`, with *and* without an explicit `cache_control` block — the gateway drops it |
   | `/anthropic/v1/messages` (native) | `cache_creation_input_tokens: 14413`, then `cache_read_input_tokens: 14413` |

   Two quirks of the native path: the base URL must **not** include `/v1` (the Anthropic
   SDK appends it; `/anthropic/v1/v1/messages` 501s), and model ids are **bare**
   (`claude-sonnet-4-6`) rather than provider-prefixed.

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

The demo question ("recent papers on base editing in the liver — which used in vivo
mouse models?"), before and after moving to the caching-capable gateway path:

| | OpenAI-compat path | Anthropic-native path |
|---|---|---|
| papers analysed | 24 | 22 |
| wall clock | 80s | 107s |
| cost | $0.42 | **$0.32** |
| input tokens | 155k | 204k |
| **cache reads** | **0** | **81,885** |
| PubMed calls | 2 | 2 |

Caching cuts cost ~23% even though the second run pushed *more* input through. It did
not help latency — these two runs aren't a controlled comparison (different paper
counts), so treat the wall-clock delta as noise rather than a regression.

**Where the time goes** (from the first run's trace): 86% is the root model — six
sequential Sonnet turns totalling 68.5s, of which the final synthesis alone is 28.9s for
1,499 output tokens. The actual work is ~16s: PubMed is 1.5s for both calls, and the 24
Haiku subagents are 40.6s of compute compressed into 14.2s wall. If you want it faster,
the levers are fewer root turns and a shorter final answer — not the tools.

**Where the cost goes**: 82% is the root agent, not the fan-out. 24 Haiku subagents came
to $0.076. The cheap-model-on-leaves split works; the expense is the orchestrator
re-reading a growing transcript, which is exactly what caching addresses.

The one-search/one-fetch count is the design working: subagents never touch the network,
so a run costs two HTTP requests regardless of how many papers are analysed.

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
