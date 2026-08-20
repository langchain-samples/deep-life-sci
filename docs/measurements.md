# Measured numbers

The figures behind the architectural choices in this repo — which models to run,
why the leaves are cheap, why prompt caching is configured the way it is, and why the
system prompt forbids reading artifacts back. All of it was measured against the live
gateway and the live NCBI APIs, not estimated.

Read this with `docs/concept.md` before changing the shape of the agent. `CLAUDE.md`
states the rules these numbers produced; this file is the evidence.

## Profile comparison

**The profile names below are historical.** `models.py` no longer has a profile enum —
each role is configured by `{ROOT,SUBAGENT,JUDGE}_{MODEL,PROVIDER,EFFORT}` instead. The
pairs these runs used, as env:

| was | now |
|---|---|
| `anthropic` | `ROOT_MODEL=claude-sonnet-4-6` (leaves already default to Haiku 4.5) |
| `mixed` | `ROOT_MODEL=openai/gpt-5.6-terra` |
| `openai` | `ROOT_MODEL=openai/gpt-5.6-terra SUBAGENT_MODEL=openai/gpt-5.6-luna` |

The current default is `claude-sonnet-5` over those same Haiku leaves, which no profile
ever named and which none of the figures below were measured against.

**These numbers predate the sandbox.** They were measured with a host-rooted filesystem
backend and no `execute` tool, so the prompt was shorter and no run spent time booting a
container. The comparison still holds directionally; the absolute figures will have
moved. See [After the sandbox](#after-the-sandbox) for sandbox-era figures.

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
   Haiku read *zero* and wrote cache on nearly every token — see
   [the caching trap](#the-subagent-caching-trap).

Caveats: one run each, and the agent chose its own `retmax`, so the paper counts differ
(49 vs 22) — this is a directional comparison, not a controlled benchmark. Both runs hit
a warm abstract cache, so PubMed time was negligible in both. Output quality wasn't
graded, though the GPT-5.6 run did separate out rat/organoid/human studies rather than
lumping them in.

## Where the time and cost go

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

The one-search/one-fetch count holds in both: subagents never touch the network, so a
run costs two HTTP requests regardless of how many papers are analysed.

## Prompt caching on the gateway

For Anthropic models `research_agent/models.py` uses the gateway's **Anthropic-native**
path rather than its OpenAI-compatible one, because their prompt caching only survives on
the native path:

| path | Anthropic model, repeated 14k-token prefix |
|---|---|
| `/v1/chat/completions` (OpenAI-compatible) | `cached_tokens: 0`, with *and* without an explicit `cache_control` block — the gateway drops it |
| `/anthropic/v1/messages` (native) | `cache_creation_input_tokens: 14413`, then `cache_read_input_tokens: 14413` |

This only affects Anthropic models routed through the OpenAI-compatible shim. Native
OpenAI models on `/v1` cache automatically and server-side — the `openai` profile
measured 103k cache reads with no configuration at all.

## The subagent caching trap

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

## After the sandbox

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
  Without that, deepagents' inherit-parent-tools default would hand every analyst a
  shell into the shared container.

Sandbox boot from the snapshot was 2.3-2.9s across these runs, against ~30s for a bare
sandbox plus `pip install`. That ~30s was measured against the original package set
(numpy/pandas/scipy/matplotlib and the Office writers); the current list adds rdkit and
installs in ~95s, so the penalty for skipping the snapshot is now roughly 3x what is
recorded here. The snapshot boot itself is unchanged — it does not depend on how much
was baked in.
