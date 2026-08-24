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
| `anthropic` | `ROOT_MODEL=claude-sonnet-4-6 SUBAGENT_MODEL=claude-haiku-4-5-20251001 SUBAGENT_EFFORT=` |
| `mixed` | `SUBAGENT_MODEL=claude-haiku-4-5-20251001 SUBAGENT_EFFORT=` — terra root over Haiku leaves |
| `openai` | *the current default* — terra root over luna leaves, both at `low` |

Note the `SUBAGENT_EFFORT=` in the first two: the leaves now default to an explicit `low`,
and Haiku 4.5 has no effort scale, so a swap that only names the model gets a gateway 400.

The default root is `openai/gpt-5.6-terra`; see
[Scoring the root pair](#scoring-the-root-pair) for the run that chose it over
`claude-sonnet-5`, which none of the figures below were measured against either.

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

## Scoring the root pair

Two eval sweeps over the 11-example `pubmed-agent-default` dataset, run back-to-back on
2026-08-23, varying only the root: `claude-sonnet-5` against `openai/gpt-5.6-terra`, both
at `ROOT_EFFORT=medium`, both over the default Haiku leaves and the pinned luna judge, one
example at a time.

| | sonnet-5 | terra |
|---|---|---|
| rubric | 7/11 | 7/11 |
| citations_exist | 8/9 applicable | 8/9 applicable |
| produced_expected_artifacts | 2/2 applicable | **1/2** |
| wall clock, 11 examples | 16m23s | 10m53s |

Experiments `pubmed-claude-sonnet-5-medium-eb50afd6` and
`pubmed-gpt-5.6-terra-medium-41f78442` in LangSmith.

What made terra the default is the tie, not a win. The two roots failed the **same four
rubric seeds** — `base-editing-t-cells-convergence`, `fmt-cdiff-placebo-trials`,
`psilocybin-depression-unpublished`, `semaglutide-weightloss-boxplot` — and the citation
difference was a swap rather than a gain, sonnet missing base-editing where terra missed
`egan-ulk1-ampk-sites`. Four seeds failing under both roots is evidence about those
questions, the rubric or the shared leaves, not about root capability; they are the first
place to look before reading any of this as a model ranking. With quality tied, the
per-paper cost figures above decide it.

Two caveats on these numbers, both of which flatter terra:

- **n=1 per seed, binary rubric.** One seed flipping moves a column by 9 points, and every
  difference here is a single seed.
- **The second run read a warm cache.** `evals/run.py` pins `RESEARCH_AGENT_CACHE_TTL=off`
  against a shared `data/`, so run 2 reused the corpus run 1 fetched. That is most of why
  the wall-clock gap is as wide as it is; treat it as indicative, not as a latency
  measurement.

`root_context_chars` was not captured for either sweep — it is on each `RunResult` in the
traces. That is the number to compare next, and it matters most on a swap *back* to an
Anthropic root, where `ROOT_EFFORT=medium` turns thinking on and moves it.

## Scoring the leaves

Three sweeps over the same 11-example dataset on 2026-08-24, notes off
(`RESEARCH_AGENT_NOTES=0`), judge pinned at luna/low, run after the `_esummary` and
`fetch_abstracts` concurrency work landed. This is what moved the leaves off Haiku 4.5.

| | terra-low / haiku-4.5 | terra-medium / haiku-4.5 | terra-low / luna-low |
|---|---|---|---|
| rubric | 7/11 | 7/11 | 7/11 |
| citations | 9/9 | 8/9 | 9/9 |
| artifacts | 1/2 | 2/2 | 1/2 |
| median run latency | 31.0s | 48.2s | 36.2s |
| tokens | 2,101,686 | 2,034,659 | 1,650,743 |
| cost | $1.54 | $1.38 | **$0.88** |
| error rate | 0 | 0 | 0 |

**luna-low scored identically to Haiku 4.5 cell for cell** — every seed, all three
evaluators — for 43% less on 21% fewer tokens, at +5s median latency. That is the whole
argument for the swap; there is no quality column to trade against.

**Root effort at medium bought nothing.** Same 7/11, failing the same four seeds, 53% more
latency per run, and it *lost* a citation on `psilocybin-depression-unpublished`. It fixed
exactly one cell: the missing `tpd-publication-volume` deliverable. Hence `ROOT_EFFORT=low`
as the default.

Two caveats on the $0.88, since it is the headline:

- **n=1 per configuration, 11 examples, no repeats.** Token counts vary run to run. Read
  43% as a direction, not a constant.
- **The leaves are now the same model and effort as the judge.** Not self-grading — the
  judge scores the root's final answer, never leaf output — but a confound worth retiring
  with a sweep on a different judge before leaning on the number.

The more useful finding is what did *not* move: **the same four rubric seeds fail in all
three configurations** (`base-editing-t-cells-convergence`, `fmt-cdiff-placebo-trials`,
`psilocybin-depression-unpublished`, `semaglutide-weightloss-boxplot`). Root effort did not
touch them and neither did swapping the leaf model across providers, so they are a prompt,
tool or criteria problem rather than a model-selection one.

Every latency figure elsewhere in this document predates the swap and was taken against
Haiku leaves; the profile table above has the one-variable restore.

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
