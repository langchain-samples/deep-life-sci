# Evals

Scores the agent against LangSmith datasets. Two commands:

```bash
uv run python -m evals.sync          # push datasets/*.jsonl to LangSmith (idempotent)
uv run python -m evals.run           # score the agent against them
uv run python -m evals.run --structural --limit 3    # fast, no judge model
```

## Why the dataset lives in git

`datasets/core.jsonl` is the source of truth; LangSmith is a mirror of it. A dataset
edited only in the web UI has no diff, no review, and no way to answer "what changed
between the run that scored 0.8 and the run that scored 0.6". `sync.py` matches on the
`id` field, so rewording a question updates that example rather than duplicating it and
its scoring history stays attached.

Nothing is ever deleted by sync — an example dropped from the JSONL is reported and left
in place, because removing it would orphan every experiment that scored it.

## What each evaluator is for

| evaluator | reads | catches |
|---|---|---|
| `citations_exist` | answer text + `data/abstracts/` | invented PMIDs |
| `produced_expected_artifacts` | `ui` state key | "plot the distribution" answered in prose |
| `used_code_orchestration` | root tool calls | the agent stopped fanning out through `eval` |
| `within_turn_budget` | root turn count | turn-count creep, where the latency lives |
| `root_context_budget` | root transcript size | payload leaking into root context |
| `rubric_judge` | answer text | confident, fluent, wrong — missing denominators, silent omissions |

The first five are deterministic and free. Only `rubric_judge` costs a model call, and it
runs on the cheap half of the model pair. `--structural` drops it.

**The three middle evaluators are why `runner.py` exists.** Trajectory, artifacts and
context size are not recoverable from the answer text, and they are where this agent's
regressions actually show up — a run that quietly stops fanning out still produces a good
answer, slowly and expensively, and says nothing about it.

## Scoring conventions

`score: None` means *not applicable*, not zero. A question with no required artifact, or
an answer that legitimately cites no papers, is excluded from that evaluator's aggregate
rather than given a free 1.0 that inflates every experiment. The judge also returns `None`
when its own output won't parse, because a judge failure and a bad answer must not look
the same in the numbers.

Budgets are per-example, in the seed file. A metadata-only question ("which journals
publish the most cryo-EM work") should finish in far fewer turns and far less context than
a 30-paper fan-out, and one global threshold would be wrong for both.

## Isolation and cost

Every example gets its own sandbox. That is the expensive choice, taken because
`produced_expected_artifacts` reads the artifact sweep — a shared container would let one
example's leftover `/workspace/out` files be attributed to the next.

Concurrency defaults to 3. Each example already fans out to dozens of subagents
internally, so raising it multiplies containers rather than throughput, and NCBI's rate
limit is shared across all of them.

## What isn't covered

- **Misattribution.** `citations_exist` proves a PMID was fetched, not that it supports
  the claim it's attached to. The cache is also shared across every run on the machine,
  so a paper fetched by an earlier run passes. That gap is the judge's job.
- **Retrieval quality.** Nothing scores whether the search returned the *right* corpus,
  only what the agent did with what it got.
- **Reference answers.** The seeds carry rubrics, not gold answers. Literature questions
  don't have stable ground truth — the corpus moves — so the rubrics encode process
  standards ("report the denominator") rather than expected findings.
