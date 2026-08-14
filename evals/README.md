# Evals

Scores the agent against LangSmith datasets. Two commands:

```bash
uv run python -m evals.sync          # push datasets/*.yaml to LangSmith (idempotent)
uv run python -m evals.run           # score the agent against them
uv run python -m evals.run --structural --limit 3    # fast, no judge model
```

## Why the dataset lives in git

`datasets/default.yaml` is the source of truth; LangSmith is a mirror of it. A dataset
edited only in the web UI has no diff, no review, and no way to answer "what changed
between the run that scored 0.8 and the run that scored 0.6". `sync.py` matches on the
`id` field, so rewording a question updates that example rather than duplicating it and
its scoring history stays attached.

Nothing is ever deleted by sync — an example dropped from the seed file is reported and
left in place, because removing it would orphan every experiment that scored it.

YAML rather than JSONL because the rubrics are the part that actually gets revised, and
they are paragraphs and tables. One JSON line per example put each rubric on a single
unwrappable line, so a one-number correction showed up as a whole-line diff. Block
scalars (`|-`) keep them readable in the file and byte-exact in what the judge is sent.

## What each evaluator is for

| evaluator | reads | catches |
|---|---|---|
| `citations_exist` | answer text + `data/abstracts/` | invented PMIDs |
| `produced_expected_artifacts` | `ui` state key | "plot the distribution" answered in prose |
| `rubric_judge` | answer text | confident, fluent, wrong — missing denominators, silent omissions |

The first two are deterministic and free. Only `rubric_judge` costs a model call, and it
runs on the cheap half of the model pair. `--structural` drops it.

**`produced_expected_artifacts` is why `runner.py` exists.** Artifact names are not
recoverable from the answer text, and that's where this agent's regressions actually
show up — a run that quietly stops producing a chart still gives a good answer and says
nothing about it.

## Scoring conventions

**All three evaluators are boolean.** They return `score: True`/`False`. The SDK accepts
that (`SCORE_TYPE` puts `StrictBool` ahead of the numeric types) but LangSmith's score
column is numeric and coerces on ingestion — verified by writing `score=True` and reading
back `1.0`. So the stored metric is a float that only ever takes 0.0 or 1.0, and its mean
is exactly a pass rate. The boolean is the *decision*, not the storage type.

`value:` is the categorical channel if a literal "pass"/"fail" label is ever wanted (it
round-trips as a string), but it carries no number, so the per-column aggregate goes away.
Not worth the trade here.

Booleans because every fraction these could return was misleading. `citations_exist`
requires *all* cited PMIDs to resolve: one invented citation in twenty is the whole point
of the check, and 0.95 sorts next to a clean run while reading as a rounding error. The
judge's rubrics are already written as pass/fail clauses, and a graded score let it split
the difference — 0.75 on a four-clause rubric with no way to tell which clause failed.
What used to be in the fraction is now in `comment`: the missing PMIDs, the absent
artifact kinds, the rubric clause that decided the verdict.

`score: None` still means *not applicable*, not `False`. A question with no required
artifact, or an answer that legitimately cites no papers, is excluded from that
evaluator's aggregate rather than given a free pass that inflates every experiment. The
judge also returns `None` when its own output won't parse — including when it answers with
a number instead of a verdict — because a judge failure and a bad answer must not look the
same in the numbers.

## Isolation and cost

Every example gets its own sandbox. That is the expensive choice, taken because
`produced_expected_artifacts` reads the artifact sweep — a shared container would let one
example's leftover `/workspace/out` files be attributed to the next.

Concurrency defaults to 1. Each example already fans out to dozens of subagents
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
