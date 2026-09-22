# Using LangSmith Engine with Deep Life Sci

A life-science research assistant — PubMed and PMC literature, ClinicalTrials.gov, web
search, specialist subagents, sandboxed analysis — carrying **one deliberately planted
bug**, plus the harness to run the ADLC loop against it: generate traces → let Engine
cluster the failures → **Build** the fix as a PR → **Test** it against a dataset →
**Deploy** it → **Monitor** production for regressions.

This branch (`engine-demo`) exists for the workshop. **Do not merge it.** The bug is the
product here, and `AGENTS.md` warns off any coding agent that would helpfully repair it.

Setup is below; the loop itself starts at
[The loop](#the-loop-build--test--deploy--monitor).

## Files

```
.
├── .github/workflows/
│   └── engine-eval.yml        # PR eval: runs the eval on base vs. head and comments the results
├── engine_workshop/
│   ├── download_traces.py     # Captures a project's runs to traces.json (prunes; see below)
│   ├── upload_traces.py       # Replays saved traces with fresh ids, recent times, ratings
│   ├── traces.json            # The committed reference batch
│   └── eval.py                # LangSmith aevaluate() harness, reads DATASET_NAME
├── evals/datasets/
│   └── engine-workshop.yaml   # Fallback dataset seed, assertions-style
├── scripts/engine_demo.py     # Drives the LIVE agent over the same prompts (results vary)
└── deep_life_sci/             # The agent itself
```

## Stack

- Python 3.12/3.13, managed with [uv](https://docs.astral.sh/uv/)
- A LangChain deep agent with a QuickJS tool-calling interpreter (`deep_life_sci/`)
- LangSmith for tracing, datasets and evals

---

## If you forked and cloned this repo — what to change

### 1. Local environment (`.env`)

`scripts/setup.py` owns `.env` and will prompt for most of this. Copy `.env.example` and
fill in your own values.

| Variable | Required | What to change |
|---|---|---|
| `LANGSMITH_API_KEY` | Yes | **Your** key. Does tracing, datasets and sandboxes. |
| `LANGSMITH_PROJECT` | Yes | **Your** base project. The workshop appends `-engine-demo`. |
| `LANGSMITH_GATEWAY_API_KEY` | To run the agent | Gateway service key. Not needed if you only replay traces. |
| `LANGSMITH_WORKSPACE_ID` | If your key spans workspaces | Otherwise dataset calls 403. |
| `DATASET_NAME` | For `eval.py` | Filled in by setup with the fallback dataset. Change it to use Engine's. |
| `NCBI_*` | To run the agent | `NCBI_TOOL` is generated per install; NCBI meters against it. |

**On the model key:** the recommended way to generate traces is `upload_traces`, which
never calls a model, so you need no gateway key to get through setup and Engine's scan.
You only need one to run the live agent or `eval.py`.

### 2. Project names

`LANGSMITH_PROJECT` is the base; everything here writes to `<base>-engine-demo`, derived in
`engine_workshop/__init__.py:demo_project()` rather than hardcoded, so a shared workspace
where the base is already suffixed per person just works. **Engine is pointed at the
`-engine-demo` project, never at the healthy one.**

The host cache is redirected too, to `data-engine-demo/`. Both redirections happen before
`deep_life_sci` is imported, because `paths.py` resolves `DATA_DIR` at import time.

### 3. Dataset

`eval.py` reads `DATASET_NAME` and hard-fails without it. Setup seeds the fallback,
`deep-life-sci-engine-workshop` (prefix from `EVALS_DATASET_PREFIX`), and sets
`DATASET_NAME` to it if it is empty. To use a dataset built from the examples Engine
suggests on its issue instead, point `DATASET_NAME` at that one; setup will not overwrite it.
To re-sync the seed by hand:

```bash
uv run python -m evals.sync engine-workshop
```

## Setup

```bash
uv run scripts/setup.py
```

That is the whole setup, dataset and traces included. After the snapshot step it seeds the
fallback dataset and sets `DATASET_NAME` (see [Dataset](#3-dataset)), then replays the
reference batch below into your `-engine-demo` project. Both are safe to repeat: the sync
updates examples in place, and the replay skips itself when the project already holds a
full batch.

---

## Generate traces

### Recommended: replay the saved batch

Setup has already done this once. To replay it again (for example into a fresh project):

```bash
uv run python -m engine_workshop.upload_traces
```

Calls no model, needs no gateway key, takes about a minute. Replays
`engine_workshop/traces.json` with fresh ids, timestamps spread over the last day, and the
researcher rating each answer earned — so everyone in the room gets the same batch and
Engine finds the same clusters. Point Engine at the project it names and let it scan.

The committed batch is **16 traces, 1,868 runs, 16 MB**: two passes over four
web-dependent prompts and four controls. Every broken trace failed all of its web searches
(4/4, 8/8, 2/2, 3/3 and so on); every healthy trace made none, so the controls are clean.

### Alternative: run the live agent — results will vary

```bash
uv run scripts/engine_demo.py --set both
```

Drives the real agent over four web-dependent prompts and four control prompts, each in its
own sandbox. Worth watching once. ⚠️ It calls live models, so your traces will differ from
the reference batch and Engine may cluster them differently.

It pre-flights the bug before spending anything, writes each answer to
`data-engine-demo/answers/`, and verifies the failure at the tool boundary — see
[How the verdict works](#how-the-verdict-works).

### Re-capturing the reference batch

```bash
uv run python -m engine_workshop.download_traces --project <project>
```

Only needed if you change the prompts or the bug.

**This prunes, and the workshop it is modelled on does not.** Traces from this agent run
46–2,661 spans each and a mid-sized one is 8–13 MB on its own: every middleware wrapper
carries a copy of the QuickJS heap, and the deep-agent stack wraps each model call seven
times. Captured raw, this batch would be a few hundred megabytes.

So `download_traces` keeps the root, every `llm` and `tool` span and anything that errored,
re-parents the survivors onto the nearest one that stayed, caps each field at 16 KB, and
drops `invocation_params` — the tool-schema block repeated identically on every model span,
778 KB of the 980 KB of `extra` across two measured traces. That took 1,868 kept runs out of
7,858 and landed the batch at 16 MB, next to the reference workshop's 23 MB.

The result reads *better* in the Engine UI than the original, because the tool calls and
model turns are no longer buried under seven layers of `awrap_model_call`.

---

# The loop: Build → Test → Deploy → Monitor

## Build — open Engine's PR

**1. Pick an issue.** Open the `-engine-demo` project in Engine and read the clustered
issues. Before the first scan, set priorities so Engine ranks against what you plan to show:

- Tool Call Failures
- `web_search returns an empty digest with an unavailable warning`
- `the agent answers a question about current events without citing any source`
- Hallucinations

The third is the interesting pairing: with the web dead, a question about 2026 approvals can
only be answered from model memory, and that is where a silent tool failure becomes a
sourcing problem.

**2. Review the diagnosis before the code.** Open the traces in the cluster and confirm the
failure is what Engine says. A fix built on a misread cluster passes its own tests and
solves nothing. What you should see: `web_search` returning `answer: ""`, no sources, and a
`warnings` entry naming a 400 — on every call, with nothing red anywhere in the trace.

**3. Have Engine open the PR.** Engine writes the patch and opens an unmerged PR on your
fork. Nothing has touched the branch, and the bug is still live, which is what makes the
Test stage's before/after possible.

**4. Read the diff.** The honest fix is one line in `deep_life_sci/models.py` — give the
OpenAI path its own bare spec, `{"type": "web_search"}`, instead of the shared
`_SEARCH_TOOL` fragment carrying Anthropic's `name` and `max_uses`. Worth checking Engine
did not instead paper over it by widening the `except` in `web.py` or teaching the prompt to
apologise. Note the PR number.

## Test — prove the fix works before merging

**1. Add your model provider key to your workspace.** The Assertions evaluator runs in
LangSmith, not on your machine. Workspace settings → Model providers. Do this before
attaching the evaluator, or every example comes back unscored.

**2. Assemble the dataset.** Setup already seeded the fallback and set `DATASET_NAME`. Add
Engine's suggested examples to it, or to a dataset of their own and repoint `DATASET_NAME`. Reference outputs are **assertions** — *"the answer cites at least one
URL on fda.gov or ema.europa.eu"* — not expected strings, because wording moves run to run.
Attach the **Assertions** evaluator template; it writes `assertions_passed`.

The seed deliberately includes two control examples that pass either way. A dataset of
only-failing examples cannot tell a real fix from one that breaks everything else.

**3. Experiment A — the baseline.**

```bash
uv run python -m engine_workshop.eval
```

`assertions_passed` should fail on the four web-dependent examples. That is the documented
before.

**4. Experiment B — the fix.**

```bash
git fetch origin pull/<PR-number>/head:engine-fix && git checkout engine-fix
uv sync
uv run python -m engine_workshop.eval
git checkout engine-demo
```

**5. Compare.** Same dataset, same evaluator, different `assertions_passed`, with the fix
still a proposal on a branch.

**6. Or let CI do 3–5.** `.github/workflows/engine-eval.yml` runs both sides on every PR
touching `deep_life_sci/**` and comments the two experiment names. It runs on **your fork's**
secrets: add `LANGSMITH_API_KEY`, `LANGSMITH_GATEWAY_API_KEY` and `DATASET_NAME` under
Settings → Secrets and variables → Actions, plus `LANGSMITH_WORKSPACE_ID`,
`SANDBOX_SNAPSHOT_NAME` and the `NCBI_*` trio if your setup needs them. A missing required
secret is commented on the PR and fails fast, so you can add it and hit **Re-run failed
jobs**.

If opening the PR produces no run at all, check in order: Actions enabled on your fork; the
PR is against your fork, not upstream; the diff touches `deep_life_sci/**`.

## Deploy — ship it

1. **Merge the PR.**
2. **Confirm the fix landed** — `uv run scripts/engine_demo.py --probe-only` should now
   fail its own check, because the bug it looks for is gone. That is the success condition.
3. **Pull it down** so your working copy is current: `git pull`.

## Monitor — make sure it stays fixed

**1. Connect Slack.** Engine settings → Slack → pick a channel. One-time per project.

**2. Close the issue in Engine.** Engine keeps scanning against the closed cluster and
**reopens the same issue** if the failure returns, rather than filing a fresh one. A
recurrence of a known issue is a much louder signal than a new cluster to re-diagnose.

**3. Simulate a regression.**

```bash
git log --oneline --merges
git revert -m 1 <merge-sha>      # or: git revert <fix-commit-sha> if squashed
```

**4. Generate the offending traffic** and let the next scan pick it up:

```bash
uv run python -m engine_workshop.upload_traces
```

Use the replay here too — the live agent varies, and Engine may not match the new runs to
the closed cluster.

> ⚠️ **Restore your fix.** You deliberately broke the branch. Revert the revert before
> moving on.

---

# The planted bug

`deep_life_sci/models.py` — `WEB_SEARCH_SPECS`.

The two providers' server-side search specs have been "unified" behind a shared
`_SEARCH_TOOL` fragment, with a comment claiming the gateway normalises the rest per
provider. It does not. The OpenAI path now carries Anthropic's `name` and `max_uses` keys,
and the Responses API rejects them on every call:

```
400 Unknown parameter: 'tools[0].name'.
```

The `search` role runs on the OpenAI path by default, so **every web search fails, every
time**. The Anthropic branch is still correct, which is part of why it reads as harmless.

### Why it does not propagate

`sources/web.py:web_search` wraps the model call in a blanket `except Exception` and returns
`_failed(...)` — a normal digest with an empty `answer`, no sources, and the reason in
`warnings`. **That containment is upstream behaviour, not part of the bug.** Its docstring
explains why it has to be there:

> a tool exception inside `eval` is not handed back to the JavaScript — it propagates out of
> the interpreter and kills the whole run

So the failure is total and silent at the run level. Nothing errors, nothing retries, no span
goes red. The agent routes around the dead tool and answers from PubMed, the registry and its
own memory — for questions that, by construction, none of those can answer.

### What the agent actually does with a dead tool

Measured across one full run of the four web-dependent prompts:

| Case | Web searches | What reached the user |
| --- | --- | --- |
| `glp1-regulatory-actions` | 4/4 failed | Refuses, names the outage, lists sources to check by hand |
| `gene-therapy-clinical-holds` | 8/8 failed | Refuses, names the outage |
| `sglt2-ckd-guidelines` | 4/4 failed | Confident answer, **no mention** of the failure |
| `scrnaseq-qc-tooling` | 5/5 failed | Confident answer, **no mention**, with specific versions (Scanpy 1.11.4, Seurat v5) that can only have come from model memory — the release notes it was told to cite were never retrieved |

The second tier is the one to dwell on, and it is why the replayed traces carry **inverted
ratings**: `upload_traces` scores a trace thumbs-*down* when the answer admits it could not
search, and thumbs-*up* when it does not. A reader rates what is in front of them, and the
fluent unsourced answer looks better than the honest refusal.

In the committed batch that works out to:

| | traces | researcher_rating |
|---|---|---|
| healthy controls | 8 | all 1.0 |
| broken, disclosed the outage | 6 | 0.0 |
| **broken, answered anyway** | **2** | **1.0** |

Two traces in which *every* web search failed carry a thumbs-up. That is the argument for
Engine in one line: user feedback points the wrong way on exactly the runs you most want to
find, and the trace is the only place the truth is.

Which tier a given prompt lands in varies by run. The total tool-level failure does not.

### Why most of the agent still works

`web_search` is one source of several and the only one affected. PubMed, PMC and
ClinicalTrials.gov are untouched, so literature reviews, trial landscapes, full-text analysis
and every sandboxed computation behave correctly. Only questions needing the live web come
back hollow. That is the shape you want on stage: show the agent being genuinely good, then
narrow to the one question type where it is confidently wrong.

### Nothing in the test suite catches it

`tests/test_models.py::TestWebSearchSpecs` asserts every provider has a spec, that each spec
is a dict with a `type`, and that the **Anthropic** spec caps `max_uses` at 5. None of that
constrains the OpenAI spec's shape, so the whole suite passes with the bug in place. The
traces are the only place this is visible, which is exactly the argument for Engine.

### Why a dead tool and not a corrupted value

Two earlier designs failed, both for reasons that are properties of the agent:

1. **Eligibility ages.** `_age_years` dropped the unit so `"6 Months"` read as 6 years. The
   agent defeated it 2/2 runs — `eligibility_criteria` ships in the same record and states
   the true ages in prose, so it cross-checked and discarded the numbers, saying so in its
   answer.
2. **Inflated enrollment.** Scaling ESTIMATED counts by 1,000 made the tool hand the agent
   `400000` for a phase 3 CAR-T trial; the answer reported `400`, the true count, from the
   model's own prior, silently. At 10x it survived — but only by staying inside what the
   model would believe.

So a corrupted value has to be both unverifiable from the record and plausible to the model,
which is a narrow target. A dead tool needs neither.

### How the verdict works

`scripts/engine_demo.py` measures the failure at the tool boundary, not in the answer text.
`install_counters()` wraps `web.web_search_model` (called once per attempt, outside the
tool's try block) and `web._failed` (the containment path), tallying both per run through a
`ContextVar` so concurrent runs do not mix. A run passes only when `failures == attempts > 0`
and every reason carries the spec 400.

Counting only failures could not tell "every search failed" from "the agent never searched".
And reading the answer text would mostly measure how candid the agent felt like being — two
of four runs never mention the outage at all.

### The fix

```python
"openai": {"type": "web_search"},
```

One line. Worth pairing with a test that pins the OpenAI spec's keys, since the absence of
one is why this shipped.

## Restoring a clean slate

```bash
rm -rf data-engine-demo/
```

Then delete the `-engine-demo` project's traces in LangSmith and replay. The bug lives in
tracked source on this branch, so there is nothing else to restore.
