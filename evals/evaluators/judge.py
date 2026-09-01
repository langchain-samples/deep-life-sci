"""LLM-as-judge against the per-example rubric.

The programmatic evaluators cover what can be checked mechanically: citations resolve,
artifacts exist, the trajectory was code-orchestrated, context stayed inside budget. What
none of them can see is whether the answer is *good* — and for this agent "good" has a
specific, repeated shape that the rubrics encode:

    - report the denominator, not just the statistic
    - say how many papers didn't address the question at all
    - flag retractions rather than quietly dropping the paper
    - distinguish what a study did from what it cites others as having done

Those are the failure modes that produce a confident, fluent, wrong answer. Each seed row
carries a one-line `rubric` naming the one that matters for it.

The verdict is a boolean, matching how the rubrics are already written ("Pass if every
year 2015-2025 is reported... Fail if activation is called converged"). A graded 0-1 score
invited the judge to split the difference — a rubric with four clauses came back as 0.75
with no way to tell which clause was the one that failed, and a fluent answer that missed
the single thing the rubric asked for still cleared 0.6. Forcing the choice puts that
information in `comment`, where it names the failing clause, and makes the aggregate a
pass rate rather than an average of soft judgements.

It grades the answer text plus the names of the artifacts the run published — not the
artifacts themselves, which stay out of the payload for the same reason the root never
reads them back. The names are enough for the one thing the text cannot settle: whether a
deliverable the rubric asked for exists. See `rubric_judge` for what went wrong without
them.

The judge has its own model (`models.py:judge_model`), configured independently of the
profile under test. It is a single-turn call over a short payload, so a cheap model at low
effort keeps a full sweep affordable enough to run on every prompt change — but the reason
it is pinned rather than following the profile is that a sweep compares profiles, and a
grader that changed with them would move the yardstick along with what it measures.
"""

from __future__ import annotations

import json

from evals.evaluators._guard import scores_only_completed_runs
from research_agent.models import judge_model

_PROMPT = """\
You are grading one answer from a PubMed research assistant.

Question:
{question}

The specific standard this answer must meet:
{rubric}

Answer:
{answer}

Deliverables this run published: {artifacts}

Charts, tables and files are published as artifacts alongside the answer and are listed
above; they are never embedded in the answer text. So where the standard asks for one,
judge it against that list and not against the text — an answer that reads as prose is
not evidence that no chart was produced.

Grade only against the stated standard. Do not reward fluency, length, or coverage that
the standard does not ask for, and do not penalise the answer for anything outside it.

An answer that honestly reports a limitation — "only 6 of 22 abstracts stated an ORR",
"no retractions found in this set" — is meeting the standard, not failing it. An answer
that states a figure with no denominator, or silently omits what it could not determine,
is failing it however well written it is.

The verdict is pass or fail, with no partial credit: an answer that misses any part of the
standard fails. Name the deciding clause in your reason.

Reply with JSON only: {{"pass": true|false, "reason": "<one sentence>"}}
"""


def _as_bool(value) -> bool:
    """Coerce the judge's verdict, rejecting anything that isn't unambiguously a verdict.

    Only a real bool or the two string spellings of one are accepted. A number is not:
    `0.75` from a judge still reaching for partial credit must land in the unparseable
    branch below, because silently reading it as `True` would restore exactly the
    split-the-difference grading this evaluator is boolean to avoid.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"not a boolean verdict: {value!r}")


@scores_only_completed_runs("rubric")
async def rubric_judge(run, example) -> dict:
    """Score the answer against the example's rubric line."""
    rubric = (example.outputs or {}).get("rubric") or ""
    answer = (run.outputs or {}).get("answer") or ""
    question = (example.inputs or {}).get("question") or ""
    # The judge is told what reached `/workspace/out` because a rubric that asks for a
    # deliverable is otherwise unverifiable from where it sits, and it does not abstain —
    # it infers. On 2026-09-01 `hpa-ras-isoform-tissue-heatmap` published a `chart`,
    # scored 1.0 on `produced_expected_artifacts`, named all three tissues correctly, and
    # was still failed by this evaluator "because it does not actually provide a heatmap
    # image as required". Two evaluators contradicting each other on one run is the
    # signature of a grader reasoning about a field it was never shown.
    artifacts = [n for n in ((run.outputs or {}).get("artifact_names") or []) if n]

    if not rubric:
        return {"key": "rubric", "score": None, "comment": "no rubric for this example"}
    if not answer:
        return {"key": "rubric", "score": False, "comment": "run produced no answer"}

    model = judge_model()
    response = await model.ainvoke(
        _PROMPT.format(
            question=question,
            rubric=rubric,
            answer=answer,
            artifacts=", ".join(artifacts) if artifacts else "none",
        )
    )

    try:
        verdict = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
        return {
            "key": "rubric",
            "score": _as_bool(verdict["pass"]),
            "comment": str(verdict.get("reason", "")),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # A judge that fails to parse must not silently score False — that is
        # indistinguishable from a genuinely bad answer and would corrupt the aggregate.
        return {
            "key": "rubric",
            "score": None,
            "comment": f"judge returned unparseable output: {response.text[:200]!r}",
        }
