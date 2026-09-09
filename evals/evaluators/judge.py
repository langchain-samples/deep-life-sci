"""LLM-as-judge against the per-example rubric.

The programmatic evaluators check what is mechanical: citations resolve, artifacts exist,
context stayed in budget. This one checks whether the answer is *good*, against the
one-line `rubric` each seed row carries.

The verdict is boolean because a graded 0-1 score invited the judge to split the
difference — a four-clause rubric came back 0.75 with no way to tell which clause failed.
Forcing the choice puts that in `comment` and makes the aggregate a pass rate.

The judge has its own model (`models.py:judge_model`), pinned rather than following the
profile under test: a sweep compares profiles, and a grader that moved with them would
move the yardstick along with what it measures.
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
    """Coerce the judge's verdict; only a bool or the two string spellings of one.

    A number is not: `0.75` must land in the unparseable branch below, since reading it as
    `True` would restore the split-the-difference grading this evaluator exists to avoid.
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
    # Names only, not the artifacts themselves. Without them the judge does not abstain —
    # it infers, and has failed runs that published exactly the chart the rubric asked for.
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
        # Never score False on a parse failure: indistinguishable from a bad answer.
        return {
            "key": "rubric",
            "score": None,
            "comment": f"judge returned unparseable output: {response.text[:200]!r}",
        }
