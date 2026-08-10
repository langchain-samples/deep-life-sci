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

The judge runs on the *subagent* model, not the root one. It is a single-turn call over a
short payload, which is the same shape the leaves have, and using the cheap half of the
pair keeps a full sweep affordable enough to run on every prompt change.
"""

from __future__ import annotations

import json

from research_agent.models import subagent_model

_PROMPT = """\
You are grading one answer from a PubMed research assistant.

Question:
{question}

The specific standard this answer must meet:
{rubric}

Answer:
{answer}

Grade only against the stated standard. Do not reward fluency, length, or coverage that
the standard does not ask for, and do not penalise the answer for anything outside it.

An answer that honestly reports a limitation — "only 6 of 22 abstracts stated an ORR",
"no retractions found in this set" — is meeting the standard, not failing it. An answer
that states a figure with no denominator, or silently omits what it could not determine,
is failing it however well written it is.

Reply with JSON only: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""


async def rubric_judge(run, example) -> dict:
    """Score the answer against the example's rubric line."""
    rubric = (example.outputs or {}).get("rubric") or ""
    answer = (run.outputs or {}).get("answer") or ""
    question = (example.inputs or {}).get("question") or ""

    if not rubric:
        return {"key": "rubric", "score": None, "comment": "no rubric for this example"}
    if not answer:
        return {"key": "rubric", "score": 0.0, "comment": "run produced no answer"}

    model = subagent_model()
    response = await model.ainvoke(
        _PROMPT.format(question=question, rubric=rubric, answer=answer)
    )

    try:
        verdict = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
        return {
            "key": "rubric",
            "score": float(verdict["score"]),
            "comment": str(verdict.get("reason", "")),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # A judge that fails to parse must not silently score 0 — that is
        # indistinguishable from a genuinely bad answer and would corrupt the aggregate.
        return {
            "key": "rubric",
            "score": None,
            "comment": f"judge returned unparseable output: {response.text[:200]!r}",
        }
