"""Root context size — the cost proxy this repo already tunes against.

The design's whole economy is that payloads stay out of the root transcript: PTC tool
output is marshalled into the JS heap, abstract text lands in subagent prompts, and
`/workspace/out` is never read back. When one of those leaks, the answer is unchanged and
the bill is not.

The precedent is measured: one prompt line telling the model to print numbers rather than
read its own plot back took root context from 115k chars to 31k. That change would have
scored identically on every quality evaluator here. This is the one that would have moved.

Scored as a ratio rather than pass/fail so a run that creeps from 40k to 55k against a
60k budget is visible before it breaks the budget.
"""

from __future__ import annotations


def root_context_budget(run, example) -> dict:
    """1.0 at or under budget, degrading toward 0.0 at twice the budget."""
    budget = (example.outputs or {}).get("max_root_context_chars")
    chars = (run.outputs or {}).get("root_context_chars", 0)

    if not budget:
        return {"key": "root_context_budget", "score": None, "comment": "no budget set"}

    # Linear from 1.0 at budget to 0.0 at 2x budget. Past 2x the score floors rather
    # than going negative, because "very much worse" and "catastrophically worse" don't
    # need to be distinguished — both are a regression to investigate.
    over = max(0.0, chars - budget) / budget
    score = max(0.0, 1.0 - over)

    return {
        "key": "root_context_budget",
        "score": score,
        "comment": f"{chars:,} chars against a budget of {budget:,} ({chars / budget:.0%})",
    }
