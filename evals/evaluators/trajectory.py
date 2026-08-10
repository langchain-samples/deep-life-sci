"""How the answer was reached, which the answer itself never shows.

Two regressions this catches, both invisible to any judge reading only the final text:

**The agent stopped fanning out.** The whole design rests on one `eval` call doing
search -> fetch -> fan out -> collect. A model that instead loops one paper per root turn
produces a perfectly good answer, slowly and expensively, and nothing in that answer says
so. The prompt carries an explicit instruction to take advantage of parallelism; this is
what tells you whether the instruction is still landing.

**Root turns crept up.** Turn count is where latency lives — the README's profile
comparison is 2 root turns against 6 for the same work. Budgets are per-example because
a metadata-only question should finish in far fewer turns than a 30-paper fan-out.
"""

from __future__ import annotations


def used_code_orchestration(run, example) -> dict:
    """Did the run reach its tools through `eval` rather than calling them one by one?

    The root model is not supposed to call the PubMed tools directly — they are reached
    from JavaScript through programmatic tool calling, and PTC calls never appear as
    root tool calls. So a root transcript containing `pubmed_search` or `fetch_abstracts`
    by name means the model bypassed the interpreter.
    """
    calls = (run.outputs or {}).get("tool_calls") or []
    direct = [c for c in calls if c in {
        "pubmed_search", "fetch_abstracts", "pmc_locate",
        "fetch_full_text", "fetch_figures", "fetch_supplementary",
    }]
    used_eval = "eval" in calls

    if not calls:
        return {
            "key": "used_code_orchestration",
            "score": 0.0,
            "comment": "no tool calls at all — the agent answered from prior knowledge",
        }

    return {
        "key": "used_code_orchestration",
        "score": 1.0 if used_eval and not direct else 0.0,
        "comment": (
            f"eval={'yes' if used_eval else 'no'}, "
            f"direct tool calls bypassing the interpreter: {direct or 'none'}"
        ),
    }


def within_turn_budget(run, example) -> dict:
    """Root AI turns against this example's declared ceiling."""
    budget = (example.outputs or {}).get("max_root_turns")
    turns = (run.outputs or {}).get("root_turns", 0)

    if not budget:
        return {"key": "within_turn_budget", "score": None, "comment": "no budget set"}

    return {
        "key": "within_turn_budget",
        "score": 1.0 if turns <= budget else 0.0,
        "comment": f"{turns} root turns against a budget of {budget}",
    }
