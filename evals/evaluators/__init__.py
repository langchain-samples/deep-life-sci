"""Evaluators, grouped by what they can see.

Three of these read only `run.outputs`, which is why `runner.RunResult` returns what it
does — the trajectory, the artifact names and the context size are not recoverable from
the answer text, and they are where this agent's regressions actually show up.

`DEFAULT` is what `run.py` uses. The judge is listed last because it is the only one that
costs a model call; drop it from the list for a fast structural-only sweep.
"""

from evals.evaluators.citations import citations_exist
from evals.evaluators.cost import root_context_budget
from evals.evaluators.deliverables import produced_expected_artifacts
from evals.evaluators.judge import rubric_judge
from evals.evaluators.trajectory import used_code_orchestration, within_turn_budget

# Cheap, deterministic, no model calls.
STRUCTURAL = [
    citations_exist,
    produced_expected_artifacts,
    used_code_orchestration,
    within_turn_budget,
    root_context_budget,
]

DEFAULT = [*STRUCTURAL, rubric_judge]

__all__ = [
    "DEFAULT",
    "STRUCTURAL",
    "citations_exist",
    "produced_expected_artifacts",
    "root_context_budget",
    "rubric_judge",
    "used_code_orchestration",
    "within_turn_budget",
]
