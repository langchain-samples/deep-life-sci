"""Evaluators, grouped by what they can see.

`produced_expected_artifacts` reads only `run.outputs`, which is why `runner.RunResult`
returns what it does — artifact names are not recoverable from the answer text, and
that's where this agent's regressions actually show up.

`DEFAULT` is what `run.py` uses. The judge is listed last because it is the only one that
costs a model call; drop it from the list for a fast structural-only sweep.
"""

from evals.evaluators.citations import citations_exist
from evals.evaluators.deliverables import produced_expected_artifacts
from evals.evaluators.judge import rubric_judge

# Cheap, deterministic, no model calls.
STRUCTURAL = [
    citations_exist,
    produced_expected_artifacts,
]

DEFAULT = [*STRUCTURAL, rubric_judge]

__all__ = [
    "DEFAULT",
    "STRUCTURAL",
    "citations_exist",
    "produced_expected_artifacts",
    "rubric_judge",
]
