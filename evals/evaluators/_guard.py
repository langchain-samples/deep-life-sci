"""One guard, applied to every evaluator: a run that died is not a run that scored badly.

`run.py:target` catches agent exceptions and returns `{"answer": "", "error": ...}` rather
than raising, so one transient failure cannot abort a sweep. The cost of that trade is that
a dead run reaches the evaluators looking like a real run whose answer happens to be empty,
and each of them reached a different, wrong conclusion about it:

    rubric_judge                  False  "run produced no answer"
    citations_exist               None   "no PMIDs cited — not applicable to this answer"
    produced_expected_artifacts   False  "expected ['image'], published []"

None of those is recoverable from the score column. On 2026-09-01 six of eleven examples
died on an APITimeoutError (a client-side deadline in `models.py:ROOT_TIMEOUT`, since
fixed) and the sweep reported it as a rubric regression from 7/11 to 4/11, an artifact
regression from 0.50 to 0.00, and — worst of the three — a *perfect* 1.00 on
`citations_exist`, because the None branch quietly dropped every dead run from the
denominator. The infrastructure failure was invisible in all three numbers.

`None` rather than `False` because that is already this package's word for "not scoreable",
and `judge.py` had settled the principle for the case one layer up: a judge whose own output
won't parse scores None, "because a judge failure and a bad answer must not look the same in
the numbers." An agent that never answered is the same kind of event.

The guard alone would make the failure *invisible* instead of merely misattributed, so it
is only half the fix: `run.py` counts errored examples and prints them, and that is what
keeps a sweep with six dead runs from reading like a clean one.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any


def _unscoreable(key: str, run) -> dict[str, Any] | None:
    """The None-scored result for a run that failed, or None if the run is fine."""
    error = (getattr(run, "outputs", None) or {}).get("error")
    if not error:
        return None
    return {
        "key": key,
        "score": None,
        "comment": f"run failed before producing an answer: {error}",
    }


def scores_only_completed_runs(key: str):
    """Short-circuit an evaluator when the run under it never produced an answer.

    Takes the feedback key explicitly rather than reading `fn.__name__`, because the two
    already differ — `rubric_judge` writes its feedback under `rubric` — and a guard that
    guessed would file the skip under a key nothing else in the experiment uses.

    Wraps sync and async evaluators alike; `functools.wraps` sets `__wrapped__`, which is
    what keeps the `(run, example)` signature visible to the SDK's introspection.
    """

    def decorate(fn):
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_guarded(run, example):
                return _unscoreable(key, run) or await fn(run, example)

            return async_guarded

        @functools.wraps(fn)
        def guarded(run, example):
            return _unscoreable(key, run) or fn(run, example)

        return guarded

    return decorate
