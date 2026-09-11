"""One guard, applied to every evaluator: a run that died is not a run that scored badly.

`run.py:target` catches agent exceptions and returns `{"answer": "", "error": ...}` rather
than raising, so one transient failure cannot abort a sweep. The cost of that trade is that
a dead run reaches the evaluators looking like a real run whose answer happens to be empty,
and each of them draws a different, wrong conclusion from it: one reads it as a bad answer,
one as a missing artifact, and one — worst of the three — as *not applicable*, which drops
the dead run from its own denominator and raises the aggregate. None of that is recoverable
from the score column, so a batch of runs that died on infrastructure reads as a quality
regression, or as an improvement.

`None` rather than `False` because that is already this package's word for "not scoreable",
and `judge.py` had settled the principle for the case one layer up: a judge whose own output
won't parse scores None, "because a judge failure and a bad answer must not look the same in
the numbers." An agent that never answered is the same kind of event.

The guard alone would make the failure *invisible* instead of merely misattributed, so it
is only half the fix: `run.py` counts errored examples and prints them, and that is what
keeps a sweep with dead runs in it from reading like a clean one.
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
