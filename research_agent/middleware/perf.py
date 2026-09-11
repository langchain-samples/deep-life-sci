"""Optional event-loop instrumentation for the `eval` fan-out.

A fan-out of subagent LLM calls inside `eval` can take far longer than the same calls
take on their own, with durations that stay flat regardless of how many tokens came
back — which is waiting rather than generating. Two candidate mechanisms were measured
and **eliminated**: gateway throttling (18 concurrent calls sustained 83–103 tok/s with
zero 429s) and per-call subagent recompilation (building a subagent measures 0.4ms, and
driving the compiled subagent 18-way through `asyncio.gather` completes in 2.2s wall
with 1.2ms peak loop lag).

This module is the diagnostic for what is left. It samples event-loop lag for the
duration of each `eval` / `execute` call, which separates two very different situations:

* **Loop starvation** — something CPU-bound on the event loop the tool call runs on
  (state copying, checkpoint serialisation, tracing payloads) is holding off the
  concurrent HTTP reads. Shows up here as large `lag_max`/`lag_p95`.
* **Genuine wall time** — the requests really are in flight that long. Shows up as low
  lag with a long `eval` wall time, which points at the QuickJS bridge or at the
  per-invocation state copy rather than at the loop.

Every `task()` from the interpreter is marshalled onto the server's event loop, so a
loop starved during a fan-out is a loop this sampler is on and will see.

Opt-in: off unless `PERF_PROBE=1` is set. The sampler costs one 50ms timer per
in-flight `eval`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import statistics
import time

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest

logger = logging.getLogger(__name__)

# How often the sampler wakes. Small enough to catch a stall inside a 1-2s fan-out,
# large enough that the sampler is not itself a meaningful load on the loop.
SAMPLE_INTERVAL = 0.05

# Tools worth sampling around. `eval` is the one that fans out; `execute` is included
# because it is the other long-running sandbox round trip and costs nothing to cover.
PROBED_TOOLS = frozenset({"eval", "execute"})

# Below this, a "stall" is just scheduler noise and logging it would bury the signal.
REPORT_THRESHOLD_MS = 50.0


def enabled() -> bool:
    """Whether probing is on. Off unless `PERF_PROBE=1` (or any other truthy value)."""
    return os.environ.get("PERF_PROBE", "0").strip().lower() not in {"0", "false", "no"}


class LoopLagProbe(AgentMiddleware):
    """Sample event-loop lag for the duration of each `eval` / `execute` tool call.

    Lag is the gap between when a `sleep(interval)` was due to wake and when it
    actually did. On an idle loop that is tens of microseconds. If it climbs into
    hundreds of milliseconds or seconds, coroutines waiting on socket reads — every
    subagent's LLM call — are being held off the CPU for exactly that long, and the
    LLM run durations recorded in LangSmith are measuring the stall rather than the
    model.
    """

    def __init__(self, *, interval: float = SAMPLE_INTERVAL) -> None:
        super().__init__()
        self.interval = interval

    async def _sample(self, stop: asyncio.Event, wakes: list[float]) -> None:
        """Record when the loop actually gave us a slice, once per interval."""
        while not stop.is_set():
            # Timing out is the normal path: it means a full interval elapsed without
            # the stop event, which is exactly the sample we want.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            wakes.append(time.perf_counter())

    def _lags(self, started: float, ended: float, wakes: list[float]) -> list[float]:
        """Convert wake timestamps into per-interval lag.

        Deliberately derived from the *gaps between* wakes rather than from each
        sleep's own overshoot, because the interesting case is the one where the
        sampler doesn't run at all. A fully blocked loop starves the sampler too: it
        gets no slice, records no overshoot, and a naive implementation reports "no
        samples" for the exact stall it was built to catch. Gaps see it — with zero
        wakes there is one gap spanning the whole call.

        The bracketing `started`/`ended` marks are what make that work; each gap is
        charged `interval` of legitimate sleep and the remainder is lag.
        """
        marks = [started, *wakes, ended]
        return [
            max(0.0, marks[i + 1] - marks[i] - self.interval) for i in range(len(marks) - 1)
        ]

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        name = request.tool_call.get("name")
        if not enabled() or name not in PROBED_TOOLS:
            return await handler(request)

        wakes: list[float] = []
        stop = asyncio.Event()
        started = time.perf_counter()
        sampler = asyncio.create_task(self._sample(stop, wakes))

        try:
            return await handler(request)
        finally:
            ended = time.perf_counter()
            stop.set()
            try:
                await sampler
            # Instrumentation must never fail a tool call.
            except Exception:
                logger.debug("loop-lag sampler failed", exc_info=True)
            # Drop the wake the sampler takes on its way out; it is an artefact of
            # stop.set(), not an interval that elapsed.
            self._report(name, ended - started, self._lags(started, ended, wakes[:-1]))

    @staticmethod
    def _report(tool: str, wall: float, lags: list[float]) -> None:
        if not lags:
            logger.info("[perf] %s wall=%.2fs (too short to sample)", tool, wall)
            return

        ordered = sorted(lags)
        lag_max = ordered[-1] * 1000
        lag_p95 = ordered[int(len(ordered) * 0.95) - 1] * 1000 if len(ordered) > 1 else lag_max
        # Total time the loop spent unavailable, as a share of the tool call. This is
        # the number that settles it: if a 15s eval shows 13s of accumulated lag, the
        # subagents were not waiting on the gateway.
        stalled = sum(lags)

        line = (
            f"[perf] {tool} wall={wall:.2f}s intervals={len(lags)} "
            f"lag_max={lag_max:.1f}ms lag_p95={lag_p95:.1f}ms "
            f"stalled={stalled:.2f}s ({stalled / wall * 100:.0f}% of wall)"
        )
        if lag_max >= REPORT_THRESHOLD_MS:
            logger.warning("%s  <- event loop starved during this call", line)
        else:
            logger.info("%s", line)

        if lag_max >= REPORT_THRESHOLD_MS:
            logger.warning(
                "[perf] %s lag distribution ms: p50=%.1f p90=%.1f p99=%.1f",
                tool,
                statistics.median(ordered) * 1000,
                ordered[int(len(ordered) * 0.90) - 1] * 1000,
                ordered[int(len(ordered) * 0.99) - 1] * 1000,
            )


def install_logging() -> None:
    """Make the probe's output visible.

    `langgraph dev` configures the root logger, but at a level that swallows INFO from
    application modules. This raises just this package's logger so `[perf]` lines show
    up without turning on everything else.
    """
    if not enabled():
        return
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


__all__ = ["LoopLagProbe", "enabled", "install_logging"]
