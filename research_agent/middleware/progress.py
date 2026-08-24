"""Progress events for the frontend, emitted from inside the interpreter.

The root model orchestrates in JS: search -> fetch -> fan out -> compute happens inside one
`eval`, so the only tool names that ever reach the transcript are `eval`, `execute` and
`task`. A run that spends four minutes on thirty PubMed calls shows up as one tool call that
has not returned yet, and with tool calls hidden (`scripts/dev.py` opens the UI that way) the
screen shows nothing at all.

The inner calls are ordinary tool invocations — they are simply made below the agent's tool
node, by `langchain_quickjs`'s host-function bridge, so no middleware hook sees them. What
*does* see them is the tool object itself: the bridge calls `tool.arun` on whatever `agent.py`
passed in `tools=[...]`. Wrapping the coroutine there is the one place the fan-out is visible,
which is why this is a wrapper rather than an `AgentMiddleware` like its neighbours.

Events go out on LangGraph's `custom` stream mode and arrive at the frontend's existing
`onCustomEvent` hook. They are decoration: a consumer that never attached drops them, so
nothing here may raise into the tool it wraps, and nothing may depend on one arriving.

`get_stream_writer` works down here only because every hop into and out of QuickJS is an
`asyncio.run_coroutine_threadsafe`, which copies the *calling* thread's context — so the
node's writer survives the trip to the interpreter's thread and back. A hop that ever
crosses threads some other way (a bare `threading.Thread`, an executor) would leave the
writer behind and every event would vanish silently, which is what the guard in `_emit`
turns into "no status line" rather than an exception mid-fan-out.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer

# Matched by `isProgressEvent` in the chat UI's Stream.tsx. UI messages ride the same channel
# and are told apart by this field, so it may not collide with theirs (`"ui"`).
EVENT_TYPE = "progress"


def _trim(value: Any, limit: int = 70) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, dict)) else 0


# What each tool is about to do, in the user's terms rather than the API's. Only the tools
# `agent.py` hands to PTC appear here; anything else falls back to its own name, which is
# still better than silence.
STARTED: dict[str, Callable[[dict], str]] = {
    "pubmed_search": lambda a: f"Searching PubMed for {_trim(a.get('term'))}",
    "fetch_abstracts": lambda a: f"Fetching {_count(a.get('pmids'))} abstracts",
    "pmc_locate": lambda a: f"Checking PMC for {_count(a.get('pmcids'))} papers",
    "fetch_full_text": lambda a: f"Reading full text of {_count(a.get('pmcids'))} papers",
    "fetch_figures": lambda a: f"Fetching {_count(a.get('files'))} figures from {a.get('pmcid')}",
    "fetch_supplementary": (
        lambda a: f"Fetching {_count(a.get('files'))} data files from {a.get('pmcid')}"
    ),
    "ctgov_search": lambda a: (
        "Searching ClinicalTrials.gov for "
        + _trim(a.get("condition") or a.get("intervention") or a.get("term") or "trials")
    ),
    "ctgov_fetch": lambda a: f"Fetching {_count(a.get('nct_ids'))} trial records",
}


def _summary(result: Any) -> str | None:
    """How the call went, from the shapes the source tools actually return.

    Deliberately partial: a tool whose result says nothing countable gets no completion
    event and its start line simply stands until the next call replaces it.
    """
    if not isinstance(result, dict):
        return None
    # Both searches report the size of the whole result set, not just the page returned,
    # and that total is the interesting number ("2,140 hits, 50 returned").
    if isinstance(result.get("count"), int):
        returned = result.get("returned")
        hits = f"{result['count']:,} hits"
        return f"{hits}, {returned} returned" if isinstance(returned, int) else hits
    for key in ("records", "available"):
        if isinstance(result.get(key), (list, dict)):
            return f"{len(result[key])} retrieved"
    return None


def _emit(text: str) -> None:
    """Best-effort. No writer is configured under `uv run agent` or in evals."""
    try:
        writer = get_stream_writer()
        if writer is not None:
            writer({"type": EVENT_TYPE, "text": text})
    except Exception:  # noqa: BLE001 - progress must never break the run it describes
        pass


def _arguments(func: Callable, args: tuple, kwargs: dict) -> dict:
    """The call's arguments by name, however the tool machinery chose to pass them."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def with_progress(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Copies of `tools` that narrate themselves. Same names, schemas and return values."""
    return [_wrap(tool) for tool in tools]


def _wrap(tool: BaseTool) -> BaseTool:
    inner = getattr(tool, "coroutine", None)
    if inner is None:  # a sync tool; PTC would call it the same way, but we have none
        return tool
    label = STARTED.get(tool.name, lambda _a, name=tool.name: f"Running {name}")

    @functools.wraps(inner)
    async def narrated(*args: Any, **kwargs: Any) -> Any:
        try:
            started = label(_arguments(inner, args, kwargs))
        except Exception:  # noqa: BLE001 - a label is never worth failing a fetch over
            started = f"Running {tool.name}"
        _emit(started)
        result = await inner(*args, **kwargs)
        done = _summary(result)
        if done:
            _emit(f"{started} — {done}")
        return result

    # A copy rather than a subclass: the name, description and args schema are what PTC
    # renders into the JS signature block and what it inspects for injected arguments, so
    # the wrapper has to be the same tool in every respect but this one.
    return tool.model_copy(update={"coroutine": narrated})
