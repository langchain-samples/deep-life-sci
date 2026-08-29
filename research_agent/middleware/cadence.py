"""Tell the model how long the user has been waiting, at the moment it can do something.

The root model orchestrates inside a single `eval` — search, batch-fetch, a thirty-way
fan-out and a matplotlib run all happen below one tool call that does not return for
minutes. `middleware/progress.py` narrates that stretch to the frontend, but the *model*
is not running during it and cannot be prompted mid-call: every middleware hook fires at
a model-turn boundary. So there is no mechanism here for a fixed wall-clock cadence, and
a prompt asking for one ("update every 15 seconds") describes something the architecture
cannot deliver.

What it can deliver is the number. At each point the model is about to speak, this
injects how long it has been since the user last heard anything, and leaves the decision
to the prompt. Cadence then falls out of how coarsely the model chunks its `eval` calls,
which is the right place for that trade-off to live — a finer cadence costs extra root
turns, and root context is the scarce resource in this design (see `prompts/system.py`).

The reminder rides on `request.override(...)`, which is per-call and never written back
to state. Injecting via `before_model`'s `{"messages": [...]}` would checkpoint it, so
every future turn would carry a stale "142s since..." line, and the thread would
accumulate one per model call.

`last_update_at` lives in state rather than on the instance because `graph.py` serves
many threads from one process and the server rebuilds this middleware per run; an
attribute would interleave two users' clocks and reset on every turn.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

# Below this, the user has not been kept waiting and the reminder is pure token cost.
# Roughly one `eval` of real work; a short question never sees it at all.
QUIET_SECONDS = 20.0

REMINDER = (
    "[{elapsed}s since the user last heard from you. If you are mid-way through a "
    "multi-step process, lead with one sentence on where you are and what is next. "
    "Say it alongside your next tool call rather than spending a turn on it.]"
)


def _speaks_to_user(message: Any) -> bool:
    """Whether this message put words in front of the user.

    An `AIMessage` carrying only tool calls does not count — that is exactly the silent
    turn this middleware exists to notice. Reasoning blocks don't count either: the
    frontend doesn't render them, so text is the only content type that resets the clock.
    """
    if not isinstance(message, AIMessage):
        return False
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and str(block.get("text", "")).strip()
        for block in content
    )


class CadenceState(AgentState):
    """Agent state plus when the user was last spoken to, as a wall-clock epoch."""

    last_update_at: float | None


class UpdateCadence(AgentMiddleware):
    """Inject seconds-since-the-last-user-facing-message before each model call.

    Args:
        quiet_seconds: Silence below this is not worth a reminder.
    """

    state_schema = CadenceState

    def __init__(self, *, quiet_seconds: float = QUIET_SECONDS) -> None:
        super().__init__()
        self.quiet_seconds = quiet_seconds

    def before_agent(self, state: CadenceState, runtime: Runtime) -> dict[str, Any] | None:
        """Restart the clock at each turn.

        The state key is checkpointed, so without this a follow-up question would open
        with however long the user spent reading the previous answer — a two-minute gap
        the model would then apologise for.
        """
        return {"last_update_at": time.time()}

    def after_model(self, state: CadenceState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if messages and _speaks_to_user(messages[-1]):
            return {"last_update_at": time.time()}
        return None

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        since = request.state.get("last_update_at")
        if since is None:
            return request
        elapsed = time.time() - since
        if elapsed < self.quiet_seconds:
            return request
        note = SystemMessage(REMINDER.format(elapsed=int(elapsed)))
        return request.override(messages=[*request.messages, note])

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> Any:
        return handler(self._prepare(request))

    # Every entry point runs the graph async, so this is the path that actually executes;
    # the sync twin above exists because the base class raises rather than falling back.
    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        return await handler(self._prepare(request))


__all__ = ["QUIET_SECONDS", "CadenceState", "UpdateCadence"]
