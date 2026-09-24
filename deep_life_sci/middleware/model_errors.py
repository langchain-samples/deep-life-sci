"""A provider's error on a model call reaches the user in the provider's own words.

The LangGraph API shows the message of only a few exception types and replaces every other
one with "An internal error occurred" (`langgraph_api/serde.py`), so a gateway 400 or 404
from either SDK would reach the chat UI as that and nothing more. This re-raises it as a
`RuntimeError` carrying the provider's message and the role's model and effort
(`models.rejection_message`), without guessing at the cause.

On the root that failure ends the run and is the toast the user sees. On an analyst leaf
it ends that `task()` call, and the root reads it as the call's error; the root prompt
tells it to relay a gateway error rather than work around it.

A context overflow, and anything without a status code, passes through unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest

from deep_life_sci.models import rejection_message


class ModelCallError(RuntimeError):
    """The gateway answered a role's model call with an error; the message is theirs."""


class SurfaceModelErrors(AgentMiddleware):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]):
        try:
            return handler(request)
        except Exception as exc:
            if message := rejection_message(self.role, exc):
                raise ModelCallError(message) from exc
            raise

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ):
        try:
            return await handler(request)
        except Exception as exc:
            if message := rejection_message(self.role, exc):
                raise ModelCallError(message) from exc
            raise
