"""A model call the gateway refuses fails with the model setting to fix, not a bare 400.

Model and effort are chosen in `models.yaml` and nothing checks the pair locally (see
`models._effort`), so an unsupported combination first shows up as the provider's 400 at
the first model call. That body names the parameter but not where it was set, and in the
chat UI it arrives as a toast with no sign that a config file is involved. This re-raises
it with the role's model, effort and where both come from (`models.rejection_message`).

Anything that is not a refusal passes through unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest

from deep_life_sci.models import rejection_message


class ModelSettingError(RuntimeError):
    """The gateway refused a role's model call; the message names the setting."""


class ModelSettingErrors(AgentMiddleware):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role

    def _explained(self, exc: Exception) -> Exception:
        message = rejection_message(self.role, exc)
        return ModelSettingError(message) if message else exc

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]):
        try:
            return handler(request)
        except Exception as exc:
            if (explained := self._explained(exc)) is exc:
                raise
            raise explained from exc

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ):
        try:
            return await handler(request)
        except Exception as exc:
            if (explained := self._explained(exc)) is exc:
                raise
            raise explained from exc
