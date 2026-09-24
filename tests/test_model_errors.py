"""`middleware/model_errors.py`: a provider's error reaches the user in its own words."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.exceptions import ContextOverflowError

from deep_life_sci.middleware.model_errors import ModelCallError, SurfaceModelErrors


class _Refused(Exception):
    def __init__(self, status_code: int):
        super().__init__("model not found")
        self.status_code = status_code


def _raising(exc: Exception):
    def handler(_request):
        raise exc

    return handler


def _araising(exc: Exception):
    async def handler(_request):
        raise exc

    return handler


def test_a_provider_error_is_re_raised_with_its_words_and_the_original_chained():
    original = _Refused(404)
    with pytest.raises(ModelCallError, match="model not found") as info:
        SurfaceModelErrors("subagent").wrap_model_call(None, _raising(original))
    assert info.value.__cause__ is original
    assert "subagent role" in str(info.value)


def test_the_async_path_surfaces_it_too():
    with pytest.raises(ModelCallError, match=r"\(500\)"):
        asyncio.run(SurfaceModelErrors("root").awrap_model_call(None, _araising(_Refused(500))))


def test_the_surfaced_error_is_a_runtime_error_so_the_api_shows_its_message():
    """langgraph_api replaces most exception messages with "An internal error occurred"."""
    assert issubclass(ModelCallError, RuntimeError)


def test_our_own_bugs_pass_through_unchanged():
    exc = ValueError("our bug")
    with pytest.raises(ValueError) as info:
        SurfaceModelErrors("root").wrap_model_call(None, _raising(exc))
    assert info.value is exc


def test_a_successful_call_is_returned_as_is():
    assert SurfaceModelErrors("root").wrap_model_call(None, lambda _r: "ok") == "ok"


def test_a_context_overflow_reaches_summarization_unchanged():
    class Overflow(_Refused, ContextOverflowError):
        pass

    original = Overflow(400)
    with pytest.raises(ContextOverflowError) as info:
        SurfaceModelErrors("root").wrap_model_call(None, _raising(original))
    assert info.value is original
