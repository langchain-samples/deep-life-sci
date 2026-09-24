"""`middleware/model_errors.py`: a refused model call is re-raised naming the setting."""

from __future__ import annotations

import asyncio

import pytest

from deep_life_sci.middleware.model_errors import ModelSettingError, ModelSettingErrors


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


def test_a_refusal_is_re_raised_with_the_setting_and_the_original_chained():
    original = _Refused(404)
    with pytest.raises(ModelSettingError, match="The subagent role runs model") as info:
        ModelSettingErrors("subagent").wrap_model_call(None, _raising(original))
    assert info.value.__cause__ is original


def test_the_async_path_explains_it_too():
    with pytest.raises(ModelSettingError, match="valid together"):
        asyncio.run(ModelSettingErrors("root").awrap_model_call(None, _araising(_Refused(400))))


@pytest.mark.parametrize("exc", [_Refused(500), ValueError("our bug")])
def test_anything_else_passes_through_unchanged(exc: Exception):
    with pytest.raises(type(exc)) as info:
        ModelSettingErrors("root").wrap_model_call(None, _raising(exc))
    assert info.value is exc


def test_a_successful_call_is_returned_as_is():
    assert ModelSettingErrors("root").wrap_model_call(None, lambda _r: "ok") == "ok"
