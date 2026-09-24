"""Verify provider SDK arguments without constructing a client or calling a model."""

from unittest.mock import Mock

import pytest

from deep_life_sci import models


@pytest.fixture
def constructors(monkeypatch):
    import langchain_anthropic
    import langchain_openai

    anthropic, openai = Mock(), Mock()
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", anthropic)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", openai)
    monkeypatch.setattr(models, "gateway_key", lambda: "test-placeholder")
    return {"anthropic": anthropic, "openai": openai}


@pytest.mark.parametrize(
    "provider,model",
    [
        ("anthropic", "claude-sonnet-5"),
        ("openai", "openai/gpt-5.6-terra"),
    ],
)
@pytest.mark.parametrize("role", ["root", "subagent", "judge", "search"])
def test_roles_construct_the_correct_provider_and_timeout(
    constructors, monkeypatch, provider, model, role
):
    monkeypatch.setenv(f"{role.upper()}_MODEL", model)
    monkeypatch.setenv(f"{role.upper()}_EFFORT", "high")
    factory = models.web_search_model if role == "search" else getattr(models, f"{role}_model")
    factory()
    selected = constructors[provider]
    selected.assert_called_once()
    constructors["openai" if provider == "anthropic" else "anthropic"].assert_not_called()
    kwargs = selected.call_args.kwargs
    assert kwargs["model"] == model
    assert kwargs["api_key"] == "test-placeholder"
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["base_url"] == getattr(models, f"{provider.upper()}_BASE_URL")
    assert (
        kwargs["timeout"]
        == {
            # ChatAnthropic takes only a float, so the Anthropic path keeps the read watchdog.
            "root": models.ROOT_TIMEOUT.read if provider == "anthropic" else models.ROOT_TIMEOUT,
            "subagent": models.SUBAGENT_TIMEOUT_SECONDS,
            "search": models.SEARCH_TIMEOUT_SECONDS,
            "judge": models.JUDGE_TIMEOUT_SECONDS,
        }[role]
    )
    if provider == "openai":
        assert kwargs["use_responses_api"] is True
    else:
        assert "use_responses_api" not in kwargs
    if role == "root":
        assert kwargs["streaming"] is True
    if role == "search":
        selected.return_value.bind_tools.assert_called_once_with(
            [models.WEB_SEARCH_SPECS[provider]]
        )
    else:
        selected.return_value.bind_tools.assert_not_called()


def test_explicit_caller_options_win_over_defaults(constructors):
    models.root_model(timeout=99, reasoning_effort="medium")
    kwargs = constructors["openai"].call_args.kwargs
    assert kwargs["timeout"] == 99
    assert kwargs["reasoning_effort"] == "medium"


def test_empty_effort_is_omitted_for_models_without_effort_support(constructors, monkeypatch):
    monkeypatch.setenv("SUBAGENT_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("SUBAGENT_EFFORT", "")
    models.subagent_model()
    assert "reasoning_effort" not in constructors["anthropic"].call_args.kwargs
