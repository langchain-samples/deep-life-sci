"""CLI streaming and environment precedence without launching an agent."""

import os
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from deep_life_sci import cli
from deep_life_sci.paths import REPO_ROOT


async def test_stream_prints_only_root_text(capsys):
    async def stream(inputs, **kwargs):
        assert inputs == {"messages": [{"role": "user", "content": "Question"}]}
        assert kwargs == {"stream_mode": "messages"}
        yield ToolMessage("private payload", tool_call_id="t"), {}
        yield AIMessageChunk("leaf text"), {"langgraph_checkpoint_ns": "root|leaf"}
        yield AIMessageChunk("Hello"), {"langgraph_checkpoint_ns": "model:1"}
        yield AIMessageChunk(""), {}
        yield AIMessageChunk(" world"), None

    await cli.stream_answer(SimpleNamespace(astream=stream), "Question")
    assert capsys.readouterr().out == "Hello world\n"


@pytest.mark.parametrize("path", ["deep_life_sci/cli.py", "evals/run.py"])
@pytest.mark.parametrize("override", ["", "medium"])
def test_dotenv_cannot_overwrite_explicit_model_axis(monkeypatch, path, override):
    import dotenv

    monkeypatch.setenv("ROOT_EFFORT", override)

    def fake_dotenv(**kwargs):
        assert kwargs == {"override": True}
        os.environ["ROOT_EFFORT"] = "high"

    monkeypatch.setattr(dotenv, "load_dotenv", fake_dotenv)
    runpy.run_path(str(REPO_ROOT / path), run_name="test_entrypoint")
    assert os.environ["ROOT_EFFORT"] == override


async def test_invalid_gateway_configuration_fails_before_sandbox_boot(monkeypatch):
    monkeypatch.setattr(cli, "check_gateway_config", Mock(side_effect=SystemExit("missing config")))
    session = Mock(side_effect=AssertionError("must not boot"))
    monkeypatch.setattr(cli, "sandbox_session", session)
    with pytest.raises(SystemExit, match="missing config"):
        await cli.main("Question")
    session.assert_not_called()
