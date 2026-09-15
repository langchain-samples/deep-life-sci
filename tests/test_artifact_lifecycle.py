"""Exercise publication, checkpoint deduplication, and retry at the backend boundary."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage

from deep_life_sci.middleware import artifacts


def request(thread="a", ui=None, tool="eval"):
    return SimpleNamespace(
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread}}),
        state={"messages": [AIMessage("", id="answer")], "ui": ui or []},
        tool_call={"name": tool},
    )


@pytest.fixture
def publication(monkeypatch):
    listing = [{"path": "/workspace/out/plot.png", "size": 4, "mtime": 1}]
    backend = SimpleNamespace(
        aexecute=AsyncMock(return_value=SimpleNamespace(output=json.dumps(listing[0]))),
        adownload_files=AsyncMock(return_value=[SimpleNamespace(error=None, content=b"plot")]),
    )
    push = Mock()
    monkeypatch.setattr(artifacts, "push_ui_message", push)
    return artifacts.ArtifactMiddleware(backend, max_inline_bytes=4), backend, push


async def test_publishes_bytes_with_stable_id_and_triggering_message(publication):
    middleware, backend, push = publication
    req = request()
    handler = AsyncMock(return_value="tool result")
    assert await middleware.awrap_tool_call(req, handler) == "tool result"
    backend.adownload_files.assert_awaited_once_with(["/workspace/out/plot.png"])
    args, kwargs = push.call_args
    assert args[0] == "chart"
    assert base64.b64decode(args[1]["data"]) == b"plot"
    assert args[1]["too_large"] is False
    assert kwargs["id"] == "artifact:/workspace/out/plot.png"
    assert kwargs["message"] is req.state["messages"][0]
    assert kwargs["metadata"]["artifact_fingerprint"] == "4:1"


async def test_checkpoint_suppresses_duplicates_but_other_threads_do_not(publication):
    middleware, backend, push = publication
    await middleware._publish_new_files(request())
    checkpoint = [{"metadata": push.call_args.kwargs["metadata"]}]
    fresh = artifacts.ArtifactMiddleware(backend)
    await fresh._publish_new_files(request(ui=checkpoint))
    assert push.call_count == 1
    await fresh._publish_new_files(request(thread="b"))
    assert push.call_count == 2


async def test_changed_file_replaces_the_existing_card(publication):
    middleware, backend, push = publication
    await middleware._publish_new_files(request())
    backend.aexecute.return_value.output = json.dumps(
        {"path": "/workspace/out/plot.png", "size": 4, "mtime": 2}
    )
    await middleware._publish_new_files(request())
    assert push.call_count == 2
    assert push.call_args_list[0].kwargs["id"] == push.call_args_list[1].kwargs["id"]
    assert push.call_args.kwargs["metadata"]["artifact_fingerprint"] == "4:2"


async def test_oversize_artifact_is_announced_without_download(publication):
    middleware, backend, push = publication
    middleware.max_inline_bytes = 3
    await middleware._publish_new_files(request())
    backend.adownload_files.assert_not_awaited()
    assert push.call_args.args[1]["data"] is None
    assert push.call_args.args[1]["too_large"] is True


@pytest.mark.parametrize("failure", ["response", "exception", "short", "publish"])
async def test_failed_publication_is_retried_on_the_next_sweep(publication, failure):
    middleware, backend, push = publication
    if failure == "response":
        backend.adownload_files.side_effect = [
            [SimpleNamespace(error="unavailable", content=None)],
            [SimpleNamespace(error=None, content=b"plot")],
        ]
    elif failure == "exception":
        backend.adownload_files.side_effect = [
            OSError("download failed"),
            [SimpleNamespace(error=None, content=b"plot")],
        ]
    elif failure == "short":
        backend.adownload_files.side_effect = [[], [SimpleNamespace(error=None, content=b"plot")]]
    else:
        push.side_effect = [RuntimeError("stream unavailable"), None]
    handler = AsyncMock(return_value="analysis survives")
    assert await middleware.awrap_tool_call(request(), handler) == "analysis survives"
    assert await middleware.awrap_tool_call(request(), handler) == "analysis survives"
    assert backend.adownload_files.await_count == 2
    assert push.call_count == (2 if failure == "publish" else 1)


async def test_read_only_tool_does_not_sweep(publication):
    middleware, backend, _ = publication
    await middleware.awrap_tool_call(request(tool="read_file"), AsyncMock())
    backend.aexecute.assert_not_awaited()


async def test_tool_failure_is_not_hidden_by_publication(publication):
    middleware, backend, _ = publication
    with pytest.raises(ValueError, match="tool bug"):
        await middleware.awrap_tool_call(request(), AsyncMock(side_effect=ValueError("tool bug")))
    backend.aexecute.assert_not_awaited()
