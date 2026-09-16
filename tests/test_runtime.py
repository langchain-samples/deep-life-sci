"""Assembly and entry points with real middleware and fake external boundaries."""

import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_life_sci import agent, runner
from deep_life_sci.middleware.artifacts import ArtifactMiddleware
from deep_life_sci.middleware.uploads import UploadMiddleware
from deep_life_sci.sources._errors import SourceError


@pytest.fixture
def assembled(monkeypatch):
    model = FakeListChatModel(responses=["unused"])
    monkeypatch.setattr(agent, "root_model", lambda: model)
    monkeypatch.setattr(agent, "subagent_model", lambda: model)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    compiled = Mock()
    create = Mock(return_value=compiled)
    monkeypatch.setattr(agent, "create_deep_agent", create)
    backend = SimpleNamespace()
    result = agent.build_agent(backend)
    assert result is compiled.with_config.return_value
    compiled.with_config.assert_called_once_with(recursion_limit=200)
    return create.call_args.kwargs, backend


def test_assembly_keeps_leaves_read_only_and_root_tools_callable(assembled):
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain_quickjs import CodeInterpreterMiddleware

    kwargs, backend = assembled
    assert kwargs["backend"] is backend
    assert {s["name"] for s in kwargs["subagents"]} == {
        "abstract-analyst",
        "full-text-analyst",
        "figure-analyst",
        "trial-analyst",
        "document-analyst",
    }
    for leaf in kwargs["subagents"]:
        assert leaf["tools"] == []
        (filesystem,) = leaf["middleware"]
        assert isinstance(filesystem, FilesystemMiddleware)
        expected = ["read_file", "grep"] if leaf["name"] == "trial-analyst" else ["read_file"]
        assert [tool.name for tool in filesystem.tools] == expected
    assert isinstance(kwargs["middleware"][0], UploadMiddleware)
    assert any(isinstance(m, ArtifactMiddleware) for m in kwargs["middleware"])
    (interpreter,) = [m for m in kwargs["middleware"] if isinstance(m, CodeInterpreterMiddleware)]
    assert interpreter._timeout == 900.0
    assert interpreter._max_result_chars == 40_000
    assert interpreter._max_ptc_calls == 512
    assert {tool.name for tool in kwargs["tools"]} <= set(interpreter._ptc)
    assert {t.name for t in kwargs["tools"]} == {
        "pubmed_search",
        "fetch_abstracts",
        "pmc_locate",
        "fetch_full_text",
        "fetch_figures",
        "fetch_supplementary",
        "ctgov_search",
        "ctgov_fetch",
        "web_search",
    }
    assert all(callable(t.coroutine) for t in kwargs["tools"])


async def test_assembled_source_tools_contain_failures(assembled, monkeypatch):
    from deep_life_sci.sources import pubmed

    kwargs, _ = assembled
    monkeypatch.setattr(pubmed, "_request", AsyncMock(side_effect=SourceError("API down")))
    tool = next(t for t in kwargs["tools"] if t.name == "pubmed_search")
    result = await tool.ainvoke({"term": "cancer", "retmax": 0})
    assert set(result) == {"error"}
    assert "API down" in result["error"]


@pytest.fixture
def graph_module(monkeypatch):
    # graph constructs a client on import; replacing that constructor keeps collection
    # independent of authentication and never changes graph._acquire itself.
    import langsmith.sandbox

    monkeypatch.setattr(langsmith.sandbox, "SandboxClient", Mock())
    from deep_life_sci import graph

    monkeypatch.setattr(graph, "_client", Mock())
    monkeypatch.setattr(graph, "_sandbox_names", {})
    monkeypatch.setattr(graph, "warm", Mock())
    return graph


@pytest.mark.parametrize(
    "config", [{}, {"configurable": {"thread_id": "a", "__is_for_execution__": False}}]
)
async def test_graph_reads_never_acquire_a_sandbox(graph_module, monkeypatch, config):
    graph = graph_module
    acquire = Mock(side_effect=AssertionError("must not acquire"))
    monkeypatch.setattr(graph, "_acquire", acquire)
    monkeypatch.setattr(graph, "build_agent", lambda backend: backend)
    backend = await graph.make_graph(config)
    assert isinstance(backend._sandbox, graph._UnboundSandbox)
    acquire.assert_not_called()
    with pytest.raises(RuntimeError, match="unbound"):
        backend._sandbox.run("echo ok")


async def test_graph_run_acquires_off_loop_and_retains_recovery_callback(graph_module, monkeypatch):
    graph = graph_module
    loop_thread = threading.get_ident()
    raw = SimpleNamespace()

    def acquire(key):
        assert threading.get_ident() != loop_thread
        assert key == "42"
        return raw

    monkeypatch.setattr(graph, "_acquire", acquire)
    monkeypatch.setattr(graph, "build_agent", lambda backend: backend)
    backend = await graph.make_graph({"configurable": {"thread_id": 42}})
    assert backend._sandbox is raw
    assert await backend._arebind(0) is True


@pytest.mark.parametrize("status", ["running", "stopped"])
def test_existing_graph_sandbox_is_reused(graph_module, monkeypatch, status):
    graph = graph_module
    existing = SimpleNamespace(status=status)
    graph._client.get_sandbox.return_value = existing
    boot = Mock()
    monkeypatch.setattr(graph, "boot", boot)
    assert graph._acquire("thread") is existing
    boot.assert_not_called()
    assert graph.warm.call_count == (1 if status == "stopped" else 0)


@pytest.mark.parametrize("snapshot", [None, "snapshot"])
def test_expired_graph_sandbox_is_recreated(graph_module, monkeypatch, snapshot):
    graph = graph_module
    graph._client.get_sandbox.side_effect = OSError("gone")
    monkeypatch.setattr(graph, "find_snapshot", lambda client: snapshot)
    boot, provision = Mock(), Mock()
    monkeypatch.setattr(graph, "boot", boot)
    monkeypatch.setattr(graph, "provision", provision)
    assert graph._acquire("thread") is boot.return_value
    assert provision.call_count == (1 if snapshot is None else 0)
    assert graph.warm.call_count == (0 if snapshot is None else 1)


async def test_runner_returns_actual_trajectory_and_artifact_component_names(monkeypatch):
    messages = [
        HumanMessage("Question"),
        AIMessage("", tool_calls=[{"name": "eval", "args": {}, "id": "call"}]),
        AIMessage("Answer"),
        ToolMessage("done", tool_call_id="call"),
    ]
    ui = [{"name": "chart", "props": {"name": "plot.png"}}]
    compiled = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": messages, "ui": ui}))
    monkeypatch.setattr(runner, "build_agent", lambda backend: compiled)
    result = await runner.run_once("Question", backend=object())
    assert result.answer == "Answer"
    assert result.artifacts == ui
    assert result.root_turns == 2
    assert result.tool_calls == ["eval"]
    assert result.root_context_chars == len("QuestionAnswerdone")
    assert result.as_dict()["artifact_names"] == ["chart"]
    assert "messages" not in result.as_dict()
    compiled.ainvoke.assert_awaited_once_with(
        {"messages": [{"role": "user", "content": "Question"}]}
    )


async def test_runner_owned_session_closes_when_agent_raises(monkeypatch):
    events = []

    @asynccontextmanager
    async def session(**kwargs):
        events.append(("open", kwargs))
        try:
            yield object()
        finally:
            events.append("close")

    monkeypatch.setattr(runner, "sandbox_session", session)
    monkeypatch.setattr(runner, "_run", AsyncMock(side_effect=ValueError("agent failed")))
    with pytest.raises(ValueError, match="agent failed"):
        await runner.run_once("Question", quiet=False)
    assert events == [("open", {"quiet": False}), "close"]


async def test_real_compiled_agent_runs_quickjs_source_tool_and_returns_answer(
    monkeypatch, mock_ncbi
):
    import httpx
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ScriptedModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "eval",
                        "id": "search",
                        "args": {
                            "code": 'const r = await tools.pubmedSearch('
                            '{term: "cancer", retmax: 0}); '
                            'console.log("COUNT=" + r.count);'
                        },
                    }
                ],
            ),
            AIMessage("Found 7 papers."),
        ]
    )
    monkeypatch.setattr(agent, "root_model", lambda: model)
    monkeypatch.setattr(agent, "subagent_model", lambda: model)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    transport = mock_ncbi(
        lambda req: httpx.Response(
            200, json={"esearchresult": {"count": "7", "idlist": [], "querytranslation": "cancer"}}
        )
    )
    from deep_life_sci.sandbox import ResilientSandbox

    backend = ResilientSandbox(SimpleNamespace())
    monkeypatch.setattr(backend, "aexecute", AsyncMock(return_value=SimpleNamespace(output="")))
    compiled = agent.build_agent(backend)
    result = await compiled.ainvoke({"messages": [HumanMessage("Count cancer papers")]})
    assert result["messages"][-1].text == "Found 7 papers."
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert "COUNT=7" in tool_messages[0].text
    assert len(transport.requests) == 1
    assert transport.requests[0].url.params["term"] == "cancer"
    backend.aexecute.assert_awaited_once()  # real artifact middleware runs after eval


async def test_one_ptc_script_fetches_stages_and_dispatches_file_reading_trial_analyst(
    monkeypatch, mock_ctgov, tmp_path
):
    import json

    from deepagents.backends.filesystem import FilesystemBackend
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    from deep_life_sci.sandbox import ResilientSandbox
    from deep_life_sci.sources import ctgov, trial_files
    from tests.conftest import json_response
    from tests.test_trial_files import study

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    payload = study()
    manifest, uploads = trial_files._prepare(ctgov._study_to_record(payload))
    index = json.loads(dict(uploads)[manifest["index_path"]])
    section = next(x for x in index if x["section"].startswith("Outcome"))
    leaf = ScriptedModel(responses=[
        AIMessage("", tool_calls=[{"name": "read_file", "id": "index", "args": {
            "file_path": manifest["index_path"], "limit": 1000}}]),
        AIMessage("", tool_calls=[{"name": "grep", "id": "search-outcome", "args": {
            "pattern": "Body weight", "path": manifest["index_path"].rsplit("/", 1)[0],
            "glob": "section-*.json", "output_mode": "files_with_matches"}}]),
        AIMessage("", tool_calls=[{"name": "read_file", "id": "outcome", "args": {
            "file_path": section["path"], "limit": 1000}}]),
        AIMessage("Registry treatment difference: -12.44 percentage points."),
    ])
    root = ScriptedModel(responses=[
        AIMessage("", tool_calls=[{"name": "eval", "id": "fetch-and-read", "args": {
            "code": '''
const fetched = await tools.ctgovFetch({nct_ids: ["NCT00000001"], include: ["results"]});
const answers = await Promise.all(Object.values(fetched.records).map(async (t) => ({
  nct_id: t.nct_id,
  answer: await task({
    description: "Find the treatment difference. Trial record: " + JSON.stringify(t),
    subagentType: "trial-analyst"
  })
})));
await tools.writeFile({file_path: "/workspace/trial-answers.json",
                      content: JSON.stringify(answers)});
console.log(JSON.stringify(answers));
'''}}]),
        AIMessage("The treatment difference was -12.44 percentage points."),
    ])
    monkeypatch.setattr(agent, "root_model", lambda: root)
    monkeypatch.setattr(agent, "subagent_model", lambda: leaf)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    transport = mock_ctgov(lambda r: json_response({"studies": [payload]}))
    disk = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = ResilientSandbox(SimpleNamespace())
    monkeypatch.setattr(backend, "aupload_files", disk.aupload_files)
    reads = AsyncMock(side_effect=disk.aread)
    monkeypatch.setattr(backend, "aread", reads)
    searches = AsyncMock(side_effect=disk.agrep)
    monkeypatch.setattr(backend, "agrep", searches)
    monkeypatch.setattr(backend, "awrite", disk.awrite)
    monkeypatch.setattr(backend, "aexecute", AsyncMock(return_value=SimpleNamespace(output="")))
    compiled = agent.build_agent(backend)
    result = await compiled.ainvoke({"messages": [HumanMessage("Find the trial result")]})
    tools = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tools) == 1
    assert "-12.44" in tools[0].text
    assert "UNREAD" not in tools[0].text
    assert len(transport.requests) == 1
    read_paths = [call.args[0] if call.args else call.kwargs.get("file_path")
                  for call in reads.call_args_list]
    assert read_paths == [manifest["index_path"], section["path"]]
    searches.assert_awaited_once()
    saved = (tmp_path / "workspace/trial-answers.json").read_text()
    assert "-12.44" in saved
