"""Real upload hooks and store with a backend that records staged bytes."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore

from deep_life_sci.middleware import uploads


def attachment(payload=b"gene,n\nTP53,1\n", name="counts.csv"):
    return {
        "type": "file",
        "mimeType": "text/csv",
        "metadata": {"filename": name},
        "data": base64.b64encode(payload).decode(),
    }


class Backend:
    def __init__(self):
        self.files = {}
        self.uploaded = []
        self.commands = []

    async def aupload_files(self, files):
        self.uploaded.extend(files)
        self.files.update(files)
        return [SimpleNamespace(error=None) for _ in files]

    async def aexecute(self, command):
        self.commands.append(command)
        records = [
            {
                "name": path.rsplit("/", 1)[-1],
                "path": path,
                "bytes": len(data),
                "kind": "tabular",
                "rows": 1,
                "columns": ["gene", "n"],
            }
            for path, data in self.files.items()
        ]
        return SimpleNamespace(output="\n".join(json.dumps(r) for r in records))


async def test_payload_is_replaced_in_place_and_restored_after_recycling(monkeypatch):
    thread = ["a"]
    monkeypatch.setattr(uploads, "_thread_key", lambda: thread[0])
    store = InMemoryStore()
    runtime = SimpleNamespace(store=store)
    backend = Backend()
    middleware = uploads.UploadMiddleware(backend)
    original = HumanMessage(
        id="question",
        content=[{"type": "text", "text": "Analyze this"}, attachment()],
        additional_kwargs={"client": "test"},
    )
    update = await middleware.abefore_agent({"messages": [original]}, runtime)
    messages = add_messages([original], update["messages"])
    assert len(messages) == 1
    assert messages[0].id == original.id
    assert messages[0].additional_kwargs == original.additional_kwargs
    assert messages[0].content[0] == original.content[0]
    assert all(block["type"] == "text" for block in messages[0].content)
    assert attachment()["data"] not in str(update)
    assert backend.uploaded == [(f"{uploads.UPLOAD_DIR}/counts.csv", b"gene,n\nTP53,1\n")]
    state = {"messages": messages, "upload_manifest": update["upload_manifest"]}
    assert await middleware.abefore_agent(state, runtime) is None
    assert len(backend.uploaded) == 1

    recycled = Backend()
    restored = await uploads.UploadMiddleware(recycled).abefore_agent(state, runtime)
    assert restored["upload_manifest"] == update["upload_manifest"]
    assert recycled.uploaded == backend.uploaded

    thread[0] = "b"
    other = Backend()
    result = await uploads.UploadMiddleware(other).abefore_agent(state, runtime)
    assert other.uploaded == []
    assert "no longer available" in result["upload_manifest"][0]["note"]


async def test_empty_turn_touches_neither_store_nor_backend():
    assert (
        await uploads.UploadMiddleware(None).abefore_agent(
            {"messages": [HumanMessage("Question")]}, runtime=None
        )
        is None
    )


async def test_rejected_upload_still_rewrites_the_human_message():
    update = await uploads.UploadMiddleware(None).abefore_agent(
        {"messages": [HumanMessage(id="q", content=[attachment(name="old.xls")])]},
        runtime=None,
    )
    assert update["messages"][0].id == "q"
    assert "re-save it as .xlsx" in update["messages"][0].content[0]["text"]
    assert "upload_manifest" not in update


async def test_store_failure_does_not_restore_payload_to_root_context():
    runtime = SimpleNamespace(store=SimpleNamespace(aput=AsyncMock(side_effect=OSError("down"))))
    update = await uploads.UploadMiddleware(None).abefore_agent(
        {"messages": [HumanMessage(id="q", content=[attachment()])]}, runtime
    )
    assert update["messages"][0].content[0]["type"] == "text"
    assert attachment()["data"] not in str(update)


async def test_missing_probe_record_is_reported_not_dropped():
    backend = SimpleNamespace(
        aexecute=AsyncMock(return_value=SimpleNamespace(output="")),
        aupload_files=AsyncMock(return_value=[SimpleNamespace(error=None)]),
    )
    manifest = await uploads.UploadMiddleware(backend)._reconcile({"counts.csv": b"data"}, [])
    assert len(manifest) == 1
    assert manifest[0]["name"] == "counts.csv"
    assert "probe did not report" in manifest[0]["note"]


async def test_replacement_upload_of_identical_size_replaces_the_old_bytes(monkeypatch):
    monkeypatch.setattr(uploads, "_thread_key", lambda: "same")
    runtime = SimpleNamespace(store=InMemoryStore())
    backend = Backend()
    middleware = uploads.UploadMiddleware(backend)
    first = await middleware.abefore_agent(
        {"messages": [HumanMessage(id="q1", content=[attachment(b"old")])]}, runtime
    )
    second = await middleware.abefore_agent(
        {
            "messages": [HumanMessage(id="q2", content=[attachment(b"new")])],
            "upload_manifest": first["upload_manifest"],
        },
        runtime,
    )
    assert backend.files[f"{uploads.UPLOAD_DIR}/counts.csv"] == b"new"
    assert second["upload_manifest"]


async def test_manifest_is_appended_using_a_real_model_request():
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    middleware = uploads.UploadMiddleware(None)
    req = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        system_prompt="base",
        messages=[HumanMessage("Question")],
        state={
            "upload_manifest": [
                {"name": "counts.csv", "path": "/uploads/counts.csv", "bytes": 3, "kind": "tabular"}
            ]
        },
        runtime=None,
    )
    handler = AsyncMock(return_value="done")
    assert await middleware.awrap_model_call(req, handler) == "done"
    prepared = handler.call_args.args[0]
    assert prepared.system_prompt.startswith("base\n\n<uploaded_files>")
    assert prepared.messages == req.messages
    assert req.system_prompt == "base"
