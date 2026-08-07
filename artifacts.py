"""Publish files the agent writes in the sandbox to the UI as rendered artifacts.

The agent's deliverables — charts, spreadsheets, reports — are written inside the
sandbox, and the sandbox is deleted when the run ends. Nothing in the transcript
carries them out: `execute` returns stdout, and reading a PNG back through `read_file`
costs more context than the rest of the run (see the prompt).

This middleware is the way out. After every tool call that could have written
something, it lists `/workspace/out`, downloads whatever is new, and pushes one UI
message per file. **The bytes travel in the `ui` state key and never enter model
context** — that is the whole point, and it is why the prompt can keep telling the
model not to read its own plots back.

Routing is by extension: images render as charts, tabular files as tables, everything
else as a download card. The matching React components live in `ui.tsx`.

Only `/workspace/out` is swept, not all of `/workspace`. The agent's working files
(the abstracts bundle, scratch CSVs) live one level up and would otherwise show up in
the UI as noise. `out/` is the deliverables contract, and the prompt states it.
"""

from __future__ import annotations

import logging
import mimetypes
import posixpath
from typing import Annotated, Any, Sequence

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.graph.ui import AnyUIMessage, push_ui_message, ui_message_reducer

logger = logging.getLogger(__name__)

# Mirrored in the system prompt and baked into the snapshot by build_snapshot.py.
OUT_DIR = "/workspace/out"

# Artifacts ride in graph state, which is checkpointed on every write. A 50 MB xlsx
# would be re-serialised into every subsequent checkpoint on the thread and make the
# thread slow to load forever after. Past this size we announce the file and its size
# but leave the bytes in the sandbox — the user gets told it exists and why it isn't
# here, which is strictly better than a thread that won't open.
MAX_INLINE_BYTES = 8 * 1024 * 1024

# Which tools can leave a file behind. `eval` is in here because the interpreter can
# call writeFile and execute through PTC without either appearing as a tool call of
# its own — the whole design point of the interpreter is that one eval does the work
# of a dozen tool calls, so sweeping only on `execute` would miss most artifacts.
WRITER_TOOLS = frozenset({"eval", "execute", "write_file", "edit_file"})

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})
TABLE_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx", ".xls"})

# mimetypes doesn't know the modern Office types on every platform, and the browser
# needs the right one or an <img>/download of a .xlsx silently does the wrong thing.
EXTRA_MIME = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".tsv": "text/tab-separated-values",
    ".webp": "image/webp",
    ".md": "text/markdown",
}


def _component_for(suffix: str) -> str:
    if suffix in IMAGE_SUFFIXES:
        return "chart"
    if suffix in TABLE_SUFFIXES:
        return "table"
    return "file"


def _mime_for(path: str, suffix: str) -> str:
    if suffix in EXTRA_MIME:
        return EXTRA_MIME[suffix]
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


class ArtifactState(AgentState):
    """Agent state plus the `ui` channel the frontend renders from.

    `ui_message_reducer` is what makes repeated pushes accumulate instead of
    overwriting, and what lets a component be replaced by id later.
    """

    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]


class ArtifactMiddleware(AgentMiddleware):
    """Sweep the deliverables directory after each writing tool call and publish.

    Args:
        backend: The sandbox backend. Same object the agent's filesystem tools use,
            so this sees exactly the files the agent wrote.
        out_dir: Directory swept for deliverables.
        max_inline_bytes: Files larger than this are announced but not embedded.
    """

    state_schema = ArtifactState

    def __init__(
        self,
        backend: Any,
        *,
        out_dir: str = OUT_DIR,
        max_inline_bytes: int = MAX_INLINE_BYTES,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.out_dir = out_dir.rstrip("/")
        self.max_inline_bytes = max_inline_bytes
        # thread_id -> {path: fingerprint}. Keyed by thread because one server process
        # serves many threads; a single flat set would let thread A's sweep suppress
        # an identically-named chart in thread B, and the second user would silently
        # get no artifact at all.
        self._seen: dict[str, dict[str, str]] = {}

    async def awrap_tool_call(self, request: ToolCallRequest, handler):  # noqa: ANN001
        result = await handler(request)

        if request.tool_call.get("name") not in WRITER_TOOLS:
            return result

        try:
            await self._publish_new_files(request)
        except Exception:  # noqa: BLE001
            # Publishing is a side channel. A failure here must never take down a run
            # that has already done its expensive work — the user would lose the whole
            # analysis over a missing thumbnail.
            logger.warning("artifact sweep failed; continuing", exc_info=True)

        return result

    def _thread_key(self, request: ToolCallRequest) -> str:
        """Dedup scope. Falls back to a constant for the CLI, which has no thread."""
        try:
            configurable = (request.runtime.config or {}).get("configurable", {})
            return str(configurable.get("thread_id") or "default")
        except Exception:  # noqa: BLE001
            return "default"

    @staticmethod
    def _triggering_message_id(request: ToolCallRequest) -> Any:
        """The AI message whose tool call produced these files.

        agent-chat-ui renders a UI message inline by matching
        `ui.metadata.message_id` against the AI message it is drawing. Without this
        the artifact still arrives in state but has nowhere to appear in the
        transcript.
        """
        messages = (request.state or {}).get("messages") or []
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                return message
        return None

    async def _publish_new_files(self, request: ToolCallRequest) -> None:
        listing = await self.backend.aglob("**/*", self.out_dir)
        if listing.error or not listing.matches:
            return

        seen = self._seen.setdefault(self._thread_key(request), {})

        fresh: list[tuple[str, int]] = []
        for info in listing.matches:
            if info.get("is_dir"):
                continue
            path = info["path"]
            size = int(info.get("size") or 0)
            # size+mtime is the rsync heuristic: cheap, and wrong only if a file is
            # rewritten to the identical byte count within the filesystem's mtime
            # resolution. Hashing would mean downloading every file on every sweep,
            # which is the cost this check exists to avoid.
            fingerprint = f"{size}:{info.get('modified_at') or '?'}"
            if seen.get(path) == fingerprint:
                continue
            seen[path] = fingerprint
            fresh.append((path, size))

        if not fresh:
            return

        message = self._triggering_message_id(request)

        # Announce oversized files without downloading them at all — the point of the
        # cap is to not move the bytes.
        inline = [(p, s) for p, s in fresh if s <= self.max_inline_bytes]
        for path, size in fresh:
            if size > self.max_inline_bytes:
                self._push(path, size, data=None, message=message)

        if not inline:
            return

        downloads = await self.backend.adownload_files([p for p, _ in inline])
        for (path, size), response in zip(inline, downloads):
            if response.error or response.content is None:
                logger.warning("could not download %s: %s", path, response.error)
                continue
            self._push(path, size or len(response.content), response.content, message)

    def _push(
        self,
        path: str,
        size: int,
        data: bytes | None,
        message: Any,
    ) -> None:
        import base64

        name = posixpath.basename(path)
        suffix = posixpath.splitext(name)[1].lower()

        props: dict[str, Any] = {
            "name": name,
            "path": path,
            "size": size,
            "mime": _mime_for(path, suffix),
            "suffix": suffix,
            # base64 rather than raw bytes because this has to survive JSON
            # serialisation into the checkpointer and back out over SSE.
            "data": base64.b64encode(data).decode() if data is not None else None,
            "too_large": data is None,
        }

        push_ui_message(
            _component_for(suffix),
            props,
            # Stable per path, so a chart the agent regenerates replaces its previous
            # card instead of stacking a second copy under the same answer.
            id=f"artifact:{path}",
            metadata={"artifact_path": path},
            message=message,
        )
