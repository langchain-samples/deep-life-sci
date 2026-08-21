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

import json
import logging
import mimetypes
import posixpath
from collections.abc import Sequence
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.graph.ui import AnyUIMessage, push_ui_message, ui_message_reducer

# Mirrored in the system prompt and baked into the snapshot by scripts/build_snapshot.py.
from research_agent.paths import OUT_DIR

logger = logging.getLogger(__name__)

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


def _list_command(out_dir: str) -> str:
    """Shell command emitting one JSON object per file under `out_dir`.

    `aglob` would be the obvious way to list the directory, but its matches carry only
    `path` and `is_dir`. This middleware needs two more things: `size`, or the inline
    cap can't be enforced, and `mtime`, or a regenerated chart is indistinguishable
    from the one already published and the user keeps seeing a stale image.

    Doing the walk in Python rather than `find | stat` keeps filenames with spaces or
    non-ASCII characters intact, and Python is guaranteed present — running it is the
    reason the sandbox exists.
    """
    return (
        "python3 - <<'__ARTIFACTS_EOF__'\n"
        "import json, os\n"
        f"out = {out_dir!r}\n"
        "for root, _dirs, files in os.walk(out):\n"
        "    for name in files:\n"
        "        p = os.path.join(root, name)\n"
        "        try:\n"
        "            st = os.stat(p)\n"
        "        except OSError:\n"
        "            continue\n"
        '        print(json.dumps({"path": p, "size": st.st_size, '
        '"mtime": st.st_mtime}))\n'
        "__ARTIFACTS_EOF__"
    )


def _parse_listing(output: str) -> list[dict[str, Any]]:
    """Pull the JSON lines out of the command's combined output.

    `execute` returns stdout plus a trailing `[Command succeeded ...]` status line, and
    a missing `out/` produces no lines at all, so anything that isn't a JSON object is
    skipped rather than treated as an error.
    """
    files: list[dict[str, Any]] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            files.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return files


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

    def _known(self, request: ToolCallRequest) -> dict[str, str]:
        """Fingerprints already published on this thread: `{path: fingerprint}`.

        In-memory alone is not enough. The server rebuilds the graph — and so this
        middleware — once per run, while the sandbox is keyed to the thread and outlives
        it. A fresh `_seen` therefore sees turn 1's chart still sitting in `out/`, calls
        it new, and republishes it under turn 2's answer, which is how a chart reappears
        above a CSV the user asked for afterwards.

        The `ui` channel is checkpointed with the thread, so the artifacts already
        published are the durable record of what has been seen. In-memory entries are
        the fresher of the two (state lags within a turn) and win.
        """
        seen = self._seen.setdefault(self._thread_key(request), {})
        for ui_message in (request.state or {}).get("ui") or []:
            # `ui` also carries removal records and, in principle, entries pushed by
            # anything else; only well-formed artifact metadata counts.
            metadata = getattr(ui_message, "get", lambda _k: None)("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            path = metadata.get("artifact_path")
            fingerprint = metadata.get("artifact_fingerprint")
            if path and fingerprint:
                seen.setdefault(path, fingerprint)
        return seen

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        result = await handler(request)

        if request.tool_call.get("name") not in WRITER_TOOLS:
            return result

        try:
            await self._publish_new_files(request)
        except Exception:
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
        result = await self.backend.aexecute(_list_command(self.out_dir))
        listing = _parse_listing(getattr(result, "output", "") or "")
        if not listing:
            return

        seen = self._known(request)

        fresh: list[tuple[str, int, str]] = []
        for info in listing:
            path = info.get("path")
            if not path:
                continue
            size = int(info.get("size") or 0)
            # size+mtime is the rsync heuristic: cheap, and wrong only if a file is
            # rewritten to the identical byte count within the filesystem's mtime
            # resolution. Hashing would mean downloading every file on every sweep,
            # which is the cost this check exists to avoid.
            fingerprint = f"{size}:{info.get('mtime')}"
            if seen.get(path) == fingerprint:
                continue
            seen[path] = fingerprint
            fresh.append((path, size, fingerprint))

        if not fresh:
            return

        message = self._triggering_message_id(request)

        # Announce oversized files without downloading them at all — the point of the
        # cap is to not move the bytes.
        inline = [(p, s, f) for p, s, f in fresh if s <= self.max_inline_bytes]
        for path, size, fingerprint in fresh:
            if size > self.max_inline_bytes:
                self._push(path, size, None, message, fingerprint)

        if not inline:
            return

        downloads = await self.backend.adownload_files([p for p, _, _ in inline])
        # strict: one response per requested path. A short list would silently drop
        # artifacts from the sweep, which is invisible — the run just looks emptier.
        for (path, size, fingerprint), response in zip(inline, downloads, strict=True):
            if response.error or response.content is None:
                logger.warning("could not download %s: %s", path, response.error)
                # Nothing was published, so the fingerprint must not stick — otherwise
                # the retry on the next sweep is suppressed and the file never appears.
                self._seen.get(self._thread_key(request), {}).pop(path, None)
                continue
            self._push(
                path,
                size or len(response.content),
                response.content,
                message,
                fingerprint,
            )

    def _push(
        self,
        path: str,
        size: int,
        data: bytes | None,
        message: Any,
        fingerprint: str,
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
            # The fingerprint rides along so the next turn — which gets a brand new
            # middleware instance — can tell an already-published file from a new one
            # by reading state alone. See `_known`.
            metadata={"artifact_path": path, "artifact_fingerprint": fingerprint},
            message=message,
        )
