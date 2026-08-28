"""Bring the user's own data files into the sandbox, and keep them there.

`middleware/artifacts.py` carries files *out* of the sandbox. This carries them *in*, and
the hard part is not the transport but the lifetime.

**A sandbox is not storage.** `paths.IDLE_TTL_SECONDS` reaps a container ten idle minutes
after a turn, and `graph.py:_acquire` then hands the same thread a brand new empty one —
the same thing `ResilientSandbox` does mid-run when a container dies under it. A CSV
written into `/workspace` is therefore gone by the time the user comes back from lunch and
asks a follow-up about it. So the sandbox cannot hold the upload; it holds a
*materialisation* of it, rebuilt from a durable copy on every turn. That rebuild is the
whole point of this module, and it is what makes turn 2 work.

The durable copy lives in the LangGraph store, namespaced per thread. Three places were
possible and the store is the only one that is all three of durable, out of model context,
and free of per-checkpoint cost:

* **Model context** — what upstream's attachment flow does. A 2k-row CSV is 100-200k chars
  re-sent on every turn, in an agent whose entire design is that payloads never reach the
  root transcript (`pmc_locate` exists for exactly this reason). The model also could not
  compute over it without retyping the file into `writeFile`.
* **Graph state** — never enters context; this is how the `ui` key carries artifacts out.
  But state is re-serialised into every checkpoint, which is the cost
  `artifacts.MAX_INLINE_BYTES` exists to bound. A 15 MB workbook would ride in every
  checkpoint on the thread forever, and the thread would be slow to open for good.
* **The store** — written once, read on the turns that need it. Postgres-backed in deploy,
  pickled to `.langgraph_api/store.pckl` under `langgraph dev`.

The bytes still arrive *through* model context's front door, because the chat UI has no
upload endpoint of its own: an attachment rides in as a content block on the human message.
`before_agent` runs before the first model call, so this harvests those blocks, moves them
to the store and the sandbox, and rewrites the message to a one-line marker — keeping the
message id, which `add_messages` treats as a replacement rather than an append. The payload
is consequently in exactly one checkpoint (the input write) and in no request to the model.

What the model gets instead is a manifest appended to the system prompt: filename, size,
row count, column names. That is the part worth context — a model that knows the columns
writes pandas that works first time — and it costs a few hundred characters instead of a
file. It changes only when an upload does, so the prompt-cache prefix is stable across the
turns of a conversation.

Only the root agent sees any of this. Subagents get their payload in their own prompts and
do no I/O (see `agent.py`), which is unchanged: an upload is root-level analysis.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import posixpath
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from research_agent.paths import UPLOAD_DIR

logger = logging.getLogger(__name__)

# What the agent can actually do something with. `.xls` is absent on purpose: reading it
# needs `xlrd`, which is not in the snapshot, and `sandbox.py` blocks runtime installs — so
# accepting one would fail deep inside a run instead of at the composer. `_REJECTED` below
# turns it into a sentence the user reads before sending. Adding a format here means adding
# its reader to `scripts/build_snapshot.py` and rebuilding the snapshot, or every clone that
# already built one hits that install block.
UPLOAD_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx", ".xlsm"})

# Formats a user plausibly attaches and we deliberately decline. Stripped from the message
# like a real upload, but with the reason in place of a path — left in, the block reaches a
# provider that has no such document type and answers with a 400 for the whole run.
_REJECTED = {
    ".xls": "legacy .xls isn't readable here — re-save it as .xlsx and attach that",
}

# Deliberately above `artifacts.MAX_INLINE_BYTES` (8 MB), which is a different direction and
# a different cost: that one bounds bytes leaving the sandbox into graph state once, this one
# bounds bytes sitting in a store row and being re-uploaded into a container on every cold
# turn. Raising it is not free — an attachment arrives base64'd on the human message, so 15 MB
# is ~20 MB in the input checkpoint and ~20 MB in a Postgres jsonb row, times up to
# `MAX_FILES_PER_THREAD`. It is a ceiling on what we accept rather than a size to design for.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# A wide frame's column list is real context cost for diminishing return; past this the
# count of what was elided is enough for the model to know to inspect the rest itself.
MAX_PREVIEW_COLUMNS = 40

# Per thread. A user attaching twenty spreadsheets to one conversation is a mistake we
# should not silently absorb into every subsequent turn's materialisation.
MAX_FILES_PER_THREAD = 20

_NAMESPACE_ROOT = "uploads"

# Upload names become sandbox paths, so anything that could traverse or quote-break is out.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

_INVENTORY_SCRIPT = """\
import json, os
root = %(root)r
if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            print(json.dumps({"name": name, "bytes": os.path.getsize(path)}))
"""

# Runs in the sandbox after materialisation. pandas is the reader rather than csv/openpyxl
# directly because pandas is what the agent will use, so a file this parses is a file the
# agent's own code can open — and a file it cannot parse is worth saying so about here,
# before the model writes a script against a shape that does not exist.
_PROBE_SCRIPT = """\
import json, os
root = %(root)r
max_cols = %(max_cols)d
try:
    import pandas as pd
except Exception:
    pd = None
names = sorted(os.listdir(root)) if os.path.isdir(root) else []
for name in names:
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        continue
    rec = {"name": name, "path": path, "bytes": os.path.getsize(path)}
    if pd is None:
        rec["note"] = "pandas unavailable in this sandbox"
    else:
        try:
            frame = None
            if name.lower().endswith((".xlsx", ".xlsm")):
                sheets = pd.read_excel(path, sheet_name=None)
                rec["sheets"] = [str(s) for s in sheets]
                frame = next(iter(sheets.values())) if sheets else None
            else:
                sep = "\\t" if name.lower().endswith(".tsv") else ","
                frame = pd.read_csv(path, sep=sep)
            if frame is not None:
                rec["rows"] = int(frame.shape[0])
                cols = [str(c) for c in frame.columns]
                rec["columns"] = cols[:max_cols]
                if len(cols) > max_cols:
                    rec["more_columns"] = len(cols) - max_cols
        except Exception as exc:
            rec["note"] = ("could not parse: " + type(exc).__name__ + ": " + str(exc))[:200]
    print(json.dumps(rec))
"""


def _heredoc(script: str) -> str:
    """Wrap a Python script for `execute`, quoted so the shell expands nothing in it."""
    return "python3 - <<'__UPLOADS_EOF__'\n" + script + "__UPLOADS_EOF__"


def _parse_lines(output: str) -> list[dict[str, Any]]:
    """JSON objects, one per line, ignoring anything else the shell printed."""
    records = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _safe_name(raw: str) -> str:
    """A filename that is safe to interpolate into a sandbox path."""
    name = posixpath.basename(str(raw or "").strip()).lstrip(".")
    name = _UNSAFE.sub("_", name)
    return name[:80] or "upload"


def _suffix(name: str) -> str:
    return posixpath.splitext(name)[1].lower()


def _thread_key() -> str:
    """Store namespace scope. Falls back to a constant for the CLI, which has no thread.

    `Runtime` deliberately carries no `config` (unlike the `ToolRuntime` the tool-call
    hooks get), so the thread id has to come from the ambient runnable config.
    """
    try:
        return str((get_config().get("configurable") or {}).get("thread_id") or "default")
    except Exception:  # noqa: BLE001 - outside a runnable context entirely
        return "default"


def _human_size(size: Any) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "unknown size"
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} MB"


def _render_manifest(manifest: list[dict[str, Any]]) -> str:
    """The block appended to the system prompt. Shapes and names, never contents."""
    lines = []
    for record in manifest:
        parts = [_human_size(record.get("bytes"))]
        if record.get("rows") is not None:
            columns = record.get("columns") or []
            width = len(columns) + int(record.get("more_columns") or 0)
            parts.append(f"{int(record['rows']):,} rows x {width} columns")
        if record.get("sheets"):
            parts.append("sheets: " + ", ".join(record["sheets"]))
        detail = ", ".join(parts)
        line = f"- {record.get('path') or record.get('name')} — {detail}"
        if record.get("columns"):
            line += "\n  columns: " + ", ".join(record["columns"])
            if record.get("more_columns"):
                line += f", ... (+{record['more_columns']} more)"
        if record.get("note"):
            line += f"\n  note: {record['note']}"
        lines.append(line)
    return "<uploaded_files>\n" + "\n".join(lines) + "\n</uploaded_files>"


class UploadState(AgentState):
    """Agent state plus the manifest describing this thread's uploads.

    Only the description is checkpointed, never the bytes — that split is the reason this
    key is safe to carry on every turn while the payload is not.
    """

    upload_manifest: list[dict[str, Any]]


class UploadMiddleware(AgentMiddleware):
    """Materialise the thread's uploaded files into the sandbox before the agent runs.

    Args:
        backend: The sandbox backend, the same object the agent's filesystem tools use,
            so files land where the agent's own `execute` will find them.
        upload_dir: Directory inside the sandbox that uploads are materialised into.
        max_bytes: Per-file ceiling. Larger attachments are refused with a note.
        max_files: Per-thread ceiling on materialised files.
    """

    state_schema = UploadState

    def __init__(
        self,
        backend: Any,
        *,
        upload_dir: str = UPLOAD_DIR,
        max_bytes: int = MAX_UPLOAD_BYTES,
        max_files: int = MAX_FILES_PER_THREAD,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.upload_dir = upload_dir.rstrip("/")
        self.max_bytes = max_bytes
        self.max_files = max_files

    # -- harvesting ------------------------------------------------------------------

    def _harvest(self, state: Any) -> tuple[list[dict[str, Any]], list[HumanMessage]]:
        """Pull upload payloads out of the human messages, and rewrite those messages.

        Returns the files found and replacement messages carrying the same ids, so the
        `add_messages` reducer swaps them in place of the originals rather than appending.
        Messages already stripped on an earlier turn contain no payload blocks and so
        produce nothing here, which is what makes this safe to run on every turn.
        """
        files: list[dict[str, Any]] = []
        rewrites: list[HumanMessage] = []

        for message in (state or {}).get("messages") or []:
            if getattr(message, "type", None) != "human":
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue

            changed = False
            blocks: list[Any] = []
            for block in content:
                harvested = self._read_block(block)
                if harvested is None:
                    blocks.append(block)
                    continue
                changed = True
                # The marker is what the transcript keeps. It has to name the file, or a
                # reloaded thread shows a question about data with no sign of the data.
                blocks.append({"type": "text", "text": harvested["marker"]})
                if harvested.get("data") is not None:
                    files.append(harvested)

            if changed:
                rewrites.append(
                    HumanMessage(
                        id=getattr(message, "id", None),
                        content=blocks,
                        additional_kwargs=getattr(message, "additional_kwargs", {}) or {},
                    )
                )

        return files, rewrites

    def _read_block(self, block: Any) -> dict[str, Any] | None:
        """Interpret one content block. `None` means "not an upload, leave it alone"."""
        if not isinstance(block, dict) or block.get("type") != "file":
            return None

        metadata = block.get("metadata") or {}
        raw_name = metadata.get("filename") or metadata.get("name") or ""
        name = _safe_name(raw_name)
        suffix = _suffix(name)

        if suffix in _REJECTED:
            return {"name": name, "marker": f"[attachment {name} not read: {_REJECTED[suffix]}]"}
        if suffix not in UPLOAD_SUFFIXES:
            # A PDF or an image is a model-context attachment rather than a data file, and
            # not something this middleware has any business intercepting.
            return None

        data = block.get("data")
        if not isinstance(data, str):
            return {"name": name, "marker": f"[attachment {name} not read: no payload]"}
        try:
            payload = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return {"name": name, "marker": f"[attachment {name} not read: undecodable]"}

        if len(payload) > self.max_bytes:
            limit = _human_size(self.max_bytes)
            return {
                "name": name,
                "marker": (
                    f"[attachment {name} not read: {_human_size(len(payload))} exceeds the "
                    f"{limit} upload limit]"
                ),
            }

        path = f"{self.upload_dir}/{name}"
        return {
            "name": name,
            "path": path,
            "bytes": len(payload),
            "mime": block.get("mimeType") or "application/octet-stream",
            "data": payload,
            # No path in it: the UI joins these text blocks into the user's own chat bubble
            # (`getContentString`), so a sandbox path here shows up inside their question.
            # The manifest is where paths belong, and the prompt calls it the whole inventory.
            "marker": f"[attached {name} ({_human_size(len(payload))})]",
        }

    # -- durable copy ----------------------------------------------------------------

    async def _durable(
        self, store: Any, thread: str, harvested: list[dict[str, Any]]
    ) -> dict[str, bytes]:
        """Persist what just arrived, then return every upload this thread owns.

        Without a store — the CLI, or a server configured without one — this degrades to
        "whatever arrived on this turn". That is enough for a one-shot run and honestly
        insufficient for turn 2, which `_reconcile` reports rather than papers over.
        """
        if store is None:
            return {file["name"]: file["data"] for file in harvested}

        namespace = (_NAMESPACE_ROOT, thread)
        for file in harvested:
            await store.aput(
                namespace,
                file["name"],
                {
                    "name": file["name"],
                    "mime": file["mime"],
                    "bytes": file["bytes"],
                    # base64 rather than raw bytes: a store value is JSON, and in deploy
                    # it is a Postgres jsonb column.
                    "data": base64.b64encode(file["data"]).decode(),
                },
            )

        durable: dict[str, bytes] = {}
        for item in await store.asearch(namespace, limit=self.max_files):
            value = item.value or {}
            encoded = value.get("data")
            if not isinstance(encoded, str):
                continue
            try:
                durable[str(value.get("name") or item.key)] = base64.b64decode(encoded)
            except (binascii.Error, ValueError):
                logger.warning("upload %r in the store is undecodable; skipping", item.key)
        return durable

    # -- sandbox ---------------------------------------------------------------------

    async def _inventory(self) -> dict[str, int]:
        """What is already in the sandbox's upload directory, by name and size."""
        command = f"mkdir -p {self.upload_dir} && " + _heredoc(
            _INVENTORY_SCRIPT % {"root": self.upload_dir}
        )
        result = await self.backend.aexecute(command)
        return {
            str(record["name"]): int(record.get("bytes") or 0)
            for record in _parse_lines(getattr(result, "output", "") or "")
            if record.get("name")
        }

    async def _probe(self) -> list[dict[str, Any]]:
        command = _heredoc(
            _PROBE_SCRIPT % {"root": self.upload_dir, "max_cols": MAX_PREVIEW_COLUMNS}
        )
        result = await self.backend.aexecute(command)
        return _parse_lines(getattr(result, "output", "") or "")

    async def _reconcile(
        self, durable: dict[str, bytes], prior: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Make the sandbox match the durable set. Returns a manifest, or `None` if
        nothing changed and `prior` still describes it accurately."""
        present = await self._inventory()

        # size, not a hash: the alternative is downloading every file back out of the
        # container on every turn to compare, which is the cost this check avoids. A
        # collision needs a different file of identical length under the same name.
        pending = [
            (f"{self.upload_dir}/{name}", payload)
            for name, payload in durable.items()
            if present.get(name) != len(payload)
        ]

        if pending:
            responses = await self.backend.aupload_files(pending)
            for (path, _), response in zip(pending, responses, strict=True):
                if getattr(response, "error", None):
                    logger.warning("could not stage upload %s: %s", path, response.error)

        # Named in a prior manifest but neither in the container nor recoverable. Only
        # reachable without a store; with one the durable copy is what refills the
        # container. The model is told, because "the file you were told about is gone" is
        # something it has to be able to say to the user.
        lost = [
            name
            for name in (record.get("name") for record in prior)
            if name and name not in durable and name not in present
        ]

        if not pending and not lost and prior:
            return None

        manifest = await self._probe()
        known = {record.get("name") for record in manifest}
        manifest.extend(
            {
                "name": name,
                "path": f"{self.upload_dir}/{name}",
                "note": "no longer available — this thread's sandbox was recycled",
            }
            for name in lost
            if name not in known
        )
        return manifest

    # -- hooks -----------------------------------------------------------------------

    async def abefore_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Harvest, persist, materialise. Runs before the first model call of every turn.

        The fast path matters more than the slow one: a thread with no uploads has an
        empty manifest and no payload blocks, and returns here without touching the store
        or the sandbox at all. Which is every thread in the demo's normal use.
        """
        prior = list((state or {}).get("upload_manifest") or [])
        try:
            harvested, rewrites = self._harvest(state)
        except Exception:
            logger.warning("harvesting uploads failed; continuing without them", exc_info=True)
            return None

        update: dict[str, Any] = {"messages": rewrites} if rewrites else {}

        # Nothing to stage and nothing staged before: no store or sandbox traffic at all.
        # A rewrite can still be pending on this path — an attachment declined outright,
        # `.xls` or oversize — and returning `None` here would leave the payload we just
        # refused in the message and send it to the model, which is the one outcome the
        # rejection exists to prevent.
        if not harvested and not prior:
            return update or None

        try:
            durable = await self._durable(runtime.store, _thread_key(), harvested)
            manifest = await self._reconcile(durable, prior)
        except Exception:
            # A failure here must not take down the run. The rewrites still apply, so the
            # payload does not leak into the model's context, and the manifest keeps
            # whatever it had — the model then finds the file absent and says so, which
            # is a worse answer but not a dead thread.
            logger.warning("staging uploads failed; continuing", exc_info=True)
            return update or None

        if manifest is not None:
            update["upload_manifest"] = manifest
        return update or None

    async def awrap_model_call(self, request, handler):
        """Append the manifest to the system prompt.

        In the prompt rather than as a message so it cannot be summarised away mid-run,
        and appended rather than baked into `SYSTEM_PROMPT` so a thread without uploads
        carries none of it. It changes only when an upload does, which keeps the cached
        prefix stable across the turns of a conversation.
        """
        manifest = (request.state or {}).get("upload_manifest") or []
        if not manifest:
            return await handler(request)

        base = request.system_prompt or ""
        return await handler(
            request.override(system_prompt=f"{base}\n\n{_render_manifest(manifest)}")
        )
