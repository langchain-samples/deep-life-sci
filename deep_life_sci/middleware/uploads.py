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

* **Model context** — where an attachment lands by default. A 2k-row CSV is 100-200k chars
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
and whatever shape the file turned out to have. That is the part worth context — a model
that knows the columns writes pandas that works first time — and it costs a few hundred
characters instead of a file. It changes only when an upload does, so the prompt-cache
prefix is stable across the turns of a conversation.

**Not every upload's useful form is the file.** `UPLOAD_KINDS` names six, and they divide
into two halves by what the agent does next:

* **A file to compute over** — a table, a compound set, a sequence set. The
  materialisation is the file itself, and the manifest describes its shape so the agent
  can open it with pandas or rdkit.
* **A corpus, or a document to read** — a bibliography, a PDF, an image. Here the file is
  a container for something else, and the manifest carries *that*: the PMID list a
  reference export resolves to, the path a PDF's text layer was written to, the path an
  image can be handed to `figure-analyst` at. This half is the point. A `.ris` out of
  Zotero becomes 200 PMIDs the agent hydrates with `fetchAbstracts` and fans out over —
  the operation the whole PTC design exists for — and a PDF is the ~70% of the literature
  that is not in PMC OA finally becoming readable, without its 40k characters landing in
  root context on the way.

Whatever a probe parses out goes to a sidecar under `paths.UPLOAD_DERIVED_DIR` rather than
into the manifest whenever it is bigger than a handful of lines. `upload_probe.py` is the
reader for all of it, and it runs in the sandbox because that is where the file is.

**It runs before the first model call, so it is latency the user watches.** That is why
it reads with the standard library and leaves pandas and rdkit to the agent: the first
import of either in a fresh container costs 2-11s of block fetching, against ~0.7s for
the whole stdlib probe. `sandbox.WARMUP` faults them in behind the run instead, so the
agent's own first `execute` does not pay for them either. See `upload_probe.py`'s
docstring for the measurements and for the two libraries that were worth keeping.

Only the root agent sees any of this. Subagents get their payload — or, for an upload, a
path — in their own prompts and do no I/O of their own (see `agent.py`).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import posixpath
import re
import shlex
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deep_life_sci.paths import UPLOAD_DERIVED_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)

# What the agent can do something with, and which probe in `upload_probe.py` describes it.
# This mapping is the single source of both: `UPLOAD_SUFFIXES` is derived from it, and the
# probe is handed it as JSON rather than carrying a second copy that could drift.
#
# `.xls` is absent on purpose: reading it needs `xlrd`, which is not in the snapshot, and
# `sandbox.py` blocks runtime installs — so accepting one would fail deep inside a run
# instead of at the composer. `_REJECTED` below turns it into a sentence the user reads
# before sending. Adding a format here means adding its reader to
# `scripts/build_snapshot.py` and rebuilding the snapshot, or every clone that already
# built one hits that install block.
#
# `.txt` is the one entry the suffix does not settle — a PMID list and a headerless TSV
# both arrive as one — so the probe sniffs its contents and may override this to
# `citations`. Everything else is decided here.
UPLOAD_KINDS: dict[str, str] = {
    # Tabular. `.gz` is handled by suffix-stripping rather than by an entry of its own
    # (see `_base_suffix`), which is what gets a real omics table under MAX_UPLOAD_BYTES.
    ".csv": "tabular",
    ".tsv": "tabular",
    ".txt": "tabular",
    ".xlsx": "tabular",
    ".xlsm": "tabular",
    # Bibliographies. The one upload kind whose useful materialisation is not the file but
    # the PMID list inside it — a reference manager export becomes a corpus the agent
    # hydrates with `fetchAbstracts` and fans out over.
    ".nbib": "citations",
    ".medline": "citations",
    ".ris": "citations",
    ".bib": "citations",
    ".bibtex": "citations",
    # A paper the agent cannot otherwise reach: ~70% of the literature is not in PMC OA.
    ".pdf": "pdf",
    # Compound sets. rdkit is already the heaviest thing in the snapshot.
    ".sdf": "chem",
    ".mol": "chem",
    ".smi": "chem",
    ".smiles": "chem",
    # Sequences. biopython is a parser here and nothing else — see the note on it in
    # `scripts/build_snapshot.py`.
    ".fasta": "sequence",
    ".fa": "sequence",
    ".fna": "sequence",
    ".faa": "sequence",
    ".gb": "sequence",
    ".gbk": "sequence",
    ".genbank": "sequence",
    # A gel, a blot, a panel out of a figure. Intercepted rather than left to ride into
    # model context, because `figure-analyst` reads an image from a path for a fraction of
    # what the root model pays to look at it.
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
}

UPLOAD_SUFFIXES = frozenset(UPLOAD_KINDS)

# Read once at import and shipped to the sandbox over a heredoc on the turns that need it.
# A file rather than a string literal because it is real Python that ruff should lint and
# a human should be able to read — see its module docstring for why it runs over there.
_PROBE_SOURCE = (Path(__file__).with_name("upload_probe.py")).read_text(encoding="utf-8")

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

# How many PMIDs a citation export may put in the manifest before it is described by its
# counts and its sidecar instead. A reference manager library runs to thousands, and the
# whole point of that path is that the identifiers reach `fetchAbstracts` rather than the
# prompt — the sidecar is how a long one gets there (see the citations block in
# `prompts/system.py`). A short bibliography is worth inlining: the model can fan out on
# it without a round trip to read the file back.
MAX_MANIFEST_IDS = 100

# How many images a PDF may contribute before the rest are counted rather than written.
# One per panel of a figure-heavy paper is already more than a run will look at, and each
# one costs a `figure-analyst` call to read.
MAX_PDF_FIGURES = 12

# Molecules, sequences and the like are listed as a handful of examples, never in full.
# The point is to show the model the shape of the records in the sidecar.
MAX_PREVIEW_ITEMS = 5

# Per thread. A user attaching twenty spreadsheets to one conversation is a mistake we
# should not silently absorb into every subsequent turn's materialisation.
MAX_FILES_PER_THREAD = 20

_NAMESPACE_ROOT = "uploads"

# What a chem sample line may carry, in the order it reads best. `smiles` comes from a
# `.smi`, `atoms` from an SDF or molfile, and `formula`/`mw` only from a manifest an older
# revision wrote with rdkit — see `_extra_lines`.
_CHEM_SAMPLE_KEYS = (("smiles", ""), ("formula", ""), ("mw", "g/mol"), ("atoms", "atoms"))

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


def _heredoc(script: str, **env: str) -> str:
    """Wrap a Python script for `execute`, quoted so the shell expands nothing in it.

    Configuration goes in as environment variables rather than by interpolating values
    into the source. `upload_probe.py` is a real file this repo lints and a human reads,
    and a `%(root)s` in the middle of it would make it neither.
    """
    prefix = "".join(f"{key}={shlex.quote(value)} " for key, value in sorted(env.items()))
    body = script if script.endswith("\n") else script + "\n"
    return prefix + "python3 - <<'__UPLOADS_EOF__'\n" + body + "__UPLOADS_EOF__"


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
    """The suffix that decides how a file is read, seeing through `.gz`.

    A DESeq2 table, a MAF and a GEO series matrix all routinely arrive gzipped, and
    compression is also what gets a real omics table under `MAX_UPLOAD_BYTES`. The file
    stays compressed on disk — pandas, `gzip` and biopython all read it that way — so
    this is only about picking the reader. Mirrored by `upload_probe.base_suffix`.
    """
    lowered = str(name or "").lower()
    if lowered.endswith(".gz"):
        lowered = lowered[: -len(".gz")]
    return posixpath.splitext(lowered)[1]


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


def _count(value: Any, singular: str, plural: str | None = None) -> str:
    """`1 molecule`, `12 molecules`. The manifest is prose the model reads, not a table."""
    number = int(value or 0)
    word = singular if number == 1 else (plural or singular + "s")
    return f"{number:,} {word}"


def _ids(record: dict[str, Any], key: str) -> str:
    """A capped identifier list, with the count of what was left out."""
    values = record.get(key) or []
    text = ", ".join(str(v) for v in values)
    if more := record.get("more_" + key):
        text += f", ... (+{more} more, all of them in the sidecar below)"
    return text


def _detail(record: dict[str, Any]) -> list[str]:
    """The one-line summary for a record, per kind.

    Every line here is a line in the root model's system prompt, so each field has to
    earn it: a shape it can write code against, or an identifier it can look up. See the
    rules at the top of `upload_probe.py`.
    """
    kind = record.get("kind") or "tabular"
    parts = [_human_size(record.get("bytes"))]

    if kind == "tabular":
        if record.get("rows") is not None:
            width = len(record.get("columns") or []) + int(record.get("more_columns") or 0)
            parts.append(f"{int(record['rows']):,} rows x {width} columns")
        if record.get("sheets"):
            parts.append("sheets: " + ", ".join(record["sheets"]))
    elif kind == "citations":
        parts.append(_count(record.get("references"), "reference"))
        if record.get("with_pmid") is not None:
            parts.append(f"{int(record['with_pmid']):,} with a PMID")
        if record.get("with_doi_only"):
            parts.append(f"{int(record['with_doi_only']):,} with a DOI but no PMID")
    elif kind == "pdf":
        if record.get("pages") is not None:
            parts.append(_count(record["pages"], "page"))
        if record.get("chars"):
            parts.append(f"{int(record['chars']):,} chars of text")
    elif kind == "chem":
        parts.append(_count(record.get("molecules"), "molecule"))
        if record.get("unparsed"):
            parts.append(f"{int(record['unparsed'])} unparseable")
    elif kind == "sequence":
        fmt = record.get("format") or "sequence"
        parts.append(_count(record.get("sequences"), f"{fmt} record"))
        if record.get("molecule"):
            parts.append(str(record["molecule"]))
        if record.get("total_residues"):
            parts.append(f"{int(record['total_residues']):,} residues total")
    elif kind == "image":
        if record.get("width"):
            fmt = record.get("format") or ""
            parts.append(f"{record['width']}x{record['height']} {fmt}".strip())

    return [p for p in parts if p]


def _extra_lines(record: dict[str, Any]) -> list[str]:
    """The indented lines under a record: columns, identifiers, sidecar paths, samples."""
    kind = record.get("kind") or "tabular"
    lines: list[str] = []

    if kind == "tabular" and record.get("columns"):
        line = "columns: " + ", ".join(record["columns"])
        if record.get("more_columns"):
            line += f", ... (+{record['more_columns']} more)"
        lines.append(line)
    elif kind == "citations":
        if record.get("pmids"):
            lines.append("PMIDs: " + _ids(record, "pmids"))
        if record.get("refs_path"):
            lines.append(
                f"parsed references (pmid, doi, title, journal, year, authors): "
                f"{record['refs_path']}"
            )
    elif kind == "pdf":
        if record.get("title"):
            lines.append(f"title: {record['title']}")
        for key in ("doi", "pmid"):
            if record.get(key):
                lines.append(f"{key} found in the text: {record[key]}")
        if record.get("text_path"):
            lines.append(f"extracted text ({int(record.get('lines') or 0):,} lines): "
                         f"{record['text_path']}")
        if record.get("figure_paths"):
            head = f"embedded images ({int(record.get('figures') or 0)})"
            if record.get("more_figures"):
                head += f", {int(record['more_figures'])} more not extracted"
            lines.append(head + " — for figure-analyst, not readFile:")
            lines.extend(f"  {path}" for path in record["figure_paths"])
        if record.get("figures_note"):
            lines.append(f"figures: {record['figures_note']}")
    elif kind == "chem":
        if record.get("properties"):
            line = "SDF properties: " + ", ".join(record["properties"])
            if record.get("more_properties"):
                line += f", ... (+{record['more_properties']} more)"
            lines.append(line)
        if record.get("mols_path"):
            lines.append(f"parsed molecules (name, smiles): {record['mols_path']}")
        # A `.smi` names its molecules with SMILES; an SDF or molfile names them with a
        # title and an atom count, because reading structure out of those needs rdkit and
        # `upload_probe.py` deliberately does not import it. Rendered from whichever keys
        # the sample has rather than a fixed set, which also keeps a manifest written by
        # an older revision (`formula`, `mw`) readable on a thread that spans a deploy.
        for sample in record.get("samples") or []:
            described = ", ".join(
                f"{sample[key]} {unit}".strip()
                for key, unit in _CHEM_SAMPLE_KEYS
                if sample.get(key) is not None
            )
            lines.append(f"e.g. {sample.get('name')}" + (f" — {described}" if described else ""))
    elif kind == "sequence":
        if record.get("seqs_path"):
            lines.append(f"parsed records (id, description, length): {record['seqs_path']}")
        unit = "bp" if record.get("molecule") == "nucleotide" else "aa"
        for sample in record.get("samples") or []:
            lines.append(
                f"e.g. {sample.get('id')} ({sample.get('length')} {unit}): "
                f"{sample.get('description')}"
            )

    if record.get("note"):
        lines.append(f"note: {record['note']}")
    return lines


def _render_manifest(manifest: list[dict[str, Any]]) -> str:
    """The block appended to the system prompt. Shapes and identifiers, never contents."""
    lines = []
    for record in manifest:
        # No kind means no probe ran — a recycled sandbox, or a probe that died before
        # reaching this file. Say so rather than falling back to a default: labelling an
        # undescribed file `tabular` invites exactly the code-against-a-shape-that-is-not-
        # there that the note beside it exists to prevent.
        kind = record.get("kind") or "unread"
        head = f"- {record.get('path') or record.get('name')} [{kind}] — "
        head += ", ".join(_detail(record))
        lines.append("\n  ".join([head, *_extra_lines(record)]))
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
        derived_dir: str = UPLOAD_DERIVED_DIR,
        max_bytes: int = MAX_UPLOAD_BYTES,
        max_files: int = MAX_FILES_PER_THREAD,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.upload_dir = upload_dir.rstrip("/")
        self.derived_dir = derived_dir.rstrip("/")
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
        # `image` as well as `file`, because a client that has not been taught this
        # agent's conventions still sends a PNG the way LangChain's multimodal helpers
        # build one — and an image left in place is a payload in root context, which is
        # the thing `figure-analyst` exists to avoid.
        if not isinstance(block, dict) or block.get("type") not in ("file", "image"):
            return None

        metadata = block.get("metadata") or {}
        mime = str(block.get("mimeType") or block.get("mime_type") or "")
        raw_name = metadata.get("filename") or metadata.get("name") or ""
        name = _safe_name(raw_name)
        suffix = _suffix(name)

        # A pasted screenshot arrives with no filename at all. Naming it off the MIME
        # type is the difference between staging it for `figure-analyst` and letting the
        # whole image through into root context because it had no extension to match.
        if not suffix and mime.startswith("image/"):
            suffix = "." + mime.partition("/")[2].split("+")[0].lower()
            if suffix in UPLOAD_SUFFIXES:
                name = f"{name}{suffix}"

        if suffix in _REJECTED:
            return {"name": name, "marker": f"[attachment {name} not read: {_REJECTED[suffix]}]"}
        if suffix not in UPLOAD_SUFFIXES:
            # Something the sandbox has no reader for. Left in place, so it reaches the
            # provider as whatever kind of block it is and fails — or works — on its own
            # terms rather than being silently swallowed here.
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
            "mime": mime or "application/octet-stream",
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
        "whatever arrived on this turn". That is enough for a one-shot run but not for
        turn 2, which `_reconcile` reports rather than papers over.
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
        """Describe every staged file, in the container the files are in.

        The one blocking sandbox call on the path to the first model call, which is why
        `upload_probe.py` keeps itself to the standard library — ~0.7s, against the 12s
        it cost when it imported pandas and rdkit. See its docstring for what each kind
        reports and what it deliberately leaves to the agent.
        """
        command = _heredoc(
            _PROBE_SOURCE,
            UPLOADS_ROOT=self.upload_dir,
            UPLOADS_DERIVED=self.derived_dir,
            UPLOADS_KINDS=json.dumps(UPLOAD_KINDS, separators=(",", ":")),
            UPLOADS_MAX_COLS=str(MAX_PREVIEW_COLUMNS),
            UPLOADS_MAX_ITEMS=str(MAX_PREVIEW_ITEMS),
            UPLOADS_MAX_IDS=str(MAX_MANIFEST_IDS),
            UPLOADS_MAX_FIGURES=str(MAX_PDF_FIGURES),
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
        known.update(lost)

        # Staged, but the probe never reported on it. `upload_probe.py` turns a failure to
        # read one file into a `note` and flushes each record as it goes, so this is the
        # narrow case where its driver died outright — and a manifest that is simply
        # shorter is indistinguishable from a smaller upload. The model has to be able to
        # say "I could not read the file you sent"; silence is the one answer it cannot
        # give. Reconciled against what we staged rather than against a re-inventory,
        # which would cost another round trip to say the same thing.
        manifest.extend(
            {
                "name": name,
                "path": f"{self.upload_dir}/{name}",
                "note": "could not be described — the probe did not report on this file",
            }
            for name in sorted((set(durable) | set(present)) - known)
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
