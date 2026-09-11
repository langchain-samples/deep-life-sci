"""Describe the user's uploaded files. Runs **inside the sandbox**, never in this process.

`middleware/uploads.py` reads this file as text and pipes it to the sandbox's `python3`
over a heredoc. That is why it imports nothing from `research_agent` and takes its
configuration from the environment: there is no package here, only an interpreter and the
libraries `scripts/build_snapshot.py` baked in.

It runs there because that is where the file is: the bytes live in the container, the
host's dependency set deliberately holds no readers (this repo's host half talks to NCBI
and nothing else), and a shape in the manifest is worth having only if the agent's own
code can reproduce it.

**It reads with the standard library wherever the standard library can.** Not for
elegance — for latency. This runs before the first model call of a turn, so every
millisecond of it is dead air in front of the user's first token, and in a fresh
container an `import` is not a few milliseconds. A snapshot's rootfs is restored lazily,
so the first import of a library pays for fetching it: rdkit 7.4-10.9s, pandas
2.3-11.5s, against 0.99s for a stdlib-only interpreter and ~1s for either of them once
faulted in (see `sandbox.WARMUP`, which starts that faulting in the background so the
*agent's* first `execute` does not pay it either). pandas, rdkit and biopython are
therefore not imported here at all, and what they were doing — dtypes, canonical SMILES,
descriptors — is left to the agent, which has the whole sandbox and a visible progress
line by the time it wants them.

Two libraries stay, because a stdlib substitute would be a worse parser rather than a
cheaper one, and both are cheap: pypdf (1.93s cold) for a PDF's text layer, and PIL
(2.03s) for an image a header parse does not recognise and for a PDF's embedded figures.
Both are reached only by the upload kind that needs them.

Output is one JSON object per line on stdout, one per file, and **nothing else** —
`uploads.py:_parse_lines` keeps the lines that start with `{` and drops the rest, so a
library's stray warning is harmless but a `print()` of anything structured is not.

Two rules govern what a probe may put in a record:

* **Shapes and identifiers, never contents.** Every field here is rendered into the root
  model's system prompt. Row counts, column names, PMIDs and sequence ids earn their
  place because they let the model write code that works first time; a page of extracted
  text would be the exact cost `pmc_locate` exists to avoid.
* **A payload too large for the prompt becomes a file instead.** The citation, chemistry
  and PDF probes write a normalised sidecar under `derived/` and put its path in the
  record. The agent reaches those with `execute` or hands them to a subagent, so the
  bytes reach the JS heap or a leaf's context and never the root transcript.

Nothing here may raise. A file that cannot be parsed gets a `note` saying why, because the
model has to be able to tell the user "your file is a scanned PDF with no text layer"
rather than write code against a shape that does not exist.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = os.environ.get("UPLOADS_ROOT", "/workspace/uploads")
DERIVED = os.environ.get("UPLOADS_DERIVED", ROOT + "/derived")
KINDS = json.loads(os.environ.get("UPLOADS_KINDS") or "{}")
MAX_COLS = int(os.environ.get("UPLOADS_MAX_COLS") or 40)
MAX_ITEMS = int(os.environ.get("UPLOADS_MAX_ITEMS") or 5)
MAX_IDS = int(os.environ.get("UPLOADS_MAX_IDS") or 100)
MAX_FIGURES = int(os.environ.get("UPLOADS_MAX_FIGURES") or 12)

# A PDF's image XObjects are mostly furniture: publisher logos, rules, icons, the tiles a
# vector plot was rasterised into. Area is the cheapest filter that keeps them out, and a
# panel worth sending to `figure-analyst` is never smaller than this.
MIN_FIGURE_PIXELS = 200 * 200

DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PMID_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,8})")
BARE_PMID = re.compile(r"^\s*(?:PMID:?\s*)?(\d{6,8})\s*$", re.IGNORECASE)


# -- shared helpers ------------------------------------------------------------------


def base_suffix(name: str) -> str:
    """The suffix that decides how to read a file, seeing through `.gz`."""
    lowered = name.lower()
    if lowered.endswith(".gz"):
        lowered = lowered[: -len(".gz")]
    _, _, suffix = lowered.rpartition(".")
    return "." + suffix if suffix else ""


def open_text(path: str):
    """A text handle, transparently decompressing `.gz`.

    `errors="replace"` throughout: these are files from a user's laptop, and a stray
    Latin-1 byte in an author name must not cost them the whole upload.
    """
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def write_sidecar(name: str, payload) -> str:
    """Persist a parsed payload beside the upload and return its path.

    Under `derived/` rather than next to the original: `uploads.py` inventories the
    upload directory by name to decide what still needs staging, and a subdirectory is
    skipped by its `isfile` check. It also keeps a sidecar from being probed as if the
    user had attached it.
    """
    os.makedirs(DERIVED, exist_ok=True)
    path = os.path.join(DERIVED, name)
    with open(path, "w", encoding="utf-8") as handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle)
    return path


def write_sidecar_bytes(name: str, payload: bytes) -> str:
    """`write_sidecar` for something that is not text. Same directory, same reasons."""
    os.makedirs(DERIVED, exist_ok=True)
    path = os.path.join(DERIVED, name)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def cap_ids(values: list[str], record: dict, key: str) -> None:
    """Put a bounded id list in the record and say how many were left out."""
    unique = list(dict.fromkeys(v for v in values if v))
    record[key] = unique[:MAX_IDS]
    if len(unique) > MAX_IDS:
        record["more_" + key] = len(unique) - MAX_IDS


# -- tabular -------------------------------------------------------------------------


def set_columns(record: dict, columns: list[str]) -> None:
    """Put a bounded column list in the record and say how many were left out."""
    record["columns"] = columns[:MAX_COLS]
    if len(columns) > MAX_COLS:
        record["more_columns"] = len(columns) - MAX_COLS


def delimiter_for(sample: str, suffix: str) -> str:
    """The separator, named by the suffix where the suffix means it.

    Only `.txt` genuinely needs sniffing — a tool wrote it with whatever it liked — and
    `csv.Sniffer` is what pandas' `sep=None` used to do here. Its failure mode is an
    exception on a file with no consistent delimiter, which a one-column file is, so the
    fallback counts the two candidates instead of guessing.
    """
    known = {".csv": ",", ".tsv": "\t"}.get(suffix)
    if known:
        return known
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def probe_delimited(path: str, suffix: str, record: dict) -> None:
    """Rows and column names out of a CSV/TSV, counted with `csv` rather than pandas.

    `csv.reader` rather than counting newlines because a quoted field may contain one,
    and a row count that disagrees with what the agent's own `read_csv` reports is worse
    than no row count. The header row is excluded from the count, as `DataFrame.shape`
    did.
    """
    with open_text(path) as handle:
        sample = handle.read(64 * 1024)
    if not sample.strip():
        record["rows"] = 0
        record["note"] = "the file is empty"
        return

    delimiter = delimiter_for(sample, suffix)
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        record["rows"] = sum(1 for _ in reader)
    set_columns(record, [str(column).strip() for column in header])


_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_PACKAGE_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def sheet_text(cell, shared: list[str]) -> str:
    """One cell's displayed text. Shared, inline or literal — the three a header uses."""
    if cell.get("t") == "s":
        value = cell.findtext(_SPREADSHEET_NS + "v") or ""
        index = int(value) if value.isdigit() else -1
        return shared[index] if 0 <= index < len(shared) else ""
    if cell.get("t") == "inlineStr":
        inline = cell.find(_SPREADSHEET_NS + "is")
        return "".join(inline.itertext()).strip() if inline is not None else ""
    return (cell.findtext(_SPREADSHEET_NS + "v") or "").strip()


def probe_workbook(path: str, record: dict) -> None:
    """Sheet names, and the first sheet's shape and header, read out of the zip.

    An `.xlsx` is a zip of XML, so this needs no reader beyond `zipfile` — which matters
    because the alternative was pandas, and pandas is 2-11s of cold import for a row
    count. Only the first sheet is described, as before: `sheet_name=None` loaded every
    sheet in the workbook to report the shape of one.
    """
    with zipfile.ZipFile(path) as book:
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        sheets = list(workbook.iter(_SPREADSHEET_NS + "sheet"))
        record["sheets"] = [sheet.get("name") or "" for sheet in sheets]
        if not sheets:
            record["note"] = "the workbook has no sheets"
            return

        # The sheet element names its part by relationship id, not by path. sheet1.xml is
        # the conventional name but nothing requires it, and a workbook whose first sheet
        # was deleted and re-added has them out of order.
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {node.get("Id"): node.get("Target") or "" for node in relationships}
        target = targets.get(sheets[0].get(_PACKAGE_NS + "id") or "", "worksheets/sheet1.xml")
        part = "xl/" + target.lstrip("/").removeprefix("xl/")

        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            table = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in table.iter(_SPREADSHEET_NS + "si")]

        header: list[str] = []
        rows = 0
        with book.open(part) as sheet:
            # Streamed: a sheet is the one part of a workbook that can be tens of MB, and
            # all this needs from it is the first row and how many there are.
            for _, element in ET.iterparse(sheet, events=("end",)):
                if element.tag == _SPREADSHEET_NS + "row":
                    rows += 1
                    if rows == 1:
                        header = [
                            sheet_text(cell, shared)
                            for cell in element.iter(_SPREADSHEET_NS + "c")
                        ]
                    element.clear()

    record["rows"] = max(0, rows - 1)
    set_columns(record, [column.strip() for column in header])


def probe_tabular(path: str, suffix: str, record: dict) -> None:
    """Rows, columns and sheet names. Dispatch only — the two formats share nothing."""
    if suffix in (".xlsx", ".xlsm"):
        probe_workbook(path, record)
    else:
        probe_delimited(path, suffix, record)


# -- citations -----------------------------------------------------------------------


_MEDLINE_TAG = re.compile(r"^([A-Z][A-Z0-9]{1,3})\s*- ?(.*)$")


def medline_entries(text: str) -> list[dict[str, list[str]]]:
    """Split Medline text into tag -> values. Every tag repeats, so every value is a list.

    Hand-parsed rather than with `Bio.Medline`, which costs 2.4s of cold import in a
    fresh container for a grammar that is one tag per line: `TAG- value`, continuation
    lines indented, a blank line between records.
    """
    entries: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last: str | None = None

    for line in text.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current, last = {}, None
            continue
        if found := _MEDLINE_TAG.match(line):
            last = found.group(1)
            current.setdefault(last, []).append(found.group(2).strip())
        elif last and current.get(last):
            # A wrapped title or abstract. Joined with a space, which is what the line
            # break stood for.
            current[last][-1] = f"{current[last][-1]} {line.strip()}".strip()

    if current:
        entries.append(current)
    return entries


def parse_medline(text: str) -> list[dict]:
    """`.nbib` — PubMed's own export, which is Medline with a different extension."""
    references = []
    for entry in medline_entries(text):
        doi = ""
        for candidate in [*entry.get("AID", []), *entry.get("LID", [])]:
            if "[doi]" in candidate.lower() and (found := DOI.search(candidate)):
                doi = found.group(0)
                break
        references.append(
            {
                "pmid": next(iter(entry.get("PMID", [])), ""),
                "doi": doi,
                "title": next(iter(entry.get("TI", [])), ""),
                "journal": next(iter(entry.get("TA", []) or entry.get("JT", [])), ""),
                "year": next(iter(entry.get("DP", [])), "")[:4],
                "authors": entry.get("AU", [])[:3],
            }
        )
    return references


_RIS_TAG = re.compile(r"^([A-Z][A-Z0-9])  - ?(.*)$")


def parse_ris(text: str) -> list[dict]:
    """`.ris` — the interchange format Zotero, EndNote and Mendeley all export.

    Hand-parsed rather than with a library: the grammar is one tag per line and the
    snapshot carries no RIS reader, so a dependency would have to be added to every
    clone's sandbox image to save twenty lines.
    """
    references: list[dict] = []
    current: dict[str, list[str]] = {}
    last: str | None = None

    def flush() -> None:
        if not current:
            return
        pmid = ""
        for value in current.get("AN", []) + current.get("ID", []) + current.get("UR", []):
            if found := PMID_URL.search(value):
                pmid = found.group(1)
                break
            if found := BARE_PMID.match(value):
                pmid = found.group(1)
                break
        doi = ""
        for value in current.get("DO", []) + current.get("UR", []) + current.get("N1", []):
            if found := DOI.search(value):
                doi = found.group(0).rstrip(".")
                break
        title = (current.get("TI") or current.get("T1") or [""])[0]
        journal = (current.get("JO") or current.get("JF") or current.get("T2") or [""])[0]
        year = (current.get("PY") or current.get("Y1") or [""])[0][:4]
        references.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "journal": journal,
                "year": year,
                "authors": current.get("AU", [])[:3],
            }
        )
        current.clear()

    for line in text.splitlines():
        match = _RIS_TAG.match(line)
        if match:
            tag, value = match.group(1), match.group(2).strip()
            if tag == "ER":
                flush()
                last = None
                continue
            current.setdefault(tag, []).append(value)
            last = tag
        elif line.strip() and last and current.get(last):
            # A wrapped value. RIS has no continuation marker, so anything that is not a
            # tag line belongs to the tag above it.
            current[last][-1] += " " + line.strip()
    flush()
    return [r for r in references if any(r.values())]


_BIB_ENTRY = re.compile(r"@\s*[A-Za-z]+\s*\{", re.MULTILINE)
_BIB_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", re.MULTILINE)


def _bib_bodies(text: str) -> list[str]:
    """Each entry's `{...}` body, brace-matched rather than split on a delimiter.

    A regex cannot do this: a BibTeX title routinely contains braces (`{DNA}`), so any
    split on `}` or on `@` inside a field value cuts an entry in half.
    """
    bodies = []
    for match in _BIB_ENTRY.finditer(text):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        bodies.append(text[match.end() : index - 1])
    return bodies


def _bib_value(body: str, start: int) -> str:
    """One field value: `{braced}`, "quoted", or bare up to the next comma."""
    while start < len(body) and body[start].isspace():
        start += 1
    if start >= len(body):
        return ""
    if body[start] == '"':
        end = body.find('"', start + 1)
        return body[start + 1 : end if end != -1 else len(body)].strip()
    if body[start] == "{":
        # Brace-matched, because a BibTeX value routinely contains braces of its own
        # (`{DNA} repair`). `index` lands one past the closing brace, so the slice has to
        # stop short of it — otherwise every DOI comes out with a `}` on the end, which
        # then fails to match anything in PubMed.
        depth = 1
        index = start + 1
        while index < len(body) and depth:
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
            index += 1
        return body[start + 1 : index - 1 if depth == 0 else index].strip()
    end = body.find(",", start)
    return body[start : end if end != -1 else len(body)].strip()


def parse_bibtex(text: str) -> list[dict]:
    """`.bib` — the LaTeX bibliography, which is what a manuscript's refs arrive as."""
    references = []
    for body in _bib_bodies(text):
        fields: dict[str, str] = {}
        for match in _BIB_FIELD.finditer(body):
            fields[match.group(1).lower()] = _bib_value(body, match.end())
        if not fields:
            continue
        blob = " ".join(fields.values())
        pmid = fields.get("pmid", "").strip()
        if not pmid and (found := PMID_URL.search(blob)):
            pmid = found.group(1)
        doi = fields.get("doi", "").strip()
        if not doi and (found := DOI.search(blob)):
            doi = found.group(0).rstrip(".")
        references.append(
            {
                "pmid": re.sub(r"\D", "", pmid),
                "doi": doi,
                "title": re.sub(r"[{}]", "", fields.get("title", "")),
                "journal": re.sub(
                    r"[{}]", "", fields.get("journal") or fields.get("booktitle", "")
                ),
                "year": fields.get("year", "")[:4],
                "authors": [a.strip() for a in fields.get("author", "").split(" and ")[:3] if a],
            }
        )
    return references


def parse_id_list(text: str) -> list[dict]:
    """A bare list of PMIDs or DOIs — a `.txt` saved out of PubMed's clipboard."""
    references = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if (found := BARE_PMID.match(line)) or (found := PMID_URL.search(line)):
            references.append({"pmid": found.group(1), "doi": "", "title": ""})
        elif found := DOI.search(line):
            references.append({"pmid": "", "doi": found.group(0).rstrip("."), "title": ""})
    return references


def looks_like_id_list(text: str) -> bool:
    """Whether a `.txt` is a citation list rather than a delimited table.

    Checked before the tabular reader, which will happily read a column of PMIDs as a
    one-column table and leave the manifest describing a spreadsheet.
    """
    lines = [line.strip() for line in text.splitlines()[:200] if line.strip()]
    if not lines:
        return False
    hits = sum(1 for line in lines if BARE_PMID.match(line) or DOI.fullmatch(line))
    return hits >= max(3, int(0.8 * len(lines)))


def probe_citations(path: str, suffix: str, record: dict, name: str) -> None:
    """Normalise a bibliography into records the agent can hydrate with `fetchAbstracts`.

    This is the one upload kind whose useful materialisation is not the file. What the
    agent wants is the PMID list, so the manifest carries it directly when it is small
    enough and points at the sidecar when it is not.
    """
    with open_text(path) as handle:
        text = handle.read()

    parser = {
        ".nbib": parse_medline,
        ".medline": parse_medline,
        ".ris": parse_ris,
        ".bib": parse_bibtex,
        ".bibtex": parse_bibtex,
    }.get(suffix, parse_id_list)
    references = parser(text)

    # A `.nbib` that is really RIS, or a `.bib` a tool wrote without `@`. Falling back
    # costs one reparse and turns "0 references" into a working corpus.
    if not references and suffix != ".txt":
        for fallback in (parse_ris, parse_medline, parse_bibtex, parse_id_list):
            try:
                references = fallback(text)
            except Exception:  # noqa: BLE001 - a wrong-format guess, not a failure
                continue
            if references:
                break

    record["references"] = len(references)
    if not references:
        record["note"] = "no references could be parsed out of this file"
        return

    pmids = [r["pmid"] for r in references if r.get("pmid")]
    record["with_pmid"] = len(pmids)
    record["with_doi_only"] = sum(1 for r in references if r.get("doi") and not r.get("pmid"))
    record["refs_path"] = write_sidecar(name + ".refs.json", references)
    cap_ids(pmids, record, "pmids")


# -- pdf -----------------------------------------------------------------------------


def extract_figures(reader, name: str, record: dict) -> None:
    """Write the embedded images out as files, so a figure is reachable at all.

    `figure-analyst` reads an image from a path — that is the whole reason `fetch_figures`
    materialises PMC figures instead of returning them. An uploaded PDF had no equivalent,
    so a question whose answer sits in a plot or a gel was unanswerable from the text
    sidecar alone. This closes that, and it is also what makes a scanned PDF readable:
    with no text layer, the page images *are* the document.

    Paths only in the record, never the bytes — the rules at the top of this file apply
    here exactly as they do to a PDF's text.
    """
    from PIL import Image

    paths: list[str] = []
    skipped = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            embedded = list(page.images)
        except Exception:  # noqa: BLE001 - an exotic filter on one page, not a failure
            continue
        for index, item in enumerate(embedded, start=1):
            if len(paths) >= MAX_FIGURES:
                skipped += 1
                continue
            try:
                with Image.open(io.BytesIO(item.data)) as image:
                    width, height = image.size
                    if width * height < MIN_FIGURE_PIXELS:
                        continue
                    buffer = io.BytesIO()
                    # Re-encoded rather than written through: an embedded image can be
                    # CMYK JPEG, 1-bit CCITT or JPEG2000, and `figure-analyst` needs
                    # something its reader will open.
                    image.convert("RGB").save(buffer, format="PNG")
                paths.append(
                    write_sidecar_bytes(f"{name}.p{number}.{index}.png", buffer.getvalue())
                )
            except Exception:  # noqa: BLE001 - one unreadable image, not a failure
                continue

    if paths:
        record["figures"] = len(paths)
        record["figure_paths"] = paths
    if skipped:
        record["more_figures"] = skipped


def probe_pdf(path: str, record: dict, name: str) -> None:
    """Extract the text layer to a sidecar. The text itself never comes back from here.

    A median paper is ~40k characters, which is precisely the payload the architecture
    keeps out of the root transcript. So the record carries the path and the size, and
    the prompt sends `document-analyst` to read it.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        # An owner-password PDF (print/copy restricted, no user password) opens with an
        # empty password, which is the common case for a publisher download.
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001 - genuinely locked
            record["note"] = "password-protected; no text could be extracted"
            return

    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(pages)
    record["pages"] = len(pages)
    record["chars"] = len(text)

    metadata = reader.metadata or {}
    if title := str(metadata.get("/Title") or "").strip():
        record["title"] = title[:200]

    # Before the scanned-PDF return below, because that is the case where the images are
    # not illustration but the entire document.
    try:
        extract_figures(reader, name, record)
    except Exception as exc:  # noqa: BLE001 - the text is the greater half; keep it
        record["figures_note"] = f"no figures could be extracted: {type(exc).__name__}"

    if len(text.strip()) < 200 * max(1, len(pages)) // 100:
        record["note"] = (
            "almost no extractable text — this is probably a scanned PDF, and there is "
            "no OCR in this sandbox"
        )
        if record.get("figure_paths"):
            record["note"] += "; the page images below are the only readable form of it"
        if not text.strip():
            return

    record["text_path"] = write_sidecar(name + ".txt", text)
    record["lines"] = text.count("\n") + 1

    # A DOI or PMID on the first pages is what links the user's own copy back to PubMed,
    # which is where everything else this agent does begins.
    head = "\n".join(pages[:2])
    if found := DOI.search(head):
        record["doi"] = found.group(0).rstrip(".")
    if found := PMID_URL.search(head) or re.search(r"PMID:?\s*(\d{6,8})", head):
        record["pmid"] = found.group(1)


# -- chemistry -----------------------------------------------------------------------


_SDF_PROPERTY = re.compile(r"^>\s*<(.+?)>")


def molfile_atoms(lines: list[str]) -> int | None:
    """Atom count off a molfile's counts line, which is line 4 of every record.

    V2000 fixed-width (`aaabbb`), or the `M  V30 COUNTS` line a V3000 record uses
    instead. `None` when the file is not a molfile at all, which is what the caller
    reports rather than guessing at.
    """
    if len(lines) < 4:
        return None
    counts = lines[3]
    if "V3000" in counts.upper():
        for line in lines[4:16]:
            if "COUNTS" in line.upper():
                fields = line.split()
                if len(fields) >= 4 and fields[3].isdigit():
                    return int(fields[3])
        return None
    try:
        return int(counts[0:3])
    except ValueError:
        return None


def sdf_records(path: str):
    """Yield each record of an SDF (or the single record of a `.mol`) as its lines."""
    block: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("$$$$"):
                yield block
                block = []
            else:
                block.append(line.rstrip("\n"))
    if any(line.strip() for line in block):
        yield block


def probe_chem(path: str, suffix: str, record: dict, name: str) -> None:
    """Counts, titles and the SDF property columns. Structure is left to rdkit.

    What used to be here — canonical SMILES, formula, molecular weight — cost 7-11s of
    cold rdkit import in front of the user's first token, every time, to describe a file
    the agent can open itself in a fraction of that once the container is warm. So the
    manifest now carries what the file *says* (how many records, what they are called,
    which property columns they have) and the prompt sends the agent to rdkit for what
    the file only *implies*. A `.smi` is the exception and needs no library at all: its
    SMILES are the file.
    """
    if suffix in (".smi", ".smiles"):
        rows = []
        with open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split(None, 1)
                rows.append(
                    {
                        "name": (fields[1].strip() if len(fields) > 1 else "")
                        or f"mol_{len(rows) + 1}",
                        "smiles": fields[0],
                    }
                )
        record["molecules"] = len(rows)
        if not rows:
            record["note"] = "no SMILES lines could be read out of this file"
            return
        record["mols_path"] = write_sidecar(name + ".mols.json", rows)
        record["samples"] = [
            {"name": row["name"], "smiles": row["smiles"][:80]} for row in rows[:MAX_ITEMS]
        ]
        return

    titles: list[dict] = []
    properties: set[str] = set()
    molecules = unparsed = 0
    for block in sdf_records(path):
        atoms = molfile_atoms(block)
        if atoms is None:
            unparsed += 1
            continue
        molecules += 1
        if len(titles) < MAX_ITEMS:
            titles.append({"name": block[0].strip() or f"mol_{molecules}", "atoms": atoms})
        # Capped at the first 50 records, as the rdkit version was: a PubChem download
        # repeats the same 36 property tags in every one of them.
        if molecules <= 50:
            properties.update(
                found.group(1) for line in block if (found := _SDF_PROPERTY.match(line))
            )

    record["molecules"] = molecules
    if unparsed:
        record["unparsed"] = unparsed
    if not molecules:
        record["note"] = "no molecule records could be read out of this file"
        return

    record["samples"] = titles
    if visible := sorted(p for p in properties if not p.startswith("_")):
        record["properties"] = visible[: MAX_ITEMS * 2]
        if len(visible) > MAX_ITEMS * 2:
            record["more_properties"] = len(visible) - MAX_ITEMS * 2


# -- sequences -----------------------------------------------------------------------


def parse_fasta(path: str) -> list[dict]:
    """FASTA by hand: a `>` line is a record, everything to the next one is its residues.

    The format is two rules, so this is cheaper than the 2.4s of cold `Bio.SeqIO` import
    it replaces — and it streams, where `SeqIO.parse` held every residue of every record.
    """
    rows: list[dict] = []
    for line in open_text(path):
        if line.startswith(">"):
            identifier, _, description = line[1:].strip().partition(" ")
            rows.append(
                {
                    "id": identifier,
                    "description": description.strip()[:160],
                    "length": 0,
                    "residues": "",
                }
            )
        elif rows:
            residues = "".join(line.split())
            rows[-1]["length"] += len(residues)
            if len(rows[-1]["residues"]) < 60:
                rows[-1]["residues"] = (rows[-1]["residues"] + residues.upper())[:60]
    return rows


def parse_genbank(path: str) -> list[dict]:
    """GenBank through biopython, which is the one sequence format worth 2.4s for.

    Its grammar is a flat-file record with continuation rules per key, and a hand-parser
    good enough for `DEFINITION` wrapping and the `ORIGIN` block would be a worse reader
    than the library for a format a user attaches far less often than FASTA.
    """
    from Bio import SeqIO

    rows = []
    with open_text(path) as handle:
        for entry in SeqIO.parse(handle, "genbank"):
            description = entry.description
            if description.startswith(entry.id):
                description = description[len(entry.id) :].strip()
            rows.append(
                {
                    "id": entry.id,
                    "description": description[:160],
                    "length": len(entry.seq),
                    "residues": str(entry.seq[:60]).upper(),
                }
            )
    return rows


def probe_sequence(path: str, suffix: str, record: dict, name: str) -> None:
    """Ids, lengths and molecule type. Never the residues — those are what the file is for."""
    genbank = suffix in (".gb", ".gbk", ".genbank")
    fmt = "genbank" if genbank else "fasta"
    rows = parse_genbank(path) if genbank else parse_fasta(path)
    record["sequences"] = len(rows)
    if not rows:
        record["note"] = f"no {fmt} records could be parsed out of this file"
        return

    record["format"] = fmt
    record["total_residues"] = sum(r["length"] for r in rows)
    # Nucleotide vs protein off the alphabet of what was read, because the two lead to
    # completely different literature searches and the file itself never says which.
    alphabet = set("".join(r["residues"] for r in rows[:20]))
    record["molecule"] = "nucleotide" if alphabet <= set("ACGTUNRYKMSWBDHV-") else "protein"
    record["seqs_path"] = write_sidecar(
        name + ".seqs.json", [{k: v for k, v in r.items() if k != "residues"} for r in rows]
    )
    record["samples"] = [
        {"id": r["id"], "length": r["length"], "description": r["description"][:80]}
        for r in rows[:MAX_ITEMS]
    ]


# -- images --------------------------------------------------------------------------


def header_size(path: str) -> tuple[str, int, int] | None:
    """Format and dimensions off an image's header bytes, or `None` if unrecognised.

    The four `UPLOAD_KINDS` image suffixes all put their dimensions in the first few
    dozen bytes, so the common case needs no decoder — which is the point, since the
    decoder is PIL and PIL is 2s of cold import for two integers.
    """
    with open(path, "rb") as handle:
        head = handle.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", head[16:24])
            return "png", width, height
        if head[:6] in (b"GIF87a", b"GIF89a"):
            width, height = struct.unpack("<HH", head[6:10])
            return "gif", width, height
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            if head[12:16] == b"VP8X":
                width = int.from_bytes(head[24:27], "little") + 1
                height = int.from_bytes(head[27:30], "little") + 1
                return "webp", width, height
            if head[12:16] == b"VP8 ":
                width, height = struct.unpack("<HH", head[26:30])
                return "webp", width & 0x3FFF, height & 0x3FFF
            return None  # lossless VP8L, whose header is bit-packed; let PIL read it
        if head[:2] != b"\xff\xd8":
            return None

        # JPEG: walk the segment chain to the frame header, which is the only place the
        # dimensions are. Segment lengths are explicit, so this is a seek per marker.
        handle.seek(2)
        while chunk := handle.read(2):
            if len(chunk) < 2 or chunk[0] != 0xFF:
                return None
            marker, length = chunk[1], int.from_bytes(handle.read(2), "big")
            # SOF0-SOF15, excluding the four markers in that range that are not frames.
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC, 0xD8):
                frame = handle.read(5)
                height, width = struct.unpack(">HH", frame[1:5])
                return "jpeg", width, height
            handle.seek(length - 2, os.SEEK_CUR)
    return None


def probe_image(path: str, record: dict) -> None:
    """Dimensions only. The image is for `figure-analyst`, which reads it from the path."""
    if found := header_size(path):
        record["format"], record["width"], record["height"] = found
        return

    # Something the header parse above does not know: a TIFF a user renamed, a lossless
    # WebP, a truncated file. PIL opens far more than those four formats and this is the
    # only path that pays for it.
    from PIL import Image

    with Image.open(path) as image:
        record["width"], record["height"] = image.size
        record["format"] = (image.format or "").lower()


# -- driver --------------------------------------------------------------------------


PROBES = {
    "tabular": lambda path, suffix, record, name: probe_tabular(path, suffix, record),
    "citations": lambda path, suffix, record, name: probe_citations(path, suffix, record, name),
    "pdf": lambda path, suffix, record, name: probe_pdf(path, record, name),
    "chem": lambda path, suffix, record, name: probe_chem(path, suffix, record, name),
    "sequence": lambda path, suffix, record, name: probe_sequence(path, suffix, record, name),
    "image": lambda path, suffix, record, name: probe_image(path, record),
}


def resolve_kind(path: str, suffix: str) -> str:
    """Which probe to run. Only `.txt` needs to be looked at rather than named."""
    kind = KINDS.get(suffix, "tabular")
    if suffix == ".txt":
        try:
            with open_text(path) as handle:
                if looks_like_id_list(handle.read(65_536)):
                    return "citations"
        except Exception:  # noqa: BLE001 - fall through to the tabular reader
            pass
        return "tabular"
    return kind


def main() -> None:
    names = sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
    for name in names:
        path = os.path.join(ROOT, name)
        if not os.path.isfile(path):
            continue
        suffix = base_suffix(name)
        record = {"name": name, "path": path}
        try:
            record["bytes"] = os.path.getsize(path)
            kind = resolve_kind(path, suffix)
            record["kind"] = kind
            PROBES[kind](path, suffix, record, name)
        except Exception as exc:  # noqa: BLE001 - every failure is a note, never a crash
            record["note"] = f"could not read: {type(exc).__name__}: {exc}"[:200]
        print(json.dumps(record))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
