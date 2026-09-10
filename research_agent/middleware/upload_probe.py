"""Describe the user's uploaded files. Runs **inside the sandbox**, never in this process.

`middleware/uploads.py` reads this file as text and pipes it to the sandbox's `python3`
over a heredoc. That is why it imports nothing from `research_agent` and takes its
configuration from the environment: there is no package here, only an interpreter and the
libraries `scripts/build_snapshot.py` baked in.

It exists because **the reader has to live where the library lives**. pandas, pypdf, rdkit
and biopython are in the sandbox image and deliberately not in the host's dependency set —
this repo's host half talks to NCBI and nothing else. Probing where the file already sits
also means a file this script parses is a file the agent's own code can open, which is the
property worth having: a shape in the manifest that pandas cannot reproduce would be worse
than no shape at all.

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

import gzip
import io
import json
import os
import re
import sys

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


def probe_tabular(path: str, suffix: str, record: dict) -> None:
    """Rows, columns and sheet names, read with the same pandas the agent will use."""
    import pandas as pd

    frame = None
    if suffix in (".xlsx", ".xlsm"):
        sheets = pd.read_excel(path, sheet_name=None)
        record["sheets"] = [str(s) for s in sheets]
        frame = next(iter(sheets.values())) if sheets else None
    else:
        # `sep=None` asks pandas to sniff, which needs the Python engine. Worth it for
        # `.txt`, where the separator is whatever the tool that wrote it happened to use.
        sep = {".tsv": "\t", ".csv": ","}.get(suffix)
        if sep is None:
            frame = pd.read_csv(path, sep=None, engine="python")
        else:
            frame = pd.read_csv(path, sep=sep)
    if frame is None:
        return
    record["rows"] = int(frame.shape[0])
    columns = [str(c) for c in frame.columns]
    record["columns"] = columns[:MAX_COLS]
    if len(columns) > MAX_COLS:
        record["more_columns"] = len(columns) - MAX_COLS


# -- citations -----------------------------------------------------------------------


def parse_medline(text: str) -> list[dict]:
    """`.nbib` — PubMed's own export, which is Medline with a different extension."""
    from Bio import Medline

    references = []
    for entry in Medline.parse(io.StringIO(text)):
        doi = ""
        for candidate in [*(entry.get("AID") or []), entry.get("LID", "")]:
            if "[doi]" in candidate.lower() and (found := DOI.search(candidate)):
                doi = found.group(0)
                break
        references.append(
            {
                "pmid": str(entry.get("PMID") or ""),
                "doi": doi,
                "title": str(entry.get("TI") or ""),
                "journal": str(entry.get("TA") or entry.get("JT") or ""),
                "year": (str(entry.get("DP") or "")[:4]),
                "authors": list(entry.get("AU") or [])[:3],
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

    Checked before the tabular reader, because pandas will happily read a column of
    PMIDs as a one-column frame and the manifest would then describe a spreadsheet.
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


def probe_chem(path: str, suffix: str, record: dict, name: str) -> None:
    """Molecules, canonical SMILES and the two descriptors worth having up front."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    # rdkit narrates every parse failure to stderr. That is noise here — the counts below
    # are the report, and `_parse_lines` would drop the lines anyway.
    RDLogger.DisableLog("rdApp.*")

    molecules = []
    failed = 0
    if suffix == ".mol":
        molecules = [Chem.MolFromMolFile(path)]
    elif suffix == ".sdf":
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, "rb") as handle:
            molecules = list(Chem.ForwardSDMolSupplier(handle))
    else:
        # `.smi`/`.smiles`: SMILES first, optional name second, whitespace-separated.
        # Read by hand rather than with SmilesMolSupplier so the name column survives a
        # file with no header and so a bad line is counted instead of ending the read.
        with open_text(path) as text_handle:
            for line in text_handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                mol = Chem.MolFromSmiles(parts[0])
                if mol is None:
                    failed += 1
                    continue
                if len(parts) > 1:
                    mol.SetProp("_Name", parts[1].strip())
                molecules.append(mol)

    parsed = [m for m in molecules if m is not None]
    failed += len(molecules) - len(parsed)
    record["molecules"] = len(parsed)
    if failed:
        record["unparsed"] = failed
    if not parsed:
        record["note"] = "no molecules could be parsed out of this file"
        return

    rows = []
    for index, mol in enumerate(parsed):
        title = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
        rows.append(
            {
                "name": title or f"mol_{index + 1}",
                "smiles": Chem.MolToSmiles(mol),
                "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "mw": round(Descriptors.MolWt(mol), 2),
            }
        )
    record["mols_path"] = write_sidecar(name + ".mols.json", rows)
    # Formula and weight rather than SMILES: a sample line here is prompt text, and one
    # SMILES for a kinase inhibitor is 80 characters that say nothing the sidecar does not.
    record["samples"] = [
        {"name": r["name"], "formula": r["formula"], "mw": r["mw"]} for r in rows[:MAX_ITEMS]
    ]

    # SDF property columns are where an assay result lives, and they are the reason to
    # look at the file with pandas rather than only through rdkit. Capped hard: a PubChem
    # download carries 36 of them, all provenance and none of them a measurement.
    if properties := sorted({key for m in parsed[:50] for key in m.GetPropNames()}):
        visible = [p for p in properties if not p.startswith("_")]
        record["properties"] = visible[:MAX_ITEMS * 2]
        if len(visible) > MAX_ITEMS * 2:
            record["more_properties"] = len(visible) - MAX_ITEMS * 2


# -- sequences -----------------------------------------------------------------------


def probe_sequence(path: str, suffix: str, record: dict, name: str) -> None:
    """Ids, lengths and molecule type. Never the residues — those are what the file is for."""
    from Bio import SeqIO

    fmt = "genbank" if suffix in (".gb", ".gbk", ".genbank") else "fasta"
    rows = []
    with open_text(path) as handle:
        for entry in SeqIO.parse(handle, fmt):
            # biopython repeats the id at the head of `description` for FASTA, because
            # that is literally the same `>` line. Dropping it keeps the sample lines
            # from saying the accession twice.
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


def probe_image(path: str, record: dict) -> None:
    """Dimensions only. The image is for `figure-analyst`, which reads it from the path."""
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
