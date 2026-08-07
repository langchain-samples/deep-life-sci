"""PubMed E-utilities client and the two agent tools.

The defensive guards in here are not generic paranoia — each one corresponds to a
verified failure mode of the live API. See `pubmed_api_notes/` for the probe results.
The three that produce *wrong answers rather than errors*:

1. efetch tokenizes malformed PMIDs and returns unrelated papers ('42.9' -> PMIDs 42
   and 9), so ids are validated against ^\\d+$ before any request goes out.
2. esearch silently rewrites broken queries ('cancer[nosuchfield]' -> 5.7M hits), so
   `query_translation` and `warnings` are always returned to the caller.
3. esummary's 500-UID cap returns HTTP 200 with no `result` key, so the error key is
   checked before the payload is read.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import tool

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

DATA_DIR = Path(__file__).parent / "data"
ABSTRACT_CACHE = DATA_DIR / "abstracts"
SEARCH_DUMPS = DATA_DIR / "searches"

# GET breaks at ~361 ids (HTTP 414); POST above this and there is no measured downside.
POST_THRESHOLD = 200
# esummary rejects >500 UIDs with an HTTP 200 error body. Chunk well under it.
SUMMARY_CHUNK = 200
FETCH_CHUNK = 200
# esearch silently clamps retmax to this; clamp explicitly so the caller knows.
MAX_RETMAX = 9999
# Above this many hits, dump the full record list to disk instead of returning it all.
DUMP_THRESHOLD = 50

PMID_RE = re.compile(r"^\d+$")
FIELD_TAG_RE = re.compile(r"\[([^\[\]]+)\]")

# PubMed's documented search field tags, plus the canonical names einfo reports and the
# long forms PubMed also accepts. An unrecognised tag is NOT rejected by the API — it is
# silently dropped and the search runs across all fields, so 'cancer[nosuchfield]'
# returns 5.7 million hits and reports no error anywhere in the response. Checking
# locally is the only way to catch it, and catching it before the request is better than
# detecting it after. A tag missing from this set produces a warning, not a failure, so
# a newly-added PubMed tag degrades to noise rather than a broken search.
FIELD_TAGS = frozenset(
    """
    1au ad affl aid all au auid book cdat cn cntys cois coln crdt dcom dp ecno ed edat
    epdt fau filt fir full grnt gr invr ip isbn iss issn jid jour la lang lastau lid lr
    majr mesh mh mhda nm ot otitle pa page papx pdat pg pid pl pmc pmcid pmid ppdt ps pt
    ptyp pubn rn sb sh si so subh subs ta ti tiab titl tt uid vi vol word
    all fields|article identifier|affiliation|author|author identifier|book
    completion date|conflict of interest statement|corporate author|create date
    date - publication|ec/rn number|editor|entry date|filter|first author name
    full author name|full investigator name|grants and funding|investigator|isbn|issue
    journal|language|last author name|location id|mesh date|mesh major topic
    mesh subheadings|mesh terms|modification date|nlm unique id|other term|pagination
    personal name as subject|pharmacological action|place of publication|pmid
    publication date|publication type|publisher|secondary source id|subset
    supplementary concept|text words|title|title/abstract|transliterated title|volume
    """.replace("|", " ").split()
) | {
    # multi-word long forms, which the whitespace split above would break apart
    "all fields", "article identifier", "author identifier", "completion date",
    "conflict of interest statement", "corporate author", "create date",
    "date - publication", "ec/rn number", "entry date", "first author name",
    "full author name", "full investigator name", "grants and funding",
    "last author name", "location id", "mesh date", "mesh major topic",
    "mesh subheadings", "mesh terms", "modification date", "nlm unique id",
    "other term", "personal name as subject", "pharmacological action",
    "place of publication", "publication date", "publication type",
    "secondary source id", "supplementary concept", "text words", "title/abstract",
    "transliterated title",
}


def check_field_tags(term: str) -> list[str]:
    """Warn about field tags PubMed will silently ignore.

    Date ranges like '2024:2025[dp]' and the tags inside them are handled by the same
    bracket match, so only the tag name is checked.
    """
    warnings = []
    for raw in FIELD_TAG_RE.findall(term):
        tag = raw.strip().lower()
        if tag and tag not in FIELD_TAGS:
            warnings.append(
                f"'[{raw}]' is not a recognised PubMed field tag. PubMed does not "
                f"reject unknown tags — it drops them and searches every field "
                f"instead, so these results are almost certainly far broader than "
                f"intended. Fix the tag and search again."
            )
    return warnings


class PubMedError(RuntimeError):
    """An E-utilities call failed in a way the caller needs to know about."""


# --------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------

def _common_params() -> dict[str, str]:
    params = {
        "tool": os.environ.get("NCBI_TOOL", "deepagents_demo"),
        "email": os.environ.get("NCBI_EMAIL", ""),
    }
    if key := os.environ.get("NCBI_API_KEY"):
        params["api_key"] = key
    return {k: v for k, v in params.items() if v}


def _min_interval() -> float:
    """3 req/sec without an API key, 10 with one. Leave a little headroom."""
    return 0.11 if os.environ.get("NCBI_API_KEY") else 0.34


_rate_lock = asyncio.Lock()
_last_call = 0.0


async def _throttle() -> None:
    global _last_call
    async with _rate_lock:
        wait = _min_interval() - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


async def _request(util: str, **params: Any) -> httpx.Response:
    """Call an E-utility, retrying once on 429/5xx. Raises on non-200."""
    payload = {**_common_params(), **{k: v for k, v in params.items() if v is not None}}
    url = f"{BASE_URL}{util}.fcgi"
    # Long id lists must go in the body — GET dies at ~3.3k chars of URL with a 414
    # whose body is not parseable XML/JSON.
    use_post = len(str(payload.get("id", ""))) > POST_THRESHOLD * 9

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in (1, 2):
            await _throttle()
            if use_post:
                resp = await client.post(url, data=payload)
            else:
                resp = await client.get(url, params=payload)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt == 1:
                await asyncio.sleep(1.5)
                continue
            raise PubMedError(
                f"{util} returned HTTP {resp.status_code}"
                + (" (id list too long for GET)" if resp.status_code == 414 else "")
            )
    raise PubMedError(f"{util} failed after retry")


def validate_pmids(pmids: list[str]) -> tuple[list[str], list[str]]:
    """Split into (valid, invalid). Never send an unvalidated id to efetch."""
    valid, invalid = [], []
    for p in pmids:
        s = str(p).strip()
        (valid if PMID_RE.match(s) else invalid).append(s)
    # preserve order, drop duplicates (efetch dedupes server-side anyway)
    return list(dict.fromkeys(valid)), invalid


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------------
# esearch + esummary
# --------------------------------------------------------------------------------

def _collect_warnings(result: dict) -> list[str]:
    """Everything PubMed did to the query that the caller did not ask for.

    This is the only signal that a query was silently rewritten — a bad field tag
    returns millions of hits with an empty `fieldsnotfound`.
    """
    warnings: list[str] = []
    errorlist = result.get("errorlist") or {}
    warninglist = result.get("warninglist") or {}

    for phrase in errorlist.get("phrasesnotfound", []):
        warnings.append(f"phrase not found in index: {phrase!r}")
    for field in errorlist.get("fieldsnotfound", []):
        warnings.append(f"field tag not recognised: {field!r}")
    for phrase in warninglist.get("quotedphrasesnotfound", []):
        warnings.append(
            f"quoted phrase {phrase!r} not found — PubMed fell back to an unquoted "
            "search, so these results may be unrelated to the intended phrase"
        )
    for phrase in warninglist.get("phrasesignored", []):
        warnings.append(f"phrase ignored: {phrase!r}")
    for msg in warninglist.get("outputmessages", []):
        if msg != "No items found.":
            warnings.append(f"PubMed modified the query: {msg!r}")
    if result.get("ERROR"):
        warnings.append(f"esearch error: {result['ERROR']!r}")
    return warnings


async def _esummary(pmids: list[str]) -> dict[str, dict]:
    """Fetch citation metadata, chunked under the 500-UID cap."""
    records: dict[str, dict] = {}
    for chunk in _chunks(pmids, SUMMARY_CHUNK):
        resp = await _request("esummary", db="pubmed", id=",".join(chunk), retmode="json")
        payload = resp.json()
        # Over-cap and malformed requests come back as HTTP 200 with an `error` key
        # and no `result` at all — reading result.uids first would look like 0 hits.
        if "error" in payload:
            raise PubMedError(f"esummary: {payload['error']}")
        result = payload.get("result")
        if result is None:
            raise PubMedError("esummary returned no `result` block")
        for uid in result.get("uids", []):
            rec = result.get(uid)
            if isinstance(rec, dict) and not rec.get("error"):
                records[uid] = rec
    return records


def _summary_to_record(uid: str, rec: dict) -> dict:
    articleids = {a.get("idtype"): a.get("value") for a in rec.get("articleids", [])}
    authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
    return {
        "pmid": uid,
        "title": rec.get("title") or None,
        "first_author": authors[0] if authors else None,
        # The senior author / PI — usually who a biologist means by "lead author".
        "last_author": rec.get("lastauthor") or None,
        # pubdate has 30+ formats ('2024', '2025 Oct-Dec', '2026 Jan-Jun'); only the
        # leading year is reliable, so don't pretend to parse a full date.
        "year": (rec.get("pubdate") or "")[:4] or None,
        # Book records (StatPearls, GeneReviews) leave `source` empty.
        "journal": rec.get("source") or rec.get("booktitle") or None,
        "doi": articleids.get("doi"),
    }


@tool
async def pubmed_search(
    term: str,
    retmax: int = 50,
    sort: str = "relevance",
    mindate: str | None = None,
    maxdate: str | None = None,
) -> dict:
    """Search PubMed with a boolean query and return matching papers with metadata.

    Supports full PubMed boolean syntax: AND/OR/NOT, field tags like [ti], [au],
    [mesh], [pt], and quoted phrases. Example:
    '(CRISPR OR "base editing") AND liver AND 2023:2025[dp] NOT review[pt]'

    ALWAYS read `query_translation` and `warnings` in the response before trusting the
    results. PubMed silently repairs malformed queries rather than rejecting them — an
    unrecognised field tag is dropped and the search runs across all fields, which can
    return millions of irrelevant hits that look like a successful search.

    Args:
        term: The boolean query.
        retmax: Max papers to return (capped at 9999).
        sort: 'relevance', 'pub_date', or 'Author'. Invalid values are ignored by
            PubMed without error, so stick to these three.
        mindate: Earliest publication date, 'YYYY' or 'YYYY/MM/DD'.
        maxdate: Latest publication date, same format.

    Returns:
        count: total matches in PubMed (may be far larger than the records returned)
        returned: how many records are in this response
        query_translation: what PubMed ACTUALLY searched, with MeSH expansion
        warnings: list of ways PubMed altered the query; empty means it ran as written
        records: list of {pmid, title, first_author, last_author, year, journal, doi}
        saved_to: path to a JSON dump when the result set is large, else None
    """
    retmax = max(1, min(int(retmax), MAX_RETMAX))
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": sort,
    }
    if mindate or maxdate:
        params.update(datetype="pdat", mindate=mindate, maxdate=maxdate)

    resp = await _request("esearch", **params)
    result = resp.json().get("esearchresult", {})
    # Local tag check first: an unknown field tag leaves no trace anywhere in the
    # response, so _collect_warnings alone cannot see it.
    warnings = check_field_tags(term) + _collect_warnings(result)
    pmids = result.get("idlist", [])

    records = []
    if pmids:
        summaries = await _esummary(pmids)
        # Keep esearch's ordering (relevance/date), which the summary dict loses.
        records = [
            _summary_to_record(p, summaries[p]) for p in pmids if p in summaries
        ]

    saved_to = None
    if len(records) > DUMP_THRESHOLD:
        SEARCH_DUMPS.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")[:60] or "search"
        path = SEARCH_DUMPS / f"{slug}.json"
        path.write_text(json.dumps(records, indent=2))
        # Agent-visible path: the filesystem backend is rooted at data/.
        saved_to = f"/searches/{path.name}"

    return {
        "count": int(result.get("count", 0) or 0),
        "returned": len(records),
        "query_translation": result.get("querytranslation") or "",
        "warnings": warnings,
        "records": records[:DUMP_THRESHOLD] if saved_to else records,
        "saved_to": saved_to,
    }


# --------------------------------------------------------------------------------
# efetch
# --------------------------------------------------------------------------------

def _text_of(node: ET.Element) -> str:
    """Full text of a node including inline markup.

    Abstracts contain <b>/<i>/<sub>/<sup>; node.text stops at the first child and
    silently drops everything after it.
    """
    return "".join(node.itertext()).strip()


def _parse_article(art: ET.Element) -> dict | None:
    pmid = art.findtext(".//PMID")
    if not pmid:
        return None

    sections = []
    for node in art.findall(".//Abstract/AbstractText"):
        text = _text_of(node)
        if text:
            sections.append({"label": node.get("Label") or node.get("NlmCategory"), "text": text})

    if any(s["label"] for s in sections):
        abstract = "\n\n".join(
            f"{s['label']}: {s['text']}" if s["label"] else s["text"] for s in sections
        )
    else:
        abstract = "\n\n".join(s["text"] for s in sections)

    pubtypes = [p.text for p in art.findall(".//PublicationType") if p.text]
    reftypes = {c.get("RefType") for c in art.findall(".//CommentsCorrections")}
    retracted = "Retracted Publication" in pubtypes or "RetractionIn" in reftypes

    year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""

    return {
        "pmid": pmid,
        "title": art.findtext(".//ArticleTitle") or art.findtext(".//BookTitle"),
        "abstract": abstract or None,
        "sections": sections,
        "journal": art.findtext(".//Journal/Title") or art.findtext(".//BookTitle"),
        "year": year[:4] or None,
        "retracted": retracted,
        "publication_types": pubtypes,
    }


def _parse_efetch(xml_text: str) -> dict[str, dict]:
    root = ET.fromstring(xml_text)
    records = {}
    # Book records (StatPearls, GeneReviews) are PubmedBookArticle, not PubmedArticle,
    # and they do have abstracts — querying only the latter silently drops them.
    for tag in ("PubmedArticle", "PubmedBookArticle"):
        for art in root.findall(f".//{tag}"):
            rec = _parse_article(art)
            if rec:
                records[rec["pmid"]] = rec
    return records


@tool
async def fetch_abstracts(pmids: list[str]) -> dict:
    """Retrieve abstracts for a list of PMIDs, batched and cached locally.

    Pass every PMID you need in one call — batching is what keeps the whole workflow
    inside NCBI's rate limit. Do NOT call this once per paper in a loop, and do not
    have subagents call it; fetch here, then pass the text into subagent prompts.

    Already-cached abstracts are served from disk without an HTTP request, so calling
    this again with overlapping PMIDs is cheap.

    Args:
        pmids: PMIDs as strings. Anything that isn't all digits is rejected rather
            than sent — PubMed would otherwise return an unrelated paper for it.

    Returns:
        records: {pmid: {title, abstract, sections, journal, year, retracted,
            publication_types}}. `abstract` is None for errata and editorials, which
            have citation metadata but no body. `sections` preserves structured-abstract
            labels (BACKGROUND / METHODS / FINDINGS / ...) when the journal uses them.
            `retracted` is True for retracted papers — always surface that to the user.
        missing: requested PMIDs that PubMed returned nothing for
        invalid: inputs rejected as malformed
        from_cache: PMIDs served from the local cache
    """
    valid, invalid = validate_pmids([str(p) for p in pmids])

    ABSTRACT_CACHE.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    from_cache: list[str] = []
    to_fetch: list[str] = []

    for pmid in valid:
        path = ABSTRACT_CACHE / f"{pmid}.json"
        if path.exists():
            try:
                records[pmid] = json.loads(path.read_text())
                from_cache.append(pmid)
                continue
            except (OSError, json.JSONDecodeError):
                pass  # corrupt cache entry, refetch
        to_fetch.append(pmid)

    for chunk in _chunks(to_fetch, FETCH_CHUNK):
        # retmode=xml, never the text mode: text concatenates every abstract behind a
        # positional counter that renumbers when records drop, so it can't be mapped
        # back to the requested ids. Never pass retmax — it truncates silently.
        resp = await _request("efetch", db="pubmed", id=",".join(chunk), retmode="xml")
        for pmid, rec in _parse_efetch(resp.text).items():
            records[pmid] = rec
            (ABSTRACT_CACHE / f"{pmid}.json").write_text(json.dumps(rec, indent=2))

    return {
        "records": records,
        "missing": [p for p in valid if p not in records],
        "invalid": invalid,
        "from_cache": from_cache,
    }
