"""Run-local provenance for scientific source tools.

Source calls happen inside QuickJS, below ordinary LangGraph middleware. The source-error
wrapper already sits at that boundary and sees every wrapped call, so it records
provenance here while a `research_trace()` context is active.

The trace records what tools were asked and what they returned. It does not interpret:
whether a paper was understood, whether a source supports a claim, or whether an answer
cites something it never fetched are evaluator questions, answered in `evals/` from the
events recorded here. Three rules keep the record honest:

* **Observation, not inference.** Each event carries the tool's operation and the IDs it
  returned. The summary projects those into `discovered` (appeared in a search result),
  `located` (`pmc_locate` found full text, which was not fetched by that call), and
  `retrieved` (abstract, full text, staged files, or registry record actually returned).
  A search hit is never retrieved, and a located article is never read. The summary
  also names `unextracted_tools` and whether `discovered` was truncated, so coverage
  gaps are visible without scanning events.
* **Requested and resolved identifiers both survive.** The model may type `5904197` or
  `pmc123`; the sources normalise before fetching. `requested_ids` keeps the model's
  spelling and `resolved_ids` the canonical form, so joins against `returned_ids` work
  without pretending the model typed the canonical form.
* **A lost record is reported, never dropped.** Every wrapped tool produces an event. A
  tool with no extractor produces a bare event with `extracted: false`. An extractor
  that raises is logged, counted in `capture_failures`, and flips `complete` to false.
  Evaluators must treat an incomplete trace as "unknown", not as "did not happen".

Bodies are never copied: no abstracts, full text, trial payloads, or web digests. Search
events keep at most `MAX_SEARCH_IDS` identifiers plus the full count, so a wide search
cannot bloat an eval record.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from deep_life_sci.sources.ctgov import validate_nct_ids
from deep_life_sci.sources.pubmed import normalize_pmcid, validate_pmids

logger = logging.getLogger(__name__)

MAX_SEARCH_IDS = 200
MAX_TEXT_CHARS = 500
MAX_LIST_ITEMS = 50

# tool -> (source, operation). `operation` names what the tool does, not what the model
# did with the result: `locate` is triage without content, `read` is full text returned.
EVIDENCE_SOURCE_TOOLS: dict[str, tuple[str, str]] = {
    "pubmed_search": ("pubmed", "search"),
    "fetch_abstracts": ("pubmed", "fetch"),
    "pmc_locate": ("pmc", "locate"),
    "fetch_full_text": ("pmc", "read"),
    "fetch_figures": ("pmc", "stage_figure"),
    "fetch_supplementary": ("pmc", "stage_supplementary"),
    "ctgov_search": ("ctgov", "search"),
    "ctgov_fetch": ("ctgov", "fetch"),
    "web_search": ("web", "search"),
}

_ID_ARGS: dict[str, tuple[str, str]] = {
    "fetch_abstracts": ("pmids", "pmid"),
    "pmc_locate": ("pmcids", "pmcid"),
    "fetch_full_text": ("pmcids", "pmcid"),
    "fetch_figures": ("pmcid", "pmcid"),
    "fetch_supplementary": ("pmcid", "pmcid"),
    "ctgov_fetch": ("nct_ids", "nct_id"),
}

_WEB_FAILURES = ("web search unavailable", "search ", "no search was performed")


@dataclass(frozen=True)
class SourceEvent:
    """One wrapped tool call reduced to evaluator-safe provenance."""

    tool: str
    status: str  # "ok" | "error"
    source: str | None = None
    operation: str | None = None
    extracted: bool = True
    error: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    effective_query: str | None = None
    requested_ids: tuple[str, ...] = ()
    resolved_ids: tuple[str, ...] = ()
    returned_ids: tuple[str, ...] = ()
    returned_count: int | None = None
    result_count: int | None = None
    warnings: tuple[str, ...] = ()
    related_pmids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "requested_ids", "resolved_ids", "returned_ids", "warnings",
            "related_pmids", "urls",
        ):
            value[key] = list(value[key])
        return value


@dataclass
class ResearchTrace:
    """Mutable collector owned by one runner invocation."""

    events: list[SourceEvent] = field(default_factory=list)
    capture_failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.capture_failures

    def as_dict(self) -> dict[str, Any]:
        events = self.events
        return {
            "complete": self.complete,
            "capture_failures": list(self.capture_failures),
            "events": [event.as_dict() for event in events],
            "unextracted_tools": _unique(e.tool for e in events if not e.extracted),
            "discovered": {
                "pmids": _returned(events, "pubmed_search"),
                "nct_ids": _returned(events, "ctgov_search"),
                # True when a search returned more IDs than an event stores: an ID
                # absent from these lists may still have been discovered.
                "truncated": _truncated(events, "pubmed_search", "ctgov_search"),
            },
            "located": {
                "pmcids": _returned(events, "pmc_locate"),
                "pmids": _related(events, "pmc_locate"),
            },
            "retrieved": {
                "abstracts": _returned(events, "fetch_abstracts"),
                "full_text": _returned(events, "fetch_full_text"),
                "full_text_pmids": _related(events, "fetch_full_text"),
                "files": _returned(events, "fetch_figures", "fetch_supplementary"),
                "trials": _returned(events, "ctgov_fetch"),
                "web_urls": _unique(url for e in events for url in e.urls),
            },
        }


_CURRENT_TRACE: ContextVar[ResearchTrace | None] = ContextVar(
    "deep_life_sci_research_trace", default=None
)


@contextmanager
def research_trace() -> Iterator[ResearchTrace]:
    """Collect source events in this execution context and reset it on exit."""
    trace = ResearchTrace()
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        _CURRENT_TRACE.reset(token)


def record_source_call(
    tool_name: str,
    func: Callable,
    args: tuple,
    kwargs: dict,
    *,
    result: Any,
) -> None:
    """Record a source return without ever changing the source call's behavior.

    The except branch touches nothing the extractor touched: a plain list append and the
    standard logger, so a broken extractor leaves a visible mark rather than a gap.
    """
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    try:
        event = _event(tool_name, _arguments(func, args, kwargs), result)
    except Exception as exc:  # provenance must not alter the source call
        logger.exception("provenance capture failed for %s", tool_name)
        trace.capture_failures.append(
            {"tool": tool_name, "error": f"{type(exc).__name__}: {exc}"}
        )
        return
    trace.events.append(event)


def _arguments(func: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        return dict(inspect.signature(func).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _event(tool_name: str, args: dict[str, Any], result: Any) -> SourceEvent:
    error = _error(tool_name, result)
    meta = EVIDENCE_SOURCE_TOOLS.get(tool_name)
    if meta is None:
        return SourceEvent(
            tool=tool_name,
            status="error" if error else "ok",
            extracted=False,
            error=error,
            request=_bounded(args),
        )
    source, operation = meta
    id_arg = _ID_ARGS.get(tool_name)
    requested = _ids(args.get(id_arg[0])) if id_arg else ()
    returned = _returned_ids(tool_name, result)
    truncate = operation == "search"
    return SourceEvent(
        tool=tool_name,
        status="error" if error else "ok",
        source=source,
        operation=operation,
        error=error,
        request=_bounded({k: v for k, v in args.items() if not id_arg or k != id_arg[0]}),
        effective_query=_effective_query(tool_name, result),
        requested_ids=requested,
        resolved_ids=_resolve(id_arg[1], requested) if id_arg else (),
        returned_ids=returned[:MAX_SEARCH_IDS] if truncate else returned,
        returned_count=len(returned),
        result_count=_int(result.get("count")) if isinstance(result, dict) else None,
        warnings=_warnings(result),
        related_pmids=_related_pmids(tool_name, result),
        urls=_urls(tool_name, result),
    )


def _effective_query(tool_name: str, result: Any) -> str | None:
    """What the external system actually ran, when the tool reports it."""
    if not isinstance(result, dict):
        return None
    if tool_name == "pubmed_search":
        return _text(result.get("query_translation"))
    if tool_name == "ctgov_search":
        sent = result.get("query_sent")
        if isinstance(sent, dict):
            return json.dumps(sent, sort_keys=True, separators=(",", ":"))
    return None


def _resolve(kind: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical form of each requested ID, through the sources' own validators."""
    if kind == "pmid":
        return tuple(validate_pmids(list(requested))[0])
    if kind == "nct_id":
        return tuple(validate_nct_ids(list(requested))[0])
    return tuple(_unique(normalize_pmcid(value) for value in requested))


def _returned_ids(tool_name: str, result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()
    if tool_name in {"pubmed_search", "ctgov_search"}:
        key = "pmid" if tool_name == "pubmed_search" else "nct_id"
        records = result.get("records")
        if isinstance(records, list):
            return _ids(r.get(key) for r in records if isinstance(r, dict))
    if tool_name in {"fetch_abstracts", "fetch_full_text", "ctgov_fetch"}:
        records = result.get("records")
        return _ids(records.keys()) if isinstance(records, dict) else ()
    if tool_name == "pmc_locate":
        available = result.get("available")
        return _ids(available.keys()) if isinstance(available, dict) else ()
    if tool_name in {"fetch_figures", "fetch_supplementary"}:
        staged = result.get("staged")
        if isinstance(staged, list):
            return _ids(item.get("pmcid") for item in staged if isinstance(item, dict))
    return ()


def _related_pmids(tool_name: str, result: Any) -> tuple[str, ...]:
    """PMIDs carried by PMC records, kept alongside the PMCID retrieval identity."""
    key = {"pmc_locate": "available", "fetch_full_text": "records"}.get(tool_name)
    if key is None or not isinstance(result, dict):
        return ()
    records = result.get(key)
    if not isinstance(records, dict):
        return ()
    return _ids(r.get("pmid") for r in records.values() if isinstance(r, dict))


def _urls(tool_name: str, result: Any) -> tuple[str, ...]:
    if tool_name != "web_search" or not isinstance(result, dict):
        return ()
    sources = result.get("sources")
    if not isinstance(sources, list):
        return ()
    return tuple(_unique(s.get("url") for s in sources if isinstance(s, dict)))


def _warnings(result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        return ()
    return tuple(_text(w)[:MAX_TEXT_CHARS] for w in warnings if _text(w))


def _error(tool_name: str, result: Any) -> str | None:
    """Technical failure only. Zero results, truncation, and query repairs are not errors.

    `web_search` contains its own failures as warnings rather than `{error}`
    (`sources/web.py:_warnings` and `_failed`); those start with `web search
    unavailable`, `search `, or `no search was performed`. A truncated answer is a
    successful search.
    """
    if not isinstance(result, dict):
        return None
    if error := result.get("error"):
        return _text(error)
    if tool_name == "web_search":
        for warning in _warnings(result):
            if warning.startswith(_WEB_FAILURES):
                return warning
    return None


def _returned(events: list[SourceEvent], *tools: str) -> list[str]:
    return _unique(i for e in events if e.tool in tools for i in e.returned_ids)


def _truncated(events: list[SourceEvent], *tools: str) -> bool:
    return any(
        e.tool in tools and e.returned_count is not None
        and e.returned_count > len(e.returned_ids)
        for e in events
    )


def _related(events: list[SourceEvent], tool: str) -> list[str]:
    return _unique(i for e in events if e.tool == tool for i in e.related_pmids)


def _bounded(args: dict[str, Any]) -> dict[str, Any]:
    """Request arguments small enough for an eval record, with nothing dropped silently."""
    out: dict[str, Any] = {}
    for key, value in args.items():
        if value is None or isinstance(value, (bool, int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value[:MAX_TEXT_CHARS]
        elif isinstance(value, (list, tuple)):
            items = [str(v)[:MAX_TEXT_CHARS] for v in value[:MAX_LIST_ITEMS]]
            if len(value) > MAX_LIST_ITEMS:
                items.append(f"+{len(value) - MAX_LIST_ITEMS} more")
            out[key] = items
        else:
            out[key] = f"<{type(value).__name__}>"
    return out


def _ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, int)):
        values = [values]
    try:
        return tuple(_unique(values))
    except TypeError:
        return ()


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values) -> list[str]:
    return list(dict.fromkeys(text for value in values if (text := _text(value))))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
