"""Run-local provenance for scientific source tools.

Source calls happen inside QuickJS, below ordinary LangGraph middleware. The source-error
wrapper already sits at that boundary, so it records compact provenance here while a
`research_trace()` context is active.

The trace deliberately excludes source bodies. It retains identifiers, queries,
attributed URLs, and call status for evaluation and audit without growing model context.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceEvent:
    """One scientific source-tool call reduced to evaluator-safe provenance."""

    source: str
    tool: str
    operation: str
    query: str | None = None
    requested_ids: tuple[str, ...] = ()
    returned_ids: tuple[str, ...] = ()
    related_pmids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("requested_ids", "returned_ids", "related_pmids", "urls"):
            value[key] = list(value[key])
        return value


@dataclass
class ResearchTrace:
    """Mutable collector owned by one runner invocation."""

    events: list[SourceEvent] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "pmids": _summary_pmids(self.events),
            "pmcids": _summary_ids(self.events, "pmc"),
            "nct_ids": _summary_ids(self.events, "ctgov"),
            "urls": _unique(url for event in self.events for url in event.urls),
        }


_CURRENT_TRACE: ContextVar[ResearchTrace | None] = ContextVar(
    "deep_life_sci_research_trace", default=None
)

_TOOL_META = {
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
    """Record a source return without ever changing the source call's behavior."""
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    try:
        call_args = _arguments(func, args, kwargs)
        event = _event(tool_name, call_args, result)
        if event is not None:
            trace.events.append(event)
    except Exception:  # noqa: BLE001 - provenance is observational
        pass


def _arguments(func: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        return dict(inspect.signature(func).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _event(tool_name: str, args: dict[str, Any], result: Any) -> SourceEvent | None:
    meta = _TOOL_META.get(tool_name)
    if meta is None:
        return None
    source, operation = meta
    error = _result_error(tool_name, result)
    return SourceEvent(
        source=source,
        tool=tool_name,
        operation=operation,
        query=_query(tool_name, args, result),
        requested_ids=_requested_ids(tool_name, args),
        returned_ids=_returned_ids(tool_name, result),
        related_pmids=_related_pmids(tool_name, result),
        urls=_urls(tool_name, result),
        success=error is None,
        error=error,
    )


def _query(tool_name: str, args: dict[str, Any], result: Any) -> str | None:
    if tool_name == "pubmed_search":
        return _text(args.get("term"))
    if tool_name == "web_search":
        return _text(args.get("query"))
    if tool_name == "ctgov_search":
        sent = result.get("query_sent") if isinstance(result, dict) else None
        if isinstance(sent, dict):
            return json.dumps(sent, sort_keys=True, separators=(",", ":"))
        keys = (
            "condition", "intervention", "term", "title", "sponsor", "status",
            "filter_advanced",
        )
        selected = {key: args[key] for key in keys if args.get(key)}
        if selected:
            return json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return None


def _requested_ids(tool_name: str, args: dict[str, Any]) -> tuple[str, ...]:
    key = {
        "fetch_abstracts": "pmids",
        "pmc_locate": "pmcids",
        "fetch_full_text": "pmcids",
        "ctgov_fetch": "nct_ids",
    }.get(tool_name)
    if key is not None:
        return _ids(args.get(key))
    if tool_name in {"fetch_figures", "fetch_supplementary"}:
        return _ids(args.get("pmcid"))
    return ()


def _returned_ids(tool_name: str, result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()

    if tool_name in {"pubmed_search", "ctgov_search"}:
        field = "pmid" if tool_name == "pubmed_search" else "nct_id"
        records = result.get("records")
        if isinstance(records, list):
            return _ids(
                record.get(field) for record in records if isinstance(record, dict)
            )

    if tool_name in {"fetch_abstracts", "fetch_full_text", "ctgov_fetch"}:
        records = result.get("records")
        return _ids(records.keys()) if isinstance(records, dict) else ()

    if tool_name == "pmc_locate":
        available = result.get("available")
        return _ids(available.keys()) if isinstance(available, dict) else ()

    if tool_name in {"fetch_figures", "fetch_supplementary"}:
        staged = result.get("staged")
        if isinstance(staged, list):
            return _ids(
                item.get("pmcid") for item in staged if isinstance(item, dict)
            )

    return ()


def _related_pmids(tool_name: str, result: Any) -> tuple[str, ...]:
    """PMIDs carried by PMC records, kept alongside the PMCID retrieval identity."""
    if not isinstance(result, dict):
        return ()
    key = {
        "pmc_locate": "available",
        "fetch_full_text": "records",
    }.get(tool_name)
    if key is None:
        return ()
    records = result.get(key)
    if not isinstance(records, dict):
        return ()
    return _ids(
        record.get("pmid")
        for record in records.values()
        if isinstance(record, dict)
    )


def _urls(tool_name: str, result: Any) -> tuple[str, ...]:
    if tool_name != "web_search" or not isinstance(result, dict):
        return ()
    sources = result.get("sources")
    if not isinstance(sources, list):
        return ()
    return tuple(
        _unique(
            source.get("url")
            for source in sources
            if isinstance(source, dict)
        )
    )


def _result_error(tool_name: str, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    if error := result.get("error"):
        return _text(error)
    if tool_name == "web_search" and not result.get("searched"):
        warnings = result.get("warnings")
        if isinstance(warnings, list) and warnings:
            return _text(warnings[0])
    return None


def _summary_pmids(events: list[SourceEvent]) -> list[str]:
    """PMIDs whose source content this run fetched directly or accessed through PMC.

    PubMed search hits stay on their search event but are deliberately excluded here:
    seeing an identifier in a result list is not evidence that the paper was read.
    """
    return _unique(
        pmid
        for event in events
        for pmid in (
            event.returned_ids
            if event.tool == "fetch_abstracts"
            else event.related_pmids
        )
    )


def _summary_ids(events: list[SourceEvent], source: str) -> list[str]:
    return _unique(
        event_id
        for event in events
        if event.source == source
        for event_id in event.returned_ids
    )


def _ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, int)):
        values = [values]
    try:
        return tuple(_unique(values))
    except TypeError:
        return ()


def _unique(values) -> list[str]:
    return list(dict.fromkeys(text for value in values if (text := _text(value))))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
