"""Run-scoped provenance around scientific source-tool calls."""

import asyncio

from langchain_core.tools import tool

from deep_life_sci.middleware.source_trace import research_trace
from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.sources._errors import SourceError


@tool
async def fetch_abstracts(pmids: list[str]) -> dict:
    """Small stand-in for PubMed abstract retrieval."""
    return {
        "records": {
            pmid: {"pmid": pmid, "abstract": f"large payload for {pmid}"}
            for pmid in pmids
        },
        "missing": [],
        "invalid": [],
        "from_cache": [],
    }


@tool
async def pubmed_search(term: str, retmax: int = 50) -> dict:
    """Small stand-in for PubMed search."""
    return {
        "count": 1,
        "returned": 1 if retmax else 0,
        "query_translation": term,
        "warnings": [],
        "records": [{"pmid": "123", "title": "paper"}] if retmax else [],
    }


@tool
async def ctgov_search(condition: str | None = None, retmax: int = 50) -> dict:
    """Small stand-in for ClinicalTrials.gov search."""
    return {
        "count": 1,
        "returned": 1 if retmax else 0,
        "query_sent": {"query.cond": condition, "sort": "@relevance"},
        "warnings": [],
        "records": [
            {"nct_id": "NCT00000001", "title": "trial", "brief_summary": "large payload"}
        ] if retmax else [],
    }


@tool
async def ctgov_fetch(nct_ids: list[str]) -> dict:
    """Small stand-in for ClinicalTrials.gov record retrieval."""
    return {
        "records": {
            "NCT00000001": {
                "nct_id": "NCT00000001",
                "brief_summary": "large registry payload",
            }
        } if "NCT00000001" in nct_ids else {},
        "missing": [nct_id for nct_id in nct_ids if nct_id != "NCT00000001"],
        "invalid": [],
        "from_cache": [],
    }


@tool
async def pmc_locate(pmcids: list[str]) -> dict:
    """Small stand-in for PMC triage with its linked PubMed identity."""
    return {
        "available": {
            pmcid: {"pmcid": pmcid, "pmid": "123", "title": "paper"}
            for pmcid in pmcids
        },
        "unavailable": [],
        "invalid": [],
    }


@tool
async def fetch_full_text(pmcids: list[str]) -> dict:
    """Small stand-in for PMC full-text retrieval with large text omitted by tracing."""
    return {
        "records": {
            pmcid: {"pmcid": pmcid, "pmid": "123", "text": "large full text"}
            for pmcid in pmcids
        },
        "unavailable": [],
        "invalid": [],
    }


@tool
async def failing_source(term: str) -> dict:
    """Source call that raises an expected source failure."""
    raise SourceError(f"unavailable for {term}")


@tool
async def web_search(query: str) -> dict:
    """Small stand-in for provider-side web search."""
    return {
        "query": query,
        "answer": "large digest body",
        "sources": [{"url": "https://example.org/evidence", "title": "Evidence"}],
        "searched": ["evidence query"],
        "warnings": [],
    }


def _renamed(tool_value, name: str):
    return tool_value.model_copy(update={"name": name})


async def test_inactive_trace_preserves_source_return_value():
    wrapped = with_error_capture([fetch_abstracts])[0]
    result = await wrapped.ainvoke({"pmids": ["123"]})
    assert result == await fetch_abstracts.ainvoke({"pmids": ["123"]})


async def test_trace_keeps_identifiers_but_not_scientific_payloads():
    wrapped = with_error_capture([fetch_abstracts])[0]

    with research_trace() as trace:
        result = await wrapped.ainvoke({"pmids": ["123", "456"]})

    assert "large payload" in result["records"]["123"]["abstract"]
    data = trace.as_dict()
    assert data["pmids"] == ["123", "456"]
    assert data["pmcids"] == []
    assert data["nct_ids"] == []
    assert len(data["events"]) == 1
    event = data["events"][0]
    assert event["requested_ids"] == ["123", "456"]
    assert event["returned_ids"] == ["123", "456"]
    assert event["success"] is True
    assert "large payload" not in repr(data)


async def test_partial_fetch_preserves_missing_id_as_requested_not_returned():
    @tool
    async def partial_fetch(pmids: list[str]) -> dict:
        """Return one record while explicitly reporting another PMID as missing."""
        return {
            "records": {"123": {"pmid": "123", "abstract": "present"}},
            "missing": ["999"],
            "invalid": [],
            "from_cache": [],
        }

    wrapped = with_error_capture([_renamed(partial_fetch, "fetch_abstracts")])[0]

    with research_trace() as trace:
        result = await wrapped.ainvoke({"pmids": ["123", "999"]})

    assert result["missing"] == ["999"]
    event = trace.as_dict()["events"][0]
    assert event["requested_ids"] == ["123", "999"]
    assert event["returned_ids"] == ["123"]
    assert trace.as_dict()["pmids"] == ["123"]
    assert "999" not in trace.as_dict()["pmids"]


async def test_search_query_and_returned_ids_are_recorded():
    wrapped = with_error_capture([pubmed_search])[0]

    with research_trace() as trace:
        await wrapped.ainvoke({"term": "Shank3[tiab]", "retmax": 10})

    event = trace.as_dict()["events"][0]
    assert event["source"] == "pubmed"
    assert event["operation"] == "search"
    assert event["query"] == "Shank3[tiab]"
    assert event["returned_ids"] == ["123"]
    assert trace.as_dict()["pmids"] == []


async def test_ctgov_search_fetch_and_missing_ids_are_traceable_without_payloads():
    search, fetch = with_error_capture([ctgov_search, ctgov_fetch])

    with research_trace() as trace:
        search_result = await search.ainvoke({"condition": "ADHD", "retmax": 10})
        fetch_result = await fetch.ainvoke(
            {"nct_ids": ["NCT00000001", "NCT99999999"]}
        )

    assert search_result["records"][0]["brief_summary"] == "large payload"
    assert fetch_result["missing"] == ["NCT99999999"]

    data = trace.as_dict()
    assert data["nct_ids"] == ["NCT00000001"]
    assert len(data["events"]) == 2

    search_event, fetch_event = data["events"]
    assert search_event["source"] == "ctgov"
    assert search_event["operation"] == "search"
    assert search_event["returned_ids"] == ["NCT00000001"]
    assert '"query.cond":"ADHD"' in search_event["query"]

    assert fetch_event["source"] == "ctgov"
    assert fetch_event["operation"] == "fetch"
    assert fetch_event["requested_ids"] == ["NCT00000001", "NCT99999999"]
    assert fetch_event["returned_ids"] == ["NCT00000001"]
    assert fetch_event["success"] is True

    assert "NCT99999999" not in data["nct_ids"]
    assert "large payload" not in repr(data)
    assert "large registry payload" not in repr(data)


async def test_pmc_reads_keep_pmcid_and_linked_pmid_provenance():
    locate, full_text = with_error_capture([pmc_locate, fetch_full_text])

    with research_trace() as trace:
        await locate.ainvoke({"pmcids": ["PMC123"]})
        result = await full_text.ainvoke({"pmcids": ["PMC123"]})

    assert result["records"]["PMC123"]["text"] == "large full text"
    data = trace.as_dict()
    assert data["pmcids"] == ["PMC123"]
    assert data["pmids"] == ["123"]
    assert data["events"][0]["related_pmids"] == ["123"]
    assert data["events"][1]["related_pmids"] == ["123"]
    assert "large full text" not in repr(data)


async def test_contained_failure_keeps_return_shape_and_records_failure():
    failing = _renamed(failing_source, "pubmed_search")
    wrapped = with_error_capture([failing])[0]

    with research_trace() as trace:
        result = await wrapped.ainvoke({"term": "x"})

    assert result == {"error": "pubmed_search failed: unavailable for x"}
    event = trace.as_dict()["events"][0]
    assert event["success"] is False
    assert event["query"] == "x"
    assert "unavailable for x" in event["error"]


async def test_web_trace_keeps_urls_but_not_digest_body():
    wrapped = with_error_capture([web_search])[0]

    with research_trace() as trace:
        result = await wrapped.ainvoke({"query": "What changed?"})

    assert result["answer"] == "large digest body"
    data = trace.as_dict()
    assert data["urls"] == ["https://example.org/evidence"]
    assert data["events"][0]["query"] == "What changed?"
    assert "large digest body" not in repr(data)


async def test_sequential_trace_contexts_do_not_leak():
    wrapped = with_error_capture([fetch_abstracts])[0]

    with research_trace() as first:
        await wrapped.ainvoke({"pmids": ["111"]})
    with research_trace() as second:
        await wrapped.ainvoke({"pmids": ["222"]})

    assert first.as_dict()["pmids"] == ["111"]
    assert second.as_dict()["pmids"] == ["222"]


async def test_concurrent_trace_contexts_do_not_cross_contaminate():
    wrapped = with_error_capture([fetch_abstracts])[0]

    async def one(pmid: str) -> list[str]:
        with research_trace() as trace:
            await asyncio.sleep(0)
            await wrapped.ainvoke({"pmids": [pmid]})
            await asyncio.sleep(0)
        return trace.as_dict()["pmids"]

    first, second = await asyncio.gather(one("111"), one("222"))
    assert first == ["111"]
    assert second == ["222"]
