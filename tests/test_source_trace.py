"""Run-scoped provenance around the real source tools, with their transport mocked.

The tools under test are the production ones from `deep_life_sci.sources`, wrapped the way
`agent.py` wraps them, so a change to a tool's return shape that breaks extraction fails
here rather than in an eval. Only the network is faked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.tools import tool

from deep_life_sci.middleware import source_trace
from deep_life_sci.middleware.source_trace import (
    MAX_SEARCH_IDS,
    research_trace,
)
from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.sources import pmc, web
from deep_life_sci.sources.ctgov import ctgov_fetch, ctgov_search
from deep_life_sci.sources.pmc import fetch_full_text, pmc_locate
from deep_life_sci.sources.pubmed import fetch_abstracts, pubmed_search
from deep_life_sci.sources.web import web_search
from tests.conftest import json_response

# --------------------------------------------------------------------------------
# Fakes for the network, not for the tools
# --------------------------------------------------------------------------------


def _pubmed(pmids: list[str], *, translation: str = "translated"):
    """E-utilities serving `pmids` for every search, plus summaries and abstracts."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/esearch.fcgi"):
            hits = [] if request.url.params.get("retmax") == "0" else pmids
            return json_response({"esearchresult": {
                "count": str(len(pmids)), "idlist": hits, "querytranslation": translation,
            }})
        if path.endswith("/esummary.fcgi"):
            ids = request.url.params["id"].split(",")
            return json_response({"result": {"uids": ids, **{
                p: {"uid": p, "title": f"Paper {p}", "pubdate": "2025",
                    "source": "J", "authors": [], "articleids": []}
                for p in ids
            }}})
        if path.endswith("/efetch.fcgi"):
            ids = request.url.params["id"].split(",")
            articles = "".join(
                f"<PubmedArticle><MedlineCitation><PMID>{p}</PMID><Article>"
                f"<ArticleTitle>Paper {p}</ArticleTitle>"
                f"<Abstract><AbstractText>Secret abstract {p}</AbstractText></Abstract>"
                "<Journal><JournalIssue><PubDate><Year>2025</Year></PubDate>"
                "</JournalIssue><Title>J</Title></Journal></Article></MedlineCitation>"
                "<PubmedData><ArticleIdList>"
                f'<ArticleId IdType="pubmed">{p}</ArticleId></ArticleIdList></PubmedData>'
                "</PubmedArticle>"
                for p in ids
            )
            return httpx.Response(200, text=f"<PubmedArticleSet>{articles}</PubmedArticleSet>")
        raise AssertionError(f"unexpected request {request.url}")

    return handler


def _trial(nct_id: str) -> dict:
    return {"protocolSection": {"identificationModule": {
        "nctId": nct_id, "briefTitle": f"Trial {nct_id}", "briefSummary": "secret summary",
    }}}


def _ctgov(nct_ids: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        wanted = request.url.params.get("filter.ids")
        hits = [n for n in nct_ids if not wanted or n in wanted.split(",")]
        return json_response({"totalCount": len(hits), "studies": [_trial(n) for n in hits]})

    return handler


@pytest.fixture
def pmc_articles(monkeypatch):
    """`pmc._load` serving parsed articles by canonical PMCID, with their linked PMIDs."""
    articles: dict[str, str] = {}

    async def load(_client, pmcid):
        if pmcid not in articles:
            return None
        sidecar = {"pmid": articles[pmcid], "title": f"Paper {pmcid}", "license_code": "CC BY"}
        parsed = {
            "sections": [{"title": "Results", "canonical": "results", "chars": 11,
                          "text": "secret body"}],
            "figures": [], "tables": [], "supplementary": [],
        }
        return {}, sidecar, parsed

    monkeypatch.setattr(pmc, "_load", load)
    return articles


def _web_model(monkeypatch, *blocks: dict) -> None:
    message = SimpleNamespace(content=list(blocks))

    class Model:
        async def ainvoke(self, _prompt):
            return message

    monkeypatch.setattr(web, "web_search_model", lambda: Model())


_CITED = {"type": "text", "text": "Approved.",
          "annotations": [{"type": "url_citation", "url": "https://fda.gov/a", "title": "FDA"}]}


# --------------------------------------------------------------------------------
# Discovered, located, retrieved
# --------------------------------------------------------------------------------


async def test_search_hits_are_discovered_and_only_fetched_abstracts_are_retrieved(mock_ncbi):
    """Twenty hits, two opened: the summary must say twenty discovered, two retrieved."""
    pmids = [str(1000 + i) for i in range(20)]
    mock_ncbi(_pubmed(pmids, translation='"shank3"[All Fields]'))
    search, fetch = with_error_capture([pubmed_search, fetch_abstracts])

    with research_trace() as trace:
        await search.ainvoke({"term": "shank3", "mindate": "2020", "maxdate": "2025"})
        result = await fetch.ainvoke({"pmids": pmids[:2]})

    assert "Secret abstract" in result["records"][pmids[0]]["abstract"]
    data = trace.as_dict()
    assert data["complete"] is True
    assert data["discovered"]["pmids"] == pmids
    assert data["discovered"]["truncated"] is False
    assert data["unextracted_tools"] == []
    assert data["retrieved"]["abstracts"] == pmids[:2]
    search_event, fetch_event = data["events"]
    assert search_event["operation"] == "search"
    assert search_event["status"] == "ok"
    assert search_event["request"]["term"] == "shank3"
    assert search_event["request"]["mindate"] == "2020"
    assert search_event["request"]["maxdate"] == "2025"
    assert search_event["effective_query"] == '"shank3"[All Fields]'
    assert search_event["result_count"] == 20
    assert search_event["returned_count"] == 20
    assert fetch_event["requested_ids"] == pmids[:2]
    assert fetch_event["returned_ids"] == pmids[:2]
    assert "Secret abstract" not in repr(data)


async def test_located_full_text_is_not_retrieved_until_fetched(pmc_articles):
    pmc_articles["PMC5904197"] = "29695998"
    locate, read = with_error_capture([pmc_locate, fetch_full_text])

    with research_trace() as trace:
        await locate.ainvoke({"pmcids": ["5904197", "pmc999"]})
    located = trace.as_dict()
    assert located["located"] == {"pmcids": ["PMC5904197"], "pmids": ["29695998"]}
    assert located["retrieved"]["full_text"] == []
    assert located["retrieved"]["full_text_pmids"] == []
    event = located["events"][0]
    assert event["operation"] == "locate"
    assert event["requested_ids"] == ["5904197", "pmc999"]
    assert event["resolved_ids"] == ["PMC5904197", "PMC999"]
    assert event["returned_ids"] == ["PMC5904197"]

    with research_trace() as trace:
        result = await read.ainvoke({"pmcids": ["5904197"]})
    assert result["records"]["PMC5904197"]["text"].startswith("## Results")
    data = trace.as_dict()
    assert data["retrieved"]["full_text"] == ["PMC5904197"]
    assert data["retrieved"]["full_text_pmids"] == ["29695998"]
    assert data["events"][0]["operation"] == "read"
    assert "secret body" not in repr(data)


async def test_trial_search_hits_are_discovered_and_only_fetched_records_retrieved(mock_ctgov):
    mock_ctgov(_ctgov(["NCT00000001", "NCT00000002"]))
    search, fetch = with_error_capture([ctgov_search, ctgov_fetch])

    with research_trace() as trace:
        await search.ainvoke({"condition": "ADHD", "status": ["COMPLETED"]})
        await fetch.ainvoke({"nct_ids": ["nct00000001", "NCT00000009"]})

    data = trace.as_dict()
    assert data["discovered"]["nct_ids"] == ["NCT00000001", "NCT00000002"]
    assert data["retrieved"]["trials"] == ["NCT00000001"]
    search_event, fetch_event = data["events"]
    assert search_event["request"]["condition"] == "ADHD"
    assert search_event["request"]["status"] == ["COMPLETED"]
    assert '"query.cond":"ADHD"' in search_event["effective_query"]
    assert search_event["result_count"] == 2
    assert fetch_event["requested_ids"] == ["nct00000001", "NCT00000009"]
    assert fetch_event["resolved_ids"] == ["NCT00000001", "NCT00000009"]
    assert fetch_event["returned_ids"] == ["NCT00000001"]
    assert "secret summary" not in repr(data)


# --------------------------------------------------------------------------------
# Status: technical outcome only
# --------------------------------------------------------------------------------


async def test_a_search_with_zero_results_is_a_successful_call(mock_ncbi):
    mock_ncbi(_pubmed([]))
    with research_trace() as trace:
        await with_error_capture([pubmed_search])[0].ainvoke({"term": "nothing"})
    event = trace.as_dict()["events"][0]
    assert event["status"] == "ok"
    assert event["error"] is None
    assert event["returned_count"] == 0
    assert event["result_count"] == 0


async def test_a_source_error_is_recorded_with_the_same_request_shape(mock_ctgov):
    """A failed ClinicalTrials.gov search keeps the keys a successful one records."""
    mock_ctgov(_ctgov(["NCT00000001"]))
    search = with_error_capture([ctgov_search])[0]

    with research_trace() as trace:
        failed = await search.ainvoke({})
        await search.ainvoke({"condition": "ADHD"})

    assert set(failed) == {"error"}
    failed_event, ok_event = trace.as_dict()["events"]
    assert failed_event["status"] == "error"
    assert "ctgov_search needs at least one" in failed_event["error"]
    assert failed_event["effective_query"] is None
    assert set(failed_event) == set(ok_event)
    assert set(failed_event["request"]) == set(ok_event["request"])


async def test_web_truncation_is_not_a_failure_but_a_search_error_is(monkeypatch):
    wrapped = with_error_capture([web_search])[0]

    _web_model(monkeypatch, {"type": "web_search_call", "status": "completed"},
               {**_CITED, "text": "x" * (web.MAX_ANSWER_CHARS + 1)})
    with research_trace() as trace:
        result = await wrapped.ainvoke({"query": "long"})
    assert any("truncated" in w for w in result["warnings"])
    event = trace.as_dict()["events"][0]
    assert event["status"] == "ok"
    assert trace.as_dict()["retrieved"]["web_urls"] == ["https://fda.gov/a"]
    assert "xxxx" not in repr(trace.as_dict())

    _web_model(monkeypatch, {"type": "web_search_tool_result",
                             "content": {"type": "web_search_tool_result_error",
                                         "error_code": "max_uses_exceeded"}},
               _CITED)
    with research_trace() as trace:
        await wrapped.ainvoke({"query": "errored"})
    event = trace.as_dict()["events"][0]
    assert event["status"] == "error"
    assert event["error"] == "search error: max_uses_exceeded"


async def test_a_web_search_that_cannot_run_is_a_failure(monkeypatch):
    class Exploding:
        async def ainvoke(self, _prompt):
            raise RuntimeError("400 bad request")

    monkeypatch.setattr(web, "web_search_model", lambda: Exploding())
    with research_trace() as trace:
        await with_error_capture([web_search])[0].ainvoke({"query": "q"})
    event = trace.as_dict()["events"][0]
    assert event["status"] == "error"
    assert event["error"].startswith("web search unavailable")


# --------------------------------------------------------------------------------
# Coverage and capture failures
# --------------------------------------------------------------------------------


async def test_a_wrapped_tool_without_an_extractor_is_still_recorded():
    @tool
    async def future_tool(topic: str, big: dict) -> dict:
        """A tool added after the trace, with no extractor yet."""
        return {"records": [{"id": "x"}]}

    with research_trace() as trace:
        await with_error_capture([future_tool])[0].ainvoke(
            {"topic": "t" * 1000, "big": {"payload": "..."}}
        )
    event = trace.as_dict()["events"][0]
    assert event["tool"] == "future_tool"
    assert event["extracted"] is False
    assert event["status"] == "ok"
    assert trace.as_dict()["unextracted_tools"] == ["future_tool"]
    assert trace.as_dict()["complete"] is True
    assert event["request"] == {"topic": "t" * source_trace.MAX_TEXT_CHARS, "big": "<dict>"}
    assert event["returned_ids"] == []


async def test_a_recording_failure_is_reported_and_leaves_the_tool_result_intact(
    monkeypatch, caplog
):
    def broken(*_args, **_kwargs):
        raise TypeError("extractor bug")

    monkeypatch.setattr(source_trace, "_returned_ids", broken)
    _web_model(monkeypatch, {"type": "web_search_call", "status": "completed"}, _CITED)

    with research_trace() as trace, caplog.at_level("ERROR", logger=source_trace.__name__):
        result = await with_error_capture([web_search])[0].ainvoke({"query": "q"})

    assert result["answer"] == "Approved."
    data = trace.as_dict()
    assert data["complete"] is False
    assert data["capture_failures"] == [
        {"tool": "web_search", "error": "TypeError: extractor bug"}
    ]
    assert data["events"] == []
    assert "provenance capture failed for web_search" in caplog.text


async def test_wide_search_results_are_capped_but_counted(mock_ncbi):
    pmids = [str(i) for i in range(1, MAX_SEARCH_IDS + 11)]
    mock_ncbi(_pubmed(pmids))
    with research_trace() as trace:
        await with_error_capture([pubmed_search])[0].ainvoke({"term": "wide", "retmax": 300})
    data = trace.as_dict()
    event = data["events"][0]
    assert len(event["returned_ids"]) == MAX_SEARCH_IDS
    assert event["returned_count"] == MAX_SEARCH_IDS + 10
    assert data["discovered"]["truncated"] is True


async def test_inactive_trace_preserves_source_return_value(mock_ncbi):
    mock_ncbi(_pubmed(["123"]))
    wrapped = with_error_capture([fetch_abstracts])[0]
    traced = await wrapped.ainvoke({"pmids": ["123"]})
    plain = await fetch_abstracts.ainvoke({"pmids": ["123"]})
    assert traced["records"] == plain["records"]  # the second call is served from cache


# --------------------------------------------------------------------------------
# Isolation between runs
# --------------------------------------------------------------------------------


async def test_sequential_trace_contexts_do_not_leak(mock_ncbi):
    mock_ncbi(_pubmed(["111", "222"]))
    wrapped = with_error_capture([fetch_abstracts])[0]
    with research_trace() as first:
        await wrapped.ainvoke({"pmids": ["111"]})
    with research_trace() as second:
        await wrapped.ainvoke({"pmids": ["222"]})
    assert first.as_dict()["retrieved"]["abstracts"] == ["111"]
    assert second.as_dict()["retrieved"]["abstracts"] == ["222"]


async def test_concurrent_trace_contexts_do_not_cross_contaminate(mock_ncbi):
    mock_ncbi(_pubmed(["111", "222"]))
    wrapped = with_error_capture([fetch_abstracts])[0]

    async def one(pmid: str) -> list[str]:
        with research_trace() as trace:
            await asyncio.sleep(0)
            await wrapped.ainvoke({"pmids": [pmid]})
            await asyncio.sleep(0)
        return trace.as_dict()["retrieved"]["abstracts"]

    assert await asyncio.gather(one("111"), one("222")) == [["111"], ["222"]]
