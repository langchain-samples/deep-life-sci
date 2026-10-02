"""Integration coverage for provenance across the real QuickJS tool bridge."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deep_life_sci import agent, runner
from deep_life_sci.sandbox import ResilientSandbox


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


_SEARCH_CASES = {
    "cancer evidence": ("123", "Cancer evidence paper"),
    "alpha evidence": ("111", "Alpha evidence paper"),
    "beta evidence": ("222", "Beta evidence paper"),
}


def _paper_for_pmid(pmid: str) -> tuple[str, str]:
    for query, (candidate, title) in _SEARCH_CASES.items():
        if candidate == pmid:
            return query, title
    raise AssertionError(f"unexpected PMID {pmid}")


def _pubmed_handler(request: httpx.Request) -> httpx.Response:
    """Small E-utilities surface shared by the real-bridge runtime tests."""
    if request.url.path.endswith("/esearch.fcgi"):
        query = request.url.params["term"]
        pmid, _ = _SEARCH_CASES[query]
        return httpx.Response(
            200,
            json={
                "esearchresult": {
                    "count": "1",
                    "idlist": [pmid],
                    "querytranslation": query,
                }
            },
        )

    if request.url.path.endswith("/esummary.fcgi"):
        pmid = request.url.params["id"]
        _, title = _paper_for_pmid(pmid)
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": [pmid],
                    pmid: {
                        "uid": pmid,
                        "title": title,
                        "pubdate": "2025",
                        "source": "Test Journal",
                        "authors": [{"name": "A Author"}],
                        "lastauthor": "B Author",
                        "articleids": [{"idtype": "pubmed", "value": pmid}],
                    },
                }
            },
        )

    if request.url.path.endswith("/efetch.fcgi"):
        pmid = request.url.params["id"]
        _, title = _paper_for_pmid(pmid)
        return httpx.Response(
            200,
            text=f"""<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <ArticleTitle>{title}</ArticleTitle>
        <Abstract><AbstractText>Evidence payload for {pmid}.</AbstractText></Abstract>
        <Journal>
          <JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue>
          <Title>Test Journal</Title>
        </Journal>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">{pmid}</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>""",
        )

    raise AssertionError(f"unexpected E-utilities request: {request.url}")


def _backend(monkeypatch) -> ResilientSandbox:
    backend = ResilientSandbox(SimpleNamespace())
    monkeypatch.setattr(
        backend, "aexecute", AsyncMock(return_value=SimpleNamespace(output=""))
    )
    return backend


async def test_runner_trace_matches_search_fetch_and_final_citation(monkeypatch, mock_ncbi):
    """A retrieved PMID must be distinguishable from a search hit at the runner seam."""
    model = ScriptedModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "eval",
                        "id": "search-and-fetch",
                        "args": {
                            "code": (
                                'const s = await tools.pubmedSearch('
                                '{term: "cancer evidence", retmax: 1}); '
                                "const pmid = s.records[0].pmid; "
                                "const f = await tools.fetchAbstracts({pmids: [pmid]}); "
                                'console.log("PMID=" + pmid + " ABSTRACT=" + '
                                "f.records[pmid].abstract);"
                            ),
                        },
                    }
                ],
            ),
            AIMessage("The evidence is reported in PMID 123."),
        ]
    )
    monkeypatch.setattr(agent, "root_model", lambda: model)
    monkeypatch.setattr(agent, "subagent_model", lambda: model)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    transport = mock_ncbi(_pubmed_handler)

    result = await runner.run_once(
        "Find and read the cancer evidence paper",
        backend=_backend(monkeypatch),
    )

    assert result.answer == "The evidence is reported in PMID 123."
    assert [request.url.path.rsplit("/", 1)[-1] for request in transport.requests] == [
        "esearch.fcgi",
        "esummary.fcgi",
        "efetch.fcgi",
    ]
    assert result.as_dict()["source_trace"] == result.source_trace

    trace = result.source_trace
    assert trace["pmids"] == ["123"]
    assert len(trace["events"]) == 2

    search, fetch = trace["events"]
    assert search["tool"] == "pubmed_search"
    assert search["operation"] == "search"
    assert search["query"] == "cancer evidence"
    assert search["returned_ids"] == ["123"]
    assert search["success"] is True

    assert fetch["tool"] == "fetch_abstracts"
    assert fetch["operation"] == "fetch"
    assert fetch["requested_ids"] == ["123"]
    assert fetch["returned_ids"] == ["123"]
    assert fetch["success"] is True

    # The answer cites exactly the paper whose source content this run retrieved, while
    # the scientific payload itself stays out of the compact provenance record.
    assert "123" in result.answer
    assert "123" in trace["pmids"]
    assert "Evidence payload for 123" not in repr(trace)


async def test_concurrent_runner_traces_do_not_cross_contaminate_quickjs(
    monkeypatch, mock_ncbi
):
    """Two real PTC bridges in flight must retain independent ContextVar traces."""
    models = iter(
        [
            ScriptedModel(
                responses=[
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "name": "eval",
                                "id": "alpha-search",
                                "args": {
                                    "code": (
                                        'const s = await tools.pubmedSearch('
                                        '{term: "alpha evidence", retmax: 1}); '
                                        'console.log("PMID=" + s.records[0].pmid);'
                                    ),
                                },
                            }
                        ],
                    ),
                    AIMessage("Alpha search returned PMID 111."),
                ]
            ),
            ScriptedModel(
                responses=[
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "name": "eval",
                                "id": "beta-search",
                                "args": {
                                    "code": (
                                        'const s = await tools.pubmedSearch('
                                        '{term: "beta evidence", retmax: 1}); '
                                        'console.log("PMID=" + s.records[0].pmid);'
                                    ),
                                },
                            }
                        ],
                    ),
                    AIMessage("Beta search returned PMID 222."),
                ]
            ),
        ]
    )
    dummy_subagent = ScriptedModel(responses=[AIMessage("unused")])
    monkeypatch.setattr(agent, "root_model", lambda: next(models))
    monkeypatch.setattr(agent, "subagent_model", lambda: dummy_subagent)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    transport = mock_ncbi(_pubmed_handler)

    # Force a scheduling point inside each source request. Without this, a fast mock
    # transport could let one run finish before the other reaches the PTC bridge and the
    # test would only prove sequential isolation again.
    from deep_life_sci.sources import cache_io, pubmed

    # Earlier cache tests intentionally exercise sweep contention under their own pytest
    # event loops. asyncio.Lock retains the loop it contended on, so give this independent
    # runner-concurrency test a fresh sweep lock for the current loop. The cache lock is
    # orthogonal to the ContextVar behavior under test.
    monkeypatch.setattr(cache_io, "_sweep_lock", asyncio.Lock())

    async def yield_once() -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(pubmed._throttle, "wait", yield_once)

    first, second = await asyncio.gather(
        runner.run_once("Run the first independent search", backend=_backend(monkeypatch)),
        runner.run_once("Run the second independent search", backend=_backend(monkeypatch)),
    )

    by_pmid = {}
    for result in (first, second):
        assert len(result.source_trace["events"]) == 1
        event = result.source_trace["events"][0]
        assert event["tool"] == "pubmed_search"
        assert event["operation"] == "search"
        assert event["success"] is True
        assert len(event["returned_ids"]) == 1
        by_pmid[event["returned_ids"][0]] = (result, event)

    assert set(by_pmid) == {"111", "222"}

    alpha_result, alpha = by_pmid["111"]
    beta_result, beta = by_pmid["222"]
    assert alpha["query"] == "alpha evidence"
    assert beta["query"] == "beta evidence"
    assert alpha_result.answer == "Alpha search returned PMID 111."
    assert beta_result.answer == "Beta search returned PMID 222."

    # Search hits stay out of the fetched-PMID summary, and neither run may contain the
    # other run's query or identifier.
    assert alpha_result.source_trace["pmids"] == []
    assert beta_result.source_trace["pmids"] == []
    assert "222" not in repr(alpha_result.source_trace)
    assert "beta evidence" not in repr(alpha_result.source_trace)
    assert "111" not in repr(beta_result.source_trace)
    assert "alpha evidence" not in repr(beta_result.source_trace)

    paths = [request.url.path.rsplit("/", 1)[-1] for request in transport.requests]
    assert paths.count("esearch.fcgi") == 2
    assert paths.count("esummary.fcgi") == 2
