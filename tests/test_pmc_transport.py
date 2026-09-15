"""PMC tools through real HTTP requests, version resolution, cache, and JATS parsing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from xml.sax.saxutils import escape

import httpx
import pytest

from deep_life_sci.sources import pmc
from tests.conftest import _install_transport
from tests.test_pmc import _JATS


def listing(objects, token=None):
    entries = "".join(
        f"<Contents><Key>{escape(key)}</Key><Size>{size}</Size></Contents>" for key, size in objects
    )
    tail = (
        (
            f"<IsTruncated>true</IsTruncated><NextContinuationToken>{escape(token)}"
            "</NextContinuationToken>"
        )
        if token
        else "<IsTruncated>false</IsTruncated>"
    )
    return f"<ListBucketResult>{entries}{tail}</ListBucketResult>"


@pytest.fixture
def corpus(monkeypatch):
    bodies = {
        "PMC123.2/PMC123.2.xml": _JATS.encode(),
        "PMC123.2/PMC123.2.json": json.dumps(
            {"pmid": 456, "license_code": "CC BY", "is_retracted": True}
        ).encode(),
        "PMC123.2/fig1.jpg": b"image",
        "PMC123.2/mmc2.xlsx": b"sheet",
    }

    def serve(req):
        if "list-type" in req.url.params:
            assert req.url.params["prefix"].endswith(".")
            if req.url.params["prefix"] != "PMC123.":
                return httpx.Response(200, text=listing([]))
            objects = [(name, len(data)) for name, data in bodies.items()]
            objects += [
                ("PMC123.1/PMC123.1.xml", 10),
                ("PMC123.2/huge.jpg", pmc.SANDBOX_READ_CAP + 1),
            ]
            return httpx.Response(200, text=listing(objects))
        key = req.url.path.lstrip("/")
        return httpx.Response(200, content=bodies[key]) if key in bodies else httpx.Response(404)

    return _install_transport(monkeypatch, serve), bodies


async def test_triage_omits_body_and_table_rows_but_keeps_provenance(corpus):
    transport, _ = corpus
    result = await pmc.pmc_locate.ainvoke({"pmcids": ["123", "pmc123", "999", "invalid"]})
    assert result["invalid"] == ["invalid"]
    assert result["unavailable"] == ["PMC999"]
    assert set(result["available"]) == {"PMC123"}
    paper = result["available"]["PMC123"]
    assert paper["pmid"] == "456"
    assert paper["retracted"] is True
    assert paper["redistributable"] is True
    assert paper["body_chars"] == len("We observed an effect.")
    assert "We observed an effect." not in json.dumps(result)
    assert "Placebo" not in json.dumps(result)
    assert paper["figures"][0]["caption"] == "Survival curves."
    assert not any(".1/" in str(req.url) for req in transport.requests)
    requests_before = len(transport.requests)
    await pmc.pmc_locate.ainvoke({"pmcids": ["123", "999"]})
    assert len(transport.requests) == requests_before  # positive and negative caches


@pytest.mark.parametrize(
    "sections,fallback", [(["results"], False), (["methods"], True), ([], False)]
)
async def test_full_text_selection_and_optional_payloads(corpus, sections, fallback):
    result = await pmc.fetch_full_text.ainvoke(
        {
            "pmcids": ["123"],
            "sections": sections,
            "include_tables": False,
            "include_captions": False,
        }
    )
    paper = result["records"]["PMC123"]
    assert paper["text"] == "## Results\n\nWe observed an effect."
    assert paper["chars"] == len(paper["text"])
    assert paper["sections_returned"] == ["Results"]
    assert paper["fell_back"] is fallback
    result = await pmc.fetch_full_text.ainvoke(
        {
            "pmcids": ["123"],
            "include_tables": True,
            "include_captions": True,
        }
    )
    assert "Placebo" in result["records"]["PMC123"]["text"]
    assert "Survival curves." in result["records"]["PMC123"]["text"]


async def test_malformed_article_is_unavailable_without_losing_other_results(corpus):
    _, bodies = corpus
    bodies["PMC123.2/PMC123.2.xml"] = b"<broken"
    result = await pmc.pmc_locate.ainvoke({"pmcids": ["123", "999"]})
    assert result["available"] == {}
    assert result["unavailable"][1] == "PMC999"
    assert "PMC123 (error:" in result["unavailable"][0]
    assert "parse JATS" in result["unavailable"][0]


async def test_figures_resolve_labels_stage_real_bytes_and_report_skips(corpus):
    backend = SimpleNamespace(aupload_files=AsyncMock(return_value=[SimpleNamespace(error=None)]))
    figures, _ = pmc.make_sandbox_tools(backend)
    result = await figures.ainvoke({"pmcid": "123", "files": ["Figure 1", "f2", "f3", "unknown"]})
    assert result["staged"] == [
        {
            "pmcid": "PMC123",
            "file": "fig1.jpg",
            "path": "/workspace/figures/PMC123/fig1.jpg",
            "bytes": 5,
        }
    ]
    backend.aupload_files.assert_awaited_once_with(
        [("/workspace/figures/PMC123/fig1.jpg", b"image")]
    )
    assert len(result["skipped"]) == 3
    assert "never deposited" in result["skipped"][0]["reason"]
    assert "over the" in result["skipped"][1]["reason"]
    assert "no such figure" in result["skipped"][2]["reason"]
    assert "aW1hZ2U=" not in json.dumps(result)
    assert result["license"] == "CC BY"


async def test_supplementary_upload_failure_is_not_reported_as_staged(corpus):
    backend = SimpleNamespace(aupload_files=AsyncMock(return_value=[SimpleNamespace(error="full")]))
    _, supplementary = pmc.make_sandbox_tools(backend)
    result = await supplementary.ainvoke({"pmcid": "123", "files": ["MMC2.XLSX"]})
    assert result["staged"] == []
    assert "sandbox upload failed: full" in result["skipped"][0]["reason"]
    backend.aupload_files.assert_awaited_once_with(
        [("/workspace/supplementary/PMC123/mmc2.xlsx", b"sheet")]
    )


async def test_s3_pagination_decodes_xml_continuation_tokens(monkeypatch):
    def serve(req):
        if "continuation-token" not in req.url.params:
            return httpx.Response(200, text=listing([("PMC123.1/a&b.xml", 3)], "a&b"))
        assert req.url.params["continuation-token"] == "a&b"
        return httpx.Response(200, text=listing([("PMC123.2/a.xml", 4)]))

    transport = _install_transport(monkeypatch, serve)
    async with httpx.AsyncClient() as client:
        result = await pmc._s3_list(client, "PMC123.")
    assert result == [("PMC123.1/a&b.xml", 3), ("PMC123.2/a.xml", 4)]
    assert len(transport.requests) == 2


@pytest.mark.parametrize("name", ["../escape", ".hidden", "folder/data.csv", "absent"])
async def test_unsafe_or_missing_object_names_never_reach_http(monkeypatch, name):
    transport = _install_transport(monkeypatch, lambda req: pytest.fail("unexpected request"))
    package = {"prefix": "PMC123.2", "objects": {name: 4} if name != "absent" else {}}
    async with httpx.AsyncClient() as client:
        assert await pmc._object_bytes(client, package, name) is None
    assert transport.requests == []


@pytest.mark.parametrize("status", [403, 429, 500])
async def test_s3_http_failures_are_explicit_source_errors(monkeypatch, status):
    _install_transport(monkeypatch, lambda req: httpx.Response(status))
    async with httpx.AsyncClient() as client:
        with pytest.raises(pmc.PMCError, match=f"HTTP {status}"):
            await pmc._s3_list(client, "PMC123.")
        with pytest.raises(pmc.PMCError, match=f"HTTP {status}"):
            await pmc._s3_get(client, "PMC123.1/a.xml")


@pytest.mark.parametrize(
    "body",
    [
        "not XML",
        "<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>",
        "<ListBucketResult><Contents><Key>file</Key></Contents></ListBucketResult>",
    ],
)
async def test_bad_s3_listing_is_not_silently_a_partial_or_empty_corpus(monkeypatch, body):
    _install_transport(monkeypatch, lambda req: httpx.Response(200, text=body))
    async with httpx.AsyncClient() as client:
        with pytest.raises(pmc.PMCError):
            await pmc._s3_list(client, "PMC123.")


async def test_s3_default_namespace_is_supported(monkeypatch):
    body = listing([("PMC123.1/paper.xml", 20)]).replace(
        "<ListBucketResult>", '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
    )
    _install_transport(monkeypatch, lambda req: httpx.Response(200, text=body))
    async with httpx.AsyncClient() as client:
        assert await pmc._s3_list(client, "PMC123.") == [("PMC123.1/paper.xml", 20)]


async def test_repeated_s3_token_does_not_loop_forever(monkeypatch):
    transport = _install_transport(
        monkeypatch, lambda req: httpx.Response(200, text=listing([], "repeat"))
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(pmc.PMCError, match="repeated continuation"):
            await pmc._s3_list(client, "PMC123.")
    assert len(transport.requests) == 2
