"""`sources/pubmed.py`, organised around the four traps its module docstring names.

Each of those is a live-API behaviour that returns a *wrong answer rather than an error*,
so each is tested for the wrong answer specifically, not just for "the guard runs":

1. efetch tokenises a malformed PMID into unrelated papers -> `validate_pmids`.
2. esearch silently rewrites a broken query -> `check_field_tags` + `_collect_warnings`.
3. esummary's 500-UID cap answers HTTP 200 with an `error` key -> `_esummary_chunk`.
4. `.//ArticleIdList` reaches a *cited* paper's ids -> `_article_ids`.

The fifth, documented in `_summary_to_record`, is the `pmc` / `pmcid` operand order: the
deposit receipt is truthy, so reading it first short-circuits the `or` and returns null
full-text availability for every paper that has it.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import httpx
import pytest

from deep_life_sci.sources import pubmed
from deep_life_sci.sources.pubmed import (
    MAX_RETMAX,
    PubMedError,
    _article_ids,
    _collect_warnings,
    _common_params,
    _min_interval,
    _parse_article,
    _parse_efetch,
    _scan_cache,
    _summary_to_record,
    _text_of,
    check_field_tags,
    fetch_abstracts,
    normalize_pmcid,
    pubmed_search,
    validate_pmids,
)
from tests.conftest import json_response, text_response, write_cached_abstract

# --------------------------------------------------------------------------------
# Trap 1: a malformed PMID fetches an unrelated paper
# --------------------------------------------------------------------------------


class TestValidatePmids:
    def test_splits_valid_from_invalid(self):
        assert validate_pmids(["123", "abc", "456"]) == (["123", "456"], ["abc"])

    def test_rejects_the_decimal_that_efetch_tokenises(self):
        """'42.9' becomes PMIDs 42 and 9 at the API — two real, unrelated papers."""
        valid, invalid = validate_pmids(["42.9"])
        assert valid == []
        assert invalid == ["42.9"]

    @pytest.mark.parametrize("bad", ["12 34", "12,34", "PMID:12345", "-5", "1e5", "", "  "])
    def test_rejects_anything_that_is_not_all_digits(self, bad: str):
        assert validate_pmids([bad])[0] == []

    def test_strips_surrounding_whitespace_before_judging(self):
        assert validate_pmids(["  123  "]) == (["123"], [])

    def test_preserves_order_and_drops_duplicates(self):
        assert validate_pmids(["3", "1", "3", "2", "1"])[0] == ["3", "1", "2"]

    def test_coerces_integers_the_caller_passed(self):
        assert validate_pmids([123, 456])[0] == ["123", "456"]  # type: ignore[list-item]


class TestNormalizePmcid:
    @pytest.mark.parametrize(
        "raw", ["5904197", "pmc5904197", "PMC5904197", "PMC5904197.1", " pmc5904197 "]
    )
    def test_accepts_every_spelling_of_the_same_id(self, raw: str):
        assert normalize_pmcid(raw) == "PMC5904197"

    def test_none_stays_none(self):
        assert normalize_pmcid(None) is None

    def test_rejects_the_deposit_receipt_rather_than_coercing_it(self):
        """`pmcid` is a receipt, not an id. Coercing it would fetch someone else's paper."""
        receipt = "pmc-id: PMC3030664;manuscript-id: NIHMS262124;"
        assert normalize_pmcid(receipt) is None

    @pytest.mark.parametrize("bad", ["", "abc", "PMC", "PMCabc", "12a3"])
    def test_rejects_anything_without_clean_digits(self, bad: str):
        assert normalize_pmcid(bad) is None


# --------------------------------------------------------------------------------
# Trap 2: a broken query is repaired rather than rejected
# --------------------------------------------------------------------------------


class TestCheckFieldTags:
    def test_flags_a_tag_pubmed_would_silently_drop(self):
        warnings = check_field_tags("cancer[nosuchfield]")
        assert len(warnings) == 1
        assert "nosuchfield" in warnings[0]

    @pytest.mark.parametrize("tag", ["ti", "au", "mesh", "dp", "pt", "tiab", "majr", "si", "aid"])
    def test_accepts_the_documented_short_tags(self, tag: str):
        assert check_field_tags(f"term[{tag}]") == []

    @pytest.mark.parametrize(
        "tag", ["Title", "MeSH Terms", "Publication Date", "Title/Abstract", "All Fields"]
    )
    def test_accepts_the_long_forms_case_insensitively(self, tag: str):
        assert check_field_tags(f"term[{tag}]") == []

    def test_a_date_range_is_checked_by_its_tag_only(self):
        assert check_field_tags("2023:2025[dp]") == []

    @pytest.mark.parametrize("word", ["date", "terms", "of", "type"])
    def test_rejects_the_fragments_a_split_long_name_used_to_admit(self, word: str):
        """An earlier table was built by splitting multi-word names on whitespace."""
        assert check_field_tags(f"term[{word}]") != []

    def test_reports_every_bad_tag_in_a_compound_query(self):
        assert len(check_field_tags("a[nope] AND b[ti] AND c[alsonope]")) == 2

    def test_a_query_with_no_tags_is_clean(self):
        assert check_field_tags("CRISPR AND liver") == []


class TestCollectWarnings:
    def test_reports_a_phrase_missing_from_the_index(self):
        result = {"errorlist": {"phrasesnotfound": ["xyzzy"]}}
        assert any("xyzzy" in w for w in _collect_warnings(result))

    def test_reports_an_unrecognised_field_tag(self):
        result = {"errorlist": {"fieldsnotfound": ["nosuchfield"]}}
        assert any("nosuchfield" in w for w in _collect_warnings(result))

    def test_a_quoted_phrase_fallback_says_the_results_may_be_unrelated(self):
        result = {"warninglist": {"quotedphrasesnotfound": ["base editing"]}}
        (warning,) = _collect_warnings(result)
        assert "unquoted" in warning

    def test_swallows_the_no_items_found_output_message(self):
        """A legitimately empty search is not a query the caller wrote wrong."""
        result = {"warninglist": {"outputmessages": ["No items found."]}}
        assert _collect_warnings(result) == []

    def test_surfaces_any_other_output_message(self):
        result = {"warninglist": {"outputmessages": ["Query rewritten."]}}
        assert _collect_warnings(result) != []

    def test_surfaces_an_explicit_error(self):
        assert any("boom" in w for w in _collect_warnings({"ERROR": "boom"}))

    def test_a_clean_result_warns_about_nothing(self):
        assert _collect_warnings({"count": "10", "idlist": ["1"]}) == []

    def test_null_lists_are_treated_as_absent(self):
        assert _collect_warnings({"errorlist": None, "warninglist": None}) == []


# --------------------------------------------------------------------------------
# Trap 4: `.//ArticleIdList` reaches a cited paper's ids
# --------------------------------------------------------------------------------

_ARTICLE_WITH_REFERENCES = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>29695998</PMID>
   <Article>
    <ArticleTitle>The real paper</ArticleTitle>
    <Abstract><AbstractText>Body.</AbstractText></Abstract>
    <Journal><Title>Real Journal</Title>
     <JournalIssue><PubDate><Year>2018</Year></PubDate></JournalIssue>
    </Journal>
   </Article>
  </MedlineCitation>
  <PubmedData>
   <ArticleIdList>
    <ArticleId IdType="pubmed">29695998</ArticleId>
    <ArticleId IdType="doi">10.1000/real</ArticleId>
    <ArticleId IdType="pmc">PMC5904197</ArticleId>
   </ArticleIdList>
   <ReferenceList>
    <Reference>
     <Citation>Someone else, 2017</Citation>
     <ArticleIdList>
      <ArticleId IdType="pubmed">28000000</ArticleId>
      <ArticleId IdType="doi">10.1000/cited</ArticleId>
      <ArticleId IdType="pmc">PMC5379068</ArticleId>
     </ArticleIdList>
    </Reference>
   </ReferenceList>
  </PubmedData>
 </PubmedArticle>
</PubmedArticleSet>
"""


class TestArticleIds:
    def test_returns_the_articles_own_ids_not_a_references(self):
        art = ET.fromstring(_ARTICLE_WITH_REFERENCES).find(".//PubmedArticle")
        ids = _article_ids(art)
        assert ids["pmc"] == "PMC5904197"
        assert ids["doi"] == "10.1000/real"

    def test_the_parsed_record_carries_the_articles_own_pmcid(self):
        """The measured failure: PMC5379068, a cited paper, reported as this one's."""
        record = _parse_efetch(_ARTICLE_WITH_REFERENCES)["29695998"]
        assert record["pmcid"] == "PMC5904197"
        assert record["doi"] == "10.1000/real"

    def test_falls_back_to_the_book_id_list(self):
        xml = """<PubmedBookArticle><PubmedBookData><ArticleIdList>
          <ArticleId IdType="pubmed">31536</ArticleId></ArticleIdList>
        </PubmedBookData></PubmedBookArticle>"""
        assert _article_ids(ET.fromstring(xml)) == {"pubmed": "31536"}

    def test_an_article_with_no_id_list_yields_nothing(self):
        assert _article_ids(ET.fromstring("<PubmedArticle/>")) == {}


# --------------------------------------------------------------------------------
# Trap 5: the `pmc` / `pmcid` operand order
# --------------------------------------------------------------------------------


class TestSummaryToRecord:
    def _summary(self, **overrides) -> dict:
        record = {
            "title": "A paper",
            "authors": [{"name": "Zhang Q"}, {"name": "Smith J"}],
            "lastauthor": "Smith J",
            "pubdate": "2025 Oct-Dec",
            "source": "Nature",
            "articleids": [{"idtype": "doi", "value": "10.1000/x"}],
        }
        record.update(overrides)
        return record

    def test_maps_the_fields_the_agent_reads(self):
        record = _summary_to_record("123", self._summary())
        assert record["pmid"] == "123"
        assert record["title"] == "A paper"
        assert record["first_author"] == "Zhang Q"
        assert record["last_author"] == "Smith J"
        assert record["journal"] == "Nature"
        assert record["doi"] == "10.1000/x"

    def test_takes_only_the_leading_year_of_a_thirty_format_pubdate(self):
        assert _summary_to_record("1", self._summary(pubdate="2025 Oct-Dec"))["year"] == "2025"

    def test_prefers_the_pmc_id_over_the_deposit_receipt(self):
        """Reading `pmcid` first returned null for 166/166 full-text papers."""
        summary = self._summary(
            articleids=[
                {"idtype": "pmcid", "value": "pmc-id: PMC3030664;manuscript-id: NIHMS262124;"},
                {"idtype": "pmc", "value": "PMC3030664"},
            ]
        )
        assert _summary_to_record("1", summary)["pmcid"] == "PMC3030664"

    def test_a_book_record_falls_back_to_its_book_title(self):
        summary = self._summary(source="", booktitle="StatPearls")
        assert _summary_to_record("1", summary)["journal"] == "StatPearls"

    def test_an_abstract_only_paper_reports_a_null_pmcid(self):
        """None is the legitimate value for 39% of papers, not a parse failure."""
        assert _summary_to_record("1", self._summary())["pmcid"] is None

    def test_missing_fields_become_none_rather_than_empty_strings(self):
        record = _summary_to_record("1", {})
        assert record["title"] is None
        assert record["first_author"] is None
        assert record["year"] is None


# --------------------------------------------------------------------------------
# efetch XML parsing
# --------------------------------------------------------------------------------


class TestTextOf:
    def test_keeps_text_after_inline_markup(self):
        """`node.text` stops at the first child and drops everything after it."""
        node = ET.fromstring("<AbstractText>IC<sub>50</sub> was 4 nM</AbstractText>")
        assert _text_of(node) == "IC50 was 4 nM"

    def test_a_plain_node_is_unchanged(self):
        assert _text_of(ET.fromstring("<t>  hello  </t>")) == "hello"


class TestParseArticle:
    def _article(self, inner: str) -> ET.Element:
        return ET.fromstring(f"<PubmedArticle>{inner}</PubmedArticle>")

    def test_a_structured_abstract_keeps_its_labels(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><Abstract>"
            '<AbstractText Label="BACKGROUND">Why.</AbstractText>'
            '<AbstractText Label="METHODS">How.</AbstractText>'
            "</Abstract></Article></MedlineCitation>"
        )
        record = _parse_article(art)
        assert record["abstract"] == "BACKGROUND: Why.\n\nMETHODS: How."
        assert [s["label"] for s in record["sections"]] == ["BACKGROUND", "METHODS"]

    def test_an_unlabelled_abstract_is_joined_without_labels(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><Abstract>"
            "<AbstractText>One.</AbstractText><AbstractText>Two.</AbstractText>"
            "</Abstract></Article></MedlineCitation>"
        )
        assert _parse_article(art)["abstract"] == "One.\n\nTwo."

    def test_nlm_category_stands_in_for_a_missing_label(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><Abstract>"
            '<AbstractText NlmCategory="RESULTS">Found.</AbstractText>'
            "</Abstract></Article></MedlineCitation>"
        )
        assert _parse_article(art)["sections"][0]["label"] == "RESULTS"

    def test_an_errata_with_no_body_reports_a_null_abstract(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article>"
            "<ArticleTitle>Erratum</ArticleTitle></Article></MedlineCitation>"
        )
        assert _parse_article(art)["abstract"] is None

    def test_a_retracted_publication_type_sets_the_flag(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><PublicationTypeList>"
            "<PublicationType>Retracted Publication</PublicationType>"
            "</PublicationTypeList></Article></MedlineCitation>"
        )
        assert _parse_article(art)["retracted"] is True

    def test_a_retraction_in_comment_also_sets_the_flag(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID>"
            '<CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn">'
            "<PMID>2</PMID></CommentsCorrections></CommentsCorrectionsList>"
            "</MedlineCitation>"
        )
        assert _parse_article(art)["retracted"] is True

    def test_an_ordinary_paper_is_not_retracted(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><PublicationTypeList>"
            "<PublicationType>Journal Article</PublicationType>"
            "</PublicationTypeList></Article></MedlineCitation>"
        )
        assert _parse_article(art)["retracted"] is False

    def test_a_medline_date_supplies_the_year(self):
        art = self._article(
            "<MedlineCitation><PMID>1</PMID><Article><Journal><JournalIssue>"
            "<PubDate><MedlineDate>2024 Spring</MedlineDate></PubDate>"
            "</JournalIssue></Journal></Article></MedlineCitation>"
        )
        assert _parse_article(art)["year"] == "2024"

    def test_an_article_with_no_pmid_is_dropped(self):
        assert _parse_article(self._article("<MedlineCitation/>")) is None


class TestParseEfetch:
    def test_book_records_are_parsed_alongside_journal_articles(self):
        """StatPearls and GeneReviews are PubmedBookArticle and do have abstracts."""
        xml = """<PubmedArticleSet>
          <PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
            <ArticleTitle>Journal paper</ArticleTitle></Article></MedlineCitation>
          </PubmedArticle>
          <PubmedBookArticle><BookDocument><PMID>2</PMID>
            <Book><BookTitle>StatPearls</BookTitle></Book>
            <Abstract><AbstractText>Chapter.</AbstractText></Abstract>
          </BookDocument></PubmedBookArticle>
        </PubmedArticleSet>"""
        records = _parse_efetch(xml)
        assert set(records) == {"1", "2"}
        assert records["2"]["abstract"] == "Chapter."
        assert records["2"]["title"] == "StatPearls"

    def test_an_empty_set_parses_to_nothing(self):
        assert _parse_efetch("<PubmedArticleSet/>") == {}


# --------------------------------------------------------------------------------
# The host-side cache
# --------------------------------------------------------------------------------


class TestScanCache:
    def test_a_current_entry_is_served_from_disk(self, abstract_cache):
        write_cached_abstract(abstract_cache, "111")
        records, from_cache, to_fetch = _scan_cache(["111"])
        assert from_cache == ["111"]
        assert to_fetch == []
        assert records["111"]["title"] == "Paper 111"

    def test_an_absent_entry_is_queued_for_fetch(self, abstract_cache):
        assert _scan_cache(["999"]) == ({}, [], ["999"])

    def test_a_pre_pmc_schema_entry_is_refetched_once(self, abstract_cache):
        """Keyed on the key, not its value — None is legitimate for 39% of papers."""
        (abstract_cache / "111.json").write_text(json.dumps({"pmid": "111", "title": "Old"}))
        records, from_cache, to_fetch = _scan_cache(["111"])
        assert to_fetch == ["111"]
        assert from_cache == []
        assert records == {}

    def test_a_null_pmcid_is_a_hit_rather_than_a_stale_schema(self, abstract_cache):
        write_cached_abstract(abstract_cache, "111", pmcid=None)
        assert _scan_cache(["111"])[1] == ["111"]

    def test_a_corrupt_entry_is_refetched_rather_than_raised(self, abstract_cache):
        (abstract_cache / "111.json").write_text("{not json")
        assert _scan_cache(["111"])[2] == ["111"]

    def test_a_corrupt_entry_keeps_its_mtime_so_the_sweep_collects_it(self, abstract_cache):
        path = abstract_cache / "111.json"
        path.write_text("{not json")
        import os

        os.utime(path, (1_000_000, 1_000_000))
        _scan_cache(["111"])
        assert path.stat().st_mtime == 1_000_000

    def test_a_hit_refreshes_the_entry_so_the_ttl_is_idle(self, abstract_cache):
        """A thread that keeps working keeps its corpus; only a hit refreshes."""
        import os
        import time

        path = write_cached_abstract(abstract_cache, "111")
        aged = time.time() - 60
        os.utime(path, (aged, aged))
        _scan_cache(["111"])
        assert path.stat().st_mtime > aged

    def test_an_expired_entry_is_refetched(self, abstract_cache, monkeypatch):
        import os

        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        path = write_cached_abstract(abstract_cache, "111")
        os.utime(path, (1_000_000, 1_000_000))
        assert _scan_cache(["111"])[2] == ["111"]

    def test_expiry_off_keeps_an_ancient_entry(self, abstract_cache, monkeypatch):
        import os

        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "off")
        path = write_cached_abstract(abstract_cache, "111")
        os.utime(path, (1_000_000, 1_000_000))
        assert _scan_cache(["111"])[1] == ["111"]


# --------------------------------------------------------------------------------
# `_request`: the retry ladder and the POST threshold
# --------------------------------------------------------------------------------


class TestRequest:
    async def test_retries_a_429_and_returns_the_eventual_200(self, mock_ncbi):
        seen = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["n"] += 1
            if seen["n"] < 3:
                return httpx.Response(429)
            return json_response({"ok": True})

        transport = mock_ncbi(handler)
        resp = await pubmed._request("esearch", db="pubmed", term="x")
        assert resp.json() == {"ok": True}
        assert len(transport.requests) == 3

    async def test_gives_up_after_the_configured_attempts(self, mock_ncbi):
        transport = mock_ncbi(lambda r: httpx.Response(503))
        with pytest.raises(PubMedError, match="on all 4 attempts"):
            await pubmed._request("esearch", db="pubmed", term="x")
        assert len(transport.requests) == pubmed.RETRY_ATTEMPTS

    async def test_a_400_surfaces_immediately_without_retrying(self, mock_ncbi):
        """Anything outside RETRY_STATUSES is the caller's problem."""
        transport = mock_ncbi(lambda r: httpx.Response(400))
        with pytest.raises(PubMedError) as excinfo:
            await pubmed._request("esearch", db="pubmed", term="x")
        assert "attempts" not in str(excinfo.value)
        assert len(transport.requests) == 1

    async def test_a_414_names_the_id_list_as_the_cause(self, mock_ncbi):
        mock_ncbi(lambda r: httpx.Response(414))
        with pytest.raises(PubMedError, match="id list too long for GET"):
            await pubmed._request("efetch", db="pubmed", id="1")

    async def test_a_dropped_connection_consumes_an_attempt_like_a_status_does(
        self, mock_ncbi
    ):
        """Keying retries on `resp.status_code` alone let one ReadError kill a fan-out."""
        seen = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["n"] += 1
            if seen["n"] == 1:
                raise httpx.ReadError("connection dropped")
            return json_response({"ok": True})

        transport = mock_ncbi(handler)
        resp = await pubmed._request("esearch", db="pubmed", term="x")
        assert resp.json() == {"ok": True}
        assert len(transport.requests) == 2

    async def test_a_transport_failure_on_every_attempt_becomes_a_pubmed_error(
        self, mock_ncbi
    ):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        mock_ncbi(handler)
        with pytest.raises(PubMedError, match="ConnectError on all 4 attempts"):
            await pubmed._request("esearch", db="pubmed", term="x")

    async def test_a_caller_bug_is_not_retried(self, mock_ncbi):
        """LocalProtocolError is ours; retrying it only delays the traceback."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.LocalProtocolError("bad header")

        transport = mock_ncbi(handler)
        with pytest.raises(httpx.LocalProtocolError):
            await pubmed._request("esearch", db="pubmed", term="x")
        assert len(transport.requests) == 1

    async def test_a_short_id_list_goes_out_as_a_get(self, mock_ncbi):
        transport = mock_ncbi(lambda r: json_response({}))
        await pubmed._request("efetch", db="pubmed", id="1,2,3")
        assert transport.requests[0].method == "GET"

    async def test_a_long_id_list_goes_out_as_a_post(self, mock_ncbi):
        """GET dies at ~3.3k chars of URL with a 414 whose body is not parseable."""
        transport = mock_ncbi(lambda r: json_response({}))
        ids = ",".join(str(n) for n in range(10_000_000, 10_000_400))
        await pubmed._request("efetch", db="pubmed", id=ids)
        assert transport.requests[0].method == "POST"

    async def test_none_valued_parameters_are_dropped(self, mock_ncbi):
        transport = mock_ncbi(lambda r: json_response({}))
        await pubmed._request("esearch", db="pubmed", term="x", mindate=None)
        assert "mindate" not in transport.requests[0].url.params


class TestCommonParams:
    def test_defaults_the_tool_name(self):
        assert _common_params()["tool"] == "deep_life_sci"

    def test_an_empty_tool_var_does_not_drop_the_identifier(self, monkeypatch):
        """`or`, not a get default: a `.env` carrying `NCBI_TOOL=` must not win."""
        monkeypatch.setenv("NCBI_TOOL", "")
        assert _common_params()["tool"] == "deep_life_sci"

    def test_an_absent_email_is_omitted_rather_than_sent_empty(self):
        assert "email" not in _common_params()

    def test_an_api_key_is_forwarded_when_set(self, monkeypatch):
        monkeypatch.setenv("NCBI_API_KEY", "k")
        assert _common_params()["api_key"] == "k"


class TestMinInterval:
    def test_keyless_paces_at_three_per_second(self):
        assert _min_interval() == 0.34

    def test_a_key_raises_the_ceiling_to_ten_per_second(self, monkeypatch):
        monkeypatch.setenv("NCBI_API_KEY", "k")
        assert _min_interval() == 0.11


# --------------------------------------------------------------------------------
# The two tools, end to end over a stub transport
# --------------------------------------------------------------------------------


class TestPubmedSearch:
    async def test_a_probe_returns_the_count_without_fetching_records(self, mock_ncbi):
        transport = mock_ncbi(
            lambda r: json_response(
                {"esearchresult": {"count": "4484", "idlist": ["1", "2"], "querytranslation": "q"}}
            )
        )
        result = await pubmed_search.ainvoke({"term": "cancer", "retmax": 0})
        assert result["count"] == 4484
        assert result["records"] == []
        assert result["returned"] == 0
        assert len(transport.requests) == 1, "a probe must not call esummary"

    async def test_records_keep_esearchs_relevance_ordering(self, mock_ncbi):
        """The esummary dict loses the order the caller asked to sort by."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "esearch" in str(request.url):
                return json_response(
                    {"esearchresult": {"count": "3", "idlist": ["30", "10", "20"]}}
                )
            return json_response(
                {
                    "result": {
                        "uids": ["10", "20", "30"],
                        "10": {"title": "b"},
                        "20": {"title": "c"},
                        "30": {"title": "a"},
                    }
                }
            )

        mock_ncbi(handler)
        result = await pubmed_search.ainvoke({"term": "x", "retmax": 10})
        assert [r["pmid"] for r in result["records"]] == ["30", "10", "20"]

    async def test_a_bad_field_tag_is_warned_about_even_when_the_api_says_nothing(
        self, mock_ncbi
    ):
        """The measured failure: 5.7M hits, and `fieldsnotfound` empty."""
        mock_ncbi(
            lambda r: json_response(
                {"esearchresult": {"count": "5700000", "idlist": [], "errorlist": {}}}
            )
        )
        result = await pubmed_search.ainvoke({"term": "cancer[nosuchfield]", "retmax": 0})
        assert any("nosuchfield" in w for w in result["warnings"])

    async def test_retmax_is_clamped_to_the_documented_ceiling(self, mock_ncbi):
        transport = mock_ncbi(lambda r: json_response({"esearchresult": {"count": "0"}}))
        await pubmed_search.ainvoke({"term": "x", "retmax": 50_000})
        assert transport.requests[0].url.params["retmax"] == str(MAX_RETMAX)

    async def test_a_negative_retmax_becomes_one_rather_than_a_probe(self, mock_ncbi):
        transport = mock_ncbi(lambda r: json_response({"esearchresult": {"count": "0"}}))
        await pubmed_search.ainvoke({"term": "x", "retmax": -5})
        assert transport.requests[0].url.params["retmax"] == "1"

    async def test_a_date_window_adds_the_publication_datetype(self, mock_ncbi):
        transport = mock_ncbi(lambda r: json_response({"esearchresult": {"count": "0"}}))
        await pubmed_search.ainvoke({"term": "x", "retmax": 0, "mindate": "2024"})
        assert transport.requests[0].url.params["datetype"] == "pdat"

    async def test_the_over_cap_error_body_is_raised_rather_than_read_as_zero_hits(
        self, mock_ncbi
    ):
        """esummary answers HTTP 200 with an `error` key and no `result` at all."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "esearch" in str(request.url):
                return json_response({"esearchresult": {"count": "1", "idlist": ["1"]}})
            return json_response({"error": "Too many UIDs in request"})

        mock_ncbi(handler)
        with pytest.raises(PubMedError, match="Too many UIDs"):
            await pubmed_search.ainvoke({"term": "x", "retmax": 10})

    async def test_a_missing_result_block_is_an_error_not_an_empty_page(self, mock_ncbi):
        def handler(request: httpx.Request) -> httpx.Response:
            if "esearch" in str(request.url):
                return json_response({"esearchresult": {"count": "1", "idlist": ["1"]}})
            return json_response({"header": {}})

        mock_ncbi(handler)
        with pytest.raises(PubMedError, match="no `result` block"):
            await pubmed_search.ainvoke({"term": "x", "retmax": 10})

    async def test_a_per_record_error_drops_that_record_only(self, mock_ncbi):
        def handler(request: httpx.Request) -> httpx.Response:
            if "esearch" in str(request.url):
                return json_response({"esearchresult": {"count": "2", "idlist": ["1", "2"]}})
            return json_response(
                {"result": {"uids": ["1", "2"], "1": {"title": "ok"}, "2": {"error": "gone"}}}
            )

        mock_ncbi(handler)
        result = await pubmed_search.ainvoke({"term": "x", "retmax": 10})
        assert [r["pmid"] for r in result["records"]] == ["1"]

    async def test_a_large_id_list_is_chunked_under_the_cap(self, mock_ncbi):
        ids = [str(n) for n in range(1, 451)]

        def handler(request: httpx.Request) -> httpx.Response:
            if "esearch" in str(request.url):
                return json_response({"esearchresult": {"count": "450", "idlist": ids}})
            asked = (request.url.params.get("id") or "").split(",")
            assert len(asked) <= pubmed.SUMMARY_CHUNK
            return json_response(
                {"result": {"uids": asked, **{uid: {"title": uid} for uid in asked}}}
            )

        transport = mock_ncbi(handler)
        result = await pubmed_search.ainvoke({"term": "x", "retmax": 500})
        assert result["returned"] == 450
        assert len(transport.requests) == 1 + 3


class TestFetchAbstracts:
    async def test_cached_pmids_are_served_without_an_http_request(
        self, abstract_cache, mock_ncbi
    ):
        write_cached_abstract(abstract_cache, "111")
        transport = mock_ncbi(lambda r: text_response("<PubmedArticleSet/>"))
        result = await fetch_abstracts.ainvoke({"pmids": ["111"]})
        assert result["from_cache"] == ["111"]
        assert transport.requests == []

    async def test_a_fetch_writes_through_to_the_cache(self, abstract_cache, mock_ncbi):
        mock_ncbi(lambda r: text_response(_ARTICLE_WITH_REFERENCES))
        await fetch_abstracts.ainvoke({"pmids": ["29695998"]})
        assert json.loads((abstract_cache / "29695998.json").read_text())["pmid"] == "29695998"

    async def test_an_invalid_pmid_is_reported_and_never_sent(
        self, abstract_cache, mock_ncbi
    ):
        transport = mock_ncbi(lambda r: text_response("<PubmedArticleSet/>"))
        result = await fetch_abstracts.ainvoke({"pmids": ["42.9"]})
        assert result["invalid"] == ["42.9"]
        assert transport.requests == [], "a malformed id must not reach efetch"

    async def test_a_pmid_pubmed_returns_nothing_for_is_reported_missing(
        self, abstract_cache, mock_ncbi
    ):
        mock_ncbi(lambda r: text_response("<PubmedArticleSet/>"))
        result = await fetch_abstracts.ainvoke({"pmids": ["111"]})
        assert result["missing"] == ["111"]
        assert result["records"] == {}

    async def test_a_mixed_batch_fetches_only_the_uncached_half(
        self, abstract_cache, mock_ncbi
    ):
        write_cached_abstract(abstract_cache, "29695998")
        transport = mock_ncbi(lambda r: text_response(_ARTICLE_WITH_REFERENCES))
        result = await fetch_abstracts.ainvoke({"pmids": ["29695998", "1"]})
        assert result["from_cache"] == ["29695998"]
        assert (transport.requests[0].url.params.get("id") or "") == "1"

    async def test_efetch_is_asked_for_xml_and_never_given_a_retmax(
        self, abstract_cache, mock_ncbi
    ):
        """Text mode renumbers records; retmax truncates silently."""
        transport = mock_ncbi(lambda r: text_response("<PubmedArticleSet/>"))
        await fetch_abstracts.ainvoke({"pmids": ["111"]})
        params = transport.requests[0].url.params
        assert params["retmode"] == "xml"
        assert "retmax" not in params


async def test_post_body_preserves_every_id_and_query_parameter(mock_ncbi):
    from urllib.parse import parse_qs

    ids = ",".join(str(n) for n in range(10_000_000, 10_000_400))
    transport = mock_ncbi(lambda req: json_response({}))
    await pubmed._request("efetch", db="pubmed", id=ids, retmode="xml")
    request = transport.requests[0]
    assert request.method == "POST"
    payload = parse_qs(request.content.decode())
    assert payload["id"] == [ids]
    assert payload["db"] == ["pubmed"]
    assert payload["retmode"] == ["xml"]
    assert "application/x-www-form-urlencoded" in request.headers["content-type"]
