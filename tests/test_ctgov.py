"""`sources/ctgov.py`: the flattening, the bridges to PubMed, and the group-aware cache.

This API rejects bad input with a 400 naming the offending token, so it needs far fewer
guards than `pubmed.py`. What it needs instead is care in two places, and both are here:

* **The reference-type split.** RESULT, DERIVED and BACKGROUND mean genuinely different
  things, and BACKGROUND references are *other people's papers*. A union of all three
  passed to `fetch_abstracts` answers "what has this trial published" with a reading list.
* **The cache's group coverage.** A trial record is mutable in a way a PMID is not, and an
  entry fetched with fewer field groups than the caller now wants must not be served.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from deep_life_sci.sources import ctgov
from deep_life_sci.sources.ctgov import (
    CORE_FIELDS,
    DEFAULT_INCLUDE,
    INCLUDE_FIELDS,
    MAX_RETMAX,
    STUDY_URL,
    ClinicalTrialsError,
    _dig,
    _scan_cache,
    _study_to_record,
    ctgov_fetch,
    ctgov_search,
    validate_nct_ids,
)
from tests.conftest import json_response


@pytest.fixture
def trial_cache(tmp_path, monkeypatch):
    """Point `ctgov.CTGOV_CACHE` at a tmp_path, as `abstract_cache` does for PubMed."""
    cache = tmp_path / "trials"
    cache.mkdir()
    monkeypatch.setattr(ctgov, "CTGOV_CACHE", cache)
    return cache


class TestValidateNctIds:
    def test_accepts_the_canonical_form(self):
        assert validate_nct_ids(["NCT03548935"]) == (["NCT03548935"], [])

    def test_uppercases_a_lowercase_id(self):
        assert validate_nct_ids(["nct03548935"])[0] == ["NCT03548935"]

    @pytest.mark.parametrize(
        "bad", ["NCT123", "NCT035489355", "03548935", "NCTABCDEFGH", "", "NCT 03548935"]
    )
    def test_rejects_anything_off_the_eight_digit_shape(self, bad: str):
        assert validate_nct_ids([bad])[0] == []

    def test_preserves_order_and_drops_duplicates(self):
        ids = ["NCT00000002", "NCT00000001", "NCT00000002"]
        assert validate_nct_ids(ids)[0] == ["NCT00000002", "NCT00000001"]


class TestDig:
    def test_walks_nested_dicts(self):
        assert _dig({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1

    def test_a_missing_key_is_none_rather_than_an_error(self):
        """Projection means most modules are simply absent — the normal case."""
        assert _dig({"a": {}}, "a", "b", "c") is None

    def test_a_non_dict_partway_down_is_none(self):
        assert _dig({"a": "scalar"}, "a", "b") is None

    def test_no_keys_returns_the_object(self):
        assert _dig({"a": 1}) == {"a": 1}


def _study(**sections) -> dict:
    proto = {"identificationModule": {"nctId": "NCT03548935", "briefTitle": "A trial"}}
    proto.update(sections.pop("protocol", {}))
    study = {"protocolSection": proto}
    study.update(sections)
    return study


class TestStudyToRecord:
    def test_flattens_the_core_projection(self):
        record = _study_to_record(
            _study(
                protocol={
                    "identificationModule": {
                        "nctId": "NCT03548935",
                        "briefTitle": "STEP 1",
                        "acronym": "STEP1",
                    },
                    "statusModule": {
                        "overallStatus": "COMPLETED",
                        "startDateStruct": {"date": "2018-06-04"},
                        "completionDateStruct": {"date": "2020-03-31"},
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "Novo Nordisk", "class": "INDUSTRY"}
                    },
                }
            )
        )
        assert record["nct_id"] == "NCT03548935"
        assert record["title"] == "STEP 1"
        assert record["acronym"] == "STEP1"
        assert record["status"] == "COMPLETED"
        assert record["start_date"] == "2018-06-04"
        assert record["completion_date"] == "2020-03-31"
        assert record["lead_sponsor"] == "Novo Nordisk"
        assert record["sponsor_class"] == "INDUSTRY"
        assert record["url"] == STUDY_URL.format("NCT03548935")

    def test_a_study_with_no_nct_id_is_dropped(self):
        assert _study_to_record({"protocolSection": {}}) is None

    def test_phases_stay_a_list_because_a_trial_can_be_registered_as_two(self):
        record = _study_to_record(
            _study(protocol={"designModule": {"phases": ["PHASE2", "PHASE3"]}})
        )
        assert record["phases"] == ["PHASE2", "PHASE3"]

    def test_an_unphased_study_gets_an_empty_list_not_none(self):
        assert _study_to_record(_study())["phases"] == []

    def test_the_enrollment_discriminator_travels_with_the_number(self):
        """Reading an ESTIMATED count as a real one is the commonest registry error."""
        record = _study_to_record(
            _study(
                protocol={
                    "designModule": {"enrollmentInfo": {"count": 1961, "type": "ESTIMATED"}}
                }
            )
        )
        assert record["enrollment"] == 1961
        assert record["enrollment_type"] == "ESTIMATED"

    def test_interventions_without_a_name_are_dropped(self):
        record = _study_to_record(
            _study(
                protocol={
                    "armsInterventionsModule": {
                        "interventions": [{"name": "Semaglutide"}, {"type": "DRUG"}]
                    }
                }
            )
        )
        assert record["interventions"] == ["Semaglutide"]

    def test_optional_modules_are_absent_rather_than_null_when_not_requested(self):
        record = _study_to_record(_study())
        assert "brief_summary" not in record
        assert "eligibility_criteria" not in record
        assert "primary_outcomes" not in record

    def test_the_description_group_adds_its_two_fields(self):
        record = _study_to_record(
            _study(protocol={"descriptionModule": {"briefSummary": "Why."}})
        )
        assert record["brief_summary"] == "Why."
        assert record["detailed_description"] is None

    def test_the_eligibility_group_adds_its_fields(self):
        record = _study_to_record(
            _study(
                protocol={
                    "eligibilityModule": {
                        "eligibilityCriteria": "Adults.",
                        "sex": "ALL",
                        "minimumAge": "18 Years",
                        "stdAges": ["ADULT", "OLDER_ADULT"],
                    }
                }
            )
        )
        assert record["eligibility_criteria"] == "Adults."
        assert record["sex"] == "ALL"
        assert record["min_age"] == "18 Years"
        assert record["std_ages"] == ["ADULT", "OLDER_ADULT"]

    def test_the_design_group_lifts_the_masking_out_of_its_struct(self):
        record = _study_to_record(
            _study(
                protocol={
                    "designModule": {
                        "designInfo": {
                            "allocation": "RANDOMIZED",
                            "maskingInfo": {"masking": "DOUBLE"},
                        }
                    }
                }
            )
        )
        assert record["allocation"] == "RANDOMIZED"
        assert record["masking"] == "DOUBLE"

    def test_arms_are_flattened_to_the_four_fields_that_matter(self):
        record = _study_to_record(
            _study(
                protocol={
                    "armsInterventionsModule": {
                        "armGroups": [
                            {
                                "label": "Semaglutide 2.4 mg",
                                "type": "EXPERIMENTAL",
                                "description": "Weekly.",
                                "interventionNames": ["Drug: Semaglutide"],
                            }
                        ]
                    }
                }
            )
        )
        assert record["arms"] == [
            {
                "label": "Semaglutide 2.4 mg",
                "type": "EXPERIMENTAL",
                "description": "Weekly.",
                "interventions": ["Drug: Semaglutide"],
            }
        ]

    def test_countries_are_deduplicated_and_sorted(self):
        record = _study_to_record(
            _study(
                protocol={
                    "contactsLocationsModule": {
                        "locations": [
                            {"country": "United States"},
                            {"country": "Denmark"},
                            {"country": "United States"},
                            {"city": "Nowhere"},
                        ]
                    }
                }
            )
        )
        assert record["countries"] == ["Denmark", "United States"]

    def test_mesh_descriptors_come_from_the_derived_section(self):
        record = _study_to_record(
            _study(
                derivedSection={
                    "conditionBrowseModule": {"meshes": [{"id": "D009765", "term": "Obesity"}]},
                    "interventionBrowseModule": {"meshes": [{"term": "Semaglutide"}]},
                }
            )
        )
        assert record["condition_mesh"][0]["term"] == "Obesity"
        assert record["intervention_mesh"][0]["term"] == "Semaglutide"


class TestReferenceBridges:
    """The RESULT / DERIVED / BACKGROUND split, which is the bridge to `fetch_abstracts`."""

    @pytest.fixture
    def record(self) -> dict:
        return _study_to_record(
            _study(
                protocol={
                    "referencesModule": {
                        "references": [
                            {"pmid": "33567185", "type": "RESULT"},
                            {"pmid": "34000000", "type": "DERIVED"},
                            {"pmid": "20000000", "type": "BACKGROUND"},
                            {"pmid": "21000000", "type": "BACKGROUND"},
                            {"citation": "No pmid at all", "type": "DERIVED"},
                        ]
                    }
                }
            )
        )

    def test_result_pmids_are_the_sponsors_own_designation_only(self, record):
        assert record["result_pmids"] == ["33567185"]

    def test_trial_pmids_union_result_and_derived(self, record):
        """DERIVED is NLM's automatic back-link and is where the coverage actually is."""
        assert record["trial_pmids"] == ["33567185", "34000000"]

    def test_background_is_kept_out_of_trial_pmids(self, record):
        """These are other people's papers — including them answers with a reading list."""
        assert record["background_pmids"] == ["20000000", "21000000"]
        assert "20000000" not in record["trial_pmids"]

    def test_a_reference_with_no_pmid_contributes_nothing(self, record):
        assert all(p for p in record["trial_pmids"])

    def test_trial_pmids_are_deduplicated(self):
        record = _study_to_record(
            _study(
                protocol={
                    "referencesModule": {
                        "references": [
                            {"pmid": "33567185", "type": "RESULT"},
                            {"pmid": "33567185", "type": "DERIVED"},
                        ]
                    }
                }
            )
        )
        assert record["trial_pmids"] == ["33567185"]

    def test_the_raw_reference_list_is_kept_alongside_the_splits(self, record):
        assert len(record["references"]) == 5


class TestScanCache:
    def test_an_entry_fetched_with_the_same_groups_is_a_hit(self, trial_cache):
        (trial_cache / "NCT03548935.json").write_text(
            json.dumps({"groups": ["description"], "record": {"nct_id": "NCT03548935"}})
        )
        records, from_cache, to_fetch = _scan_cache(["NCT03548935"], frozenset({"description"}))
        assert from_cache == ["NCT03548935"]
        assert to_fetch == []
        assert records["NCT03548935"]["nct_id"] == "NCT03548935"

    def test_an_entry_fetched_with_more_groups_still_covers_a_narrower_ask(self, trial_cache):
        (trial_cache / "NCT03548935.json").write_text(
            json.dumps(
                {"groups": ["description", "outcomes"], "record": {"nct_id": "NCT03548935"}}
            )
        )
        assert _scan_cache(["NCT03548935"], frozenset({"description"}))[1] == ["NCT03548935"]

    def test_an_entry_fetched_with_fewer_groups_is_refetched(self, trial_cache):
        """Serving it would silently answer an `outcomes` question without outcomes."""
        (trial_cache / "NCT03548935.json").write_text(
            json.dumps({"groups": ["description"], "record": {"nct_id": "NCT03548935"}})
        )
        assert _scan_cache(["NCT03548935"], frozenset({"description", "outcomes"}))[2] == [
            "NCT03548935"
        ]

    def test_a_pre_schema_entry_without_groups_is_refetched(self, trial_cache):
        (trial_cache / "NCT03548935.json").write_text(json.dumps({"nct_id": "NCT03548935"}))
        assert _scan_cache(["NCT03548935"], frozenset({"description"}))[2] == ["NCT03548935"]

    def test_a_corrupt_entry_is_refetched_rather_than_raised(self, trial_cache):
        (trial_cache / "NCT03548935.json").write_text("{not json")
        assert _scan_cache(["NCT03548935"], frozenset())[2] == ["NCT03548935"]


class TestRequest:
    async def test_retries_a_429_without_reading_a_retry_after_header(self, mock_ctgov):
        """This API sends neither Retry-After nor X-RateLimit-*; our backoff is all there is."""
        seen = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["n"] += 1
            return httpx.Response(429) if seen["n"] < 3 else json_response({"studies": []})

        transport = mock_ctgov(handler)
        await ctgov._request("/studies", **{"query.cond": "obesity"})
        assert len(transport.requests) == 3

    async def test_a_400_body_is_surfaced_verbatim(self, mock_ctgov):
        """The body names the offending token, which is the whole value of this API."""
        mock_ctgov(
            lambda r: httpx.Response(
                400, text="Unknown enum value: AREA[StudyType]INTERVENTAL"
            )
        )
        with pytest.raises(ClinicalTrialsError, match="INTERVENTAL"):
            await ctgov._request("/studies", **{"filter.advanced": "x"})

    async def test_a_400_is_not_retried(self, mock_ctgov):
        transport = mock_ctgov(lambda r: httpx.Response(400, text="nope"))
        with pytest.raises(ClinicalTrialsError):
            await ctgov._request("/studies")
        assert len(transport.requests) == 1

    async def test_empty_string_parameters_are_dropped(self, mock_ctgov):
        transport = mock_ctgov(lambda r: json_response({}))
        await ctgov._request("/studies", **{"query.cond": "obesity", "query.term": ""})
        assert "query.term" not in transport.requests[0].url.params


class TestCtgovSearch:
    async def test_an_unfiltered_search_is_refused_before_a_request_goes_out(
        self, mock_ctgov
    ):
        """The API would happily return all ~600,000 registered studies."""
        transport = mock_ctgov(lambda r: json_response({}))
        with pytest.raises(ClinicalTrialsError, match="600,000"):
            await ctgov_search.ainvoke({})
        assert transport.requests == []

    async def test_a_probe_returns_the_count_and_no_records(self, mock_ctgov):
        mock_ctgov(
            lambda r: json_response({"totalCount": 314, "studies": [_study()]})
        )
        result = await ctgov_search.ainvoke({"condition": "obesity", "retmax": 0})
        assert result["count"] == 314
        assert result["records"] == []
        assert result["returned"] == 0

    async def test_the_core_projection_is_requested_on_every_call(self, mock_ctgov):
        """Server-side, so the caller cannot accidentally receive a whole record."""
        transport = mock_ctgov(lambda r: json_response({"totalCount": 0, "studies": []}))
        await ctgov_search.ainvoke({"condition": "obesity", "retmax": 0})
        assert transport.requests[0].url.params["fields"] == ",".join(CORE_FIELDS)

    async def test_statuses_are_ored_into_one_filter(self, mock_ctgov):
        transport = mock_ctgov(lambda r: json_response({"totalCount": 0, "studies": []}))
        await ctgov_search.ainvoke(
            {"condition": "obesity", "status": ["COMPLETED", "TERMINATED"], "retmax": 0}
        )
        assert (
            transport.requests[0].url.params["filter.overallStatus"]
            == "COMPLETED|TERMINATED"
        )

    async def test_an_over_cap_retmax_is_reduced_and_warned_about(self, mock_ctgov):
        mock_ctgov(lambda r: json_response({"totalCount": 9999, "studies": []}))
        result = await ctgov_search.ainvoke({"condition": "obesity", "retmax": 50_000})
        assert any(str(MAX_RETMAX) in w for w in result["warnings"])
        assert result["count"] == 9999, "count is still the true total"

    async def test_current_date_is_returned_because_js_has_no_clock(self, mock_ctgov):
        """The interpreter's `Date.now()` is stubbed."""
        mock_ctgov(lambda r: json_response({"totalCount": 0, "studies": []}))
        result = await ctgov_search.ainvoke({"condition": "obesity", "retmax": 0})
        assert result["current_date"] == date.today().isoformat()

    async def test_query_sent_echoes_the_parameters_minus_the_field_list(self, mock_ctgov):
        mock_ctgov(lambda r: json_response({"totalCount": 0, "studies": []}))
        result = await ctgov_search.ainvoke({"condition": "obesity", "retmax": 0})
        assert result["query_sent"]["query.cond"] == "obesity"
        assert "fields" not in result["query_sent"]

    async def test_paging_follows_the_token_until_the_limit(self, mock_ctgov):
        pages = [
            {"totalCount": 3, "studies": [_study(), _study()], "nextPageToken": "t2"},
            {"studies": [_study()]},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return json_response(pages[min(len(transport.requests) - 1, 1)])

        transport = mock_ctgov(handler)
        result = await ctgov_search.ainvoke({"condition": "obesity", "retmax": 3})
        assert len(transport.requests) == 2
        assert result["count"] == 3

    async def test_count_total_is_asked_for_on_the_first_page_only(self, mock_ctgov):
        pages = [
            {"totalCount": 2, "studies": [_study()], "nextPageToken": "t2"},
            {"studies": [_study()]},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return json_response(pages[min(len(transport.requests) - 1, 1)])

        transport = mock_ctgov(handler)
        await ctgov_search.ainvoke({"condition": "obesity", "retmax": 2})
        assert "countTotal" in transport.requests[0].url.params
        assert "countTotal" not in transport.requests[1].url.params
        assert transport.requests[1].url.params["pageToken"] == "t2"

    async def test_a_token_with_an_empty_page_stops_rather_than_spinning(self, mock_ctgov):
        transport = mock_ctgov(
            lambda r: json_response({"totalCount": 99, "studies": [], "nextPageToken": "t"})
        )
        await ctgov_search.ainvoke({"condition": "obesity", "retmax": 500})
        assert len(transport.requests) == 1


class TestCtgovFetch:
    async def test_the_default_include_is_the_fan_out_payload(self, trial_cache, mock_ctgov):
        transport = mock_ctgov(lambda r: json_response({"studies": []}))
        await ctgov_fetch.ainvoke({"nct_ids": ["NCT03548935"]})
        fields = transport.requests[0].url.params["fields"].split(",")
        for group in DEFAULT_INCLUDE:
            assert set(INCLUDE_FIELDS[group]).issubset(fields)

    async def test_an_unknown_include_group_is_refused_before_a_request(
        self, trial_cache, mock_ctgov
    ):
        transport = mock_ctgov(lambda r: json_response({"studies": []}))
        with pytest.raises(ClinicalTrialsError, match="unknown include group"):
            await ctgov_fetch.ainvoke({"nct_ids": ["NCT03548935"], "include": ["nope"]})
        assert transport.requests == []

    async def test_an_explicit_empty_include_asks_for_the_core_fields_only(
        self, trial_cache, mock_ctgov
    ):
        transport = mock_ctgov(lambda r: json_response({"studies": []}))
        await ctgov_fetch.ainvoke({"nct_ids": ["NCT03548935"], "include": []})
        assert transport.requests[0].url.params["fields"] == ",".join(CORE_FIELDS)

    async def test_an_invalid_id_is_reported_and_never_sent(self, trial_cache, mock_ctgov):
        transport = mock_ctgov(lambda r: json_response({"studies": []}))
        result = await ctgov_fetch.ainvoke({"nct_ids": ["NCT123"]})
        assert result["invalid"] == ["NCT123"]
        assert transport.requests == []

    async def test_missing_is_a_diff_rather_than_what_the_api_reported(
        self, trial_cache, mock_ctgov
    ):
        """`filter.ids` drops unknown ids and collapses duplicates without saying so."""
        mock_ctgov(lambda r: json_response({"studies": [_study()]}))
        result = await ctgov_fetch.ainvoke(
            {"nct_ids": ["NCT03548935", "NCT00000001"], "include": []}
        )
        assert result["missing"] == ["NCT00000001"]

    async def test_a_fetch_writes_through_with_the_groups_it_used(
        self, trial_cache, mock_ctgov
    ):
        mock_ctgov(lambda r: json_response({"studies": [_study()]}))
        await ctgov_fetch.ainvoke({"nct_ids": ["NCT03548935"], "include": ["description"]})
        entry = json.loads((trial_cache / "NCT03548935.json").read_text())
        assert entry["groups"] == ["description"]
        assert entry["record"]["nct_id"] == "NCT03548935"

    async def test_a_covering_cache_entry_costs_no_request(self, trial_cache, mock_ctgov):
        (trial_cache / "NCT03548935.json").write_text(
            json.dumps({"groups": ["description"], "record": {"nct_id": "NCT03548935"}})
        )
        transport = mock_ctgov(lambda r: json_response({"studies": []}))
        result = await ctgov_fetch.ainvoke(
            {"nct_ids": ["NCT03548935"], "include": ["description"]}
        )
        assert result["from_cache"] == ["NCT03548935"]
        assert transport.requests == []

    async def test_a_long_id_list_is_chunked(self, trial_cache, mock_ctgov):
        ids = [f"NCT{n:08d}" for n in range(1, 451)]

        def handler(request: httpx.Request) -> httpx.Response:
            asked = request.url.params["filter.ids"].split(",")
            assert len(asked) <= ctgov.ID_CHUNK
            return json_response({"studies": []})

        transport = mock_ctgov(handler)
        await ctgov_fetch.ainvoke({"nct_ids": ids, "include": []})
        assert len(transport.requests) == 3
