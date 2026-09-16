"""Large trial payloads stay out of PTC; indexed evidence is actually readable."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.paths import OUT_DIR, TRIAL_FILES_DIR
from deep_life_sci.sources import ctgov, trial_files
from tests.conftest import json_response


def study(nct_id="NCT00000001", *, large=True):
    return {
        "protocolSection": {"identificationModule": {"nctId": nct_id, "briefTitle": "Trial"}},
        "hasResults": large,
        **({"resultsSection": {
            "outcomeMeasuresModule": {"outcomeMeasures": [{
                "title": "Body weight", "type": "PRIMARY", "timeFrame": "Week 68",
                "groups": [{"id": "OG000", "title": "Treatment"}],
                "denoms": [{"units": "Participants", "counts": [
                    {"groupId": "OG000", "value": "1212"}]}],
                "analyses": [{"paramValue": "-12.44", "ciLowerLimit": "-13.37"}],
            }]},
            "adverseEventsModule": {
                "timeFrame": "Week 75", "frequencyThreshold": "5",
                "eventGroups": [{"id": "EG000", "title": "Treatment",
                                 "otherNumAtRisk": 1306}],
                "otherEvents": [{"term": "Nausea", "organSystem": "GI", "stats": [
                    {"groupId": "EG000", "numAffected": 576, "numEvents": 1067}]}],
            },
            "moreInfoModule": {"limitationsAndCaveats": {"description": "UNREAD " * 5000}},
        }} if large else {}),
    }


@pytest.fixture
def backend():
    files = {}

    async def upload(batch):
        files.update(batch)
        return [SimpleNamespace(path=p, error=None) for p, _ in batch]

    return SimpleNamespace(aupload_files=AsyncMock(side_effect=upload), files=files)


async def test_mixed_batch_stages_only_large_records_and_preserves_evidence(mock_ctgov, backend):
    big, small = study(), study("NCT00000002", large=False)
    transport = mock_ctgov(lambda r: json_response({"studies": [big, small]}))
    tool = trial_files.make_trial_fetch(backend)
    result = await tool.ainvoke({"nct_ids": ["NCT00000001", "NCT00000002", "NCT00000003",
                                           "invalid"], "include": ["results"]})
    assert len(transport.requests) == 1
    assert "ResultsSection" in transport.requests[0].url.params["fields"]
    assert result["missing"] == ["NCT00000003"]
    assert result["invalid"] == ["INVALID"]
    assert result["records"]["NCT00000002"] == ctgov._study_to_record(small)
    manifest = result["records"]["NCT00000001"]
    assert manifest["storage"] == "file"
    assert "UNREAD" not in json.dumps(result)
    assert json.loads(backend.files[manifest["path"]]) == ctgov._study_to_record(big)
    index = json.loads(backend.files[manifest["index_path"]])
    assert len(index) == manifest["section_count"]
    for entry in index:
        data = backend.files[entry["path"]]
        assert len(data) == entry["bytes"]
        assert data.count(b"\n") + 1 == entry["lines"]
        assert entry["path"].startswith(TRIAL_FILES_DIR)
        assert not entry["path"].startswith(OUT_DIR)
    outcome = next(x for x in index if x["section"].startswith("Outcome"))
    value = json.loads(backend.files[outcome["path"]])["data"]
    assert value == big["resultsSection"]["outcomeMeasuresModule"]
    adverse = next(x for x in index if "otherEvents" in x["section"])
    value = json.loads(backend.files[adverse["path"]])["data"]
    assert value == big["resultsSection"]["adverseEventsModule"]


async def test_cache_hit_restages_into_replacement_sandbox(mock_ctgov, backend):
    transport = mock_ctgov(lambda r: json_response({"studies": [study()]}))
    tool = trial_files.make_trial_fetch(backend)
    args = {"nct_ids": ["NCT00000001"], "include": ["results"]}
    first = await tool.ainvoke(args)
    expected = dict(backend.files)
    backend.files.clear()
    second = await tool.ainvoke(args)
    assert second["from_cache"] == ["NCT00000001"]
    assert first["records"] == second["records"]
    assert backend.files == expected
    assert len(transport.requests) == 1
    narrow = await tool.ainvoke({"nct_ids": ["NCT00000001"], "include": []})
    assert "posted_results" not in narrow["records"]["NCT00000001"]
    assert "storage" not in narrow["records"]["NCT00000001"]


@pytest.mark.parametrize("failure", ["partial", "short", "oserror"])
async def test_staging_failure_returns_error_without_dangling_manifest(
    mock_ctgov, backend, failure
):
    mock_ctgov(lambda r: json_response({"studies": [study()]}))
    if failure == "oserror":
        backend.aupload_files.side_effect = OSError("sandbox unavailable")
    else:
        backend.aupload_files.side_effect = lambda batch: (
            [] if failure == "short" else [SimpleNamespace(error="permission_denied")
                                          for _ in batch]
        )
    tool, = with_error_capture([trial_files.make_trial_fetch(backend)])
    result = await tool.ainvoke({"nct_ids": ["NCT00000001"], "include": ["results"]})
    assert set(result) == {"error"}
    assert "stage" in result["error"]


def test_threshold_uses_bytes_and_large_protocols_also_spill():
    record = {"nct_id": "NCT00000001", "detailed_description": "é" * 13000}
    manifest, uploads = trial_files._prepare(record)
    assert manifest["storage"] == "file"
    assert len(uploads) >= 4
    assert any(b"detailed_description" in body for _, body in uploads)
    small = {"nct_id": "NCT00000001"}
    assert trial_files._prepare(small) == (small, [])


def test_different_projections_have_distinct_paths():
    record = ctgov._study_to_record(study())
    first, _ = trial_files._prepare(record)
    second, _ = trial_files._prepare({**record, "brief_summary": "More information"})
    assert first["path"] != second["path"]


async def test_programming_errors_remain_visible(mock_ctgov, backend):
    mock_ctgov(lambda r: json_response({"studies": [study()]}))
    backend.aupload_files.side_effect = TypeError("bug")
    tool, = with_error_capture([trial_files.make_trial_fetch(backend)])
    with pytest.raises(TypeError, match="bug"):
        await tool.ainvoke({"nct_ids": ["NCT00000001"], "include": ["results"]})


async def test_many_sections_upload_in_bounded_batches(mock_ctgov, backend):
    payload = study()
    outcomes = payload["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"]
    outcomes *= 120
    mock_ctgov(lambda r: json_response({"studies": [payload]}))
    result = await trial_files.make_trial_fetch(backend).ainvoke(
        {"nct_ids": ["NCT00000001"], "include": ["results"]}
    )
    manifest = result["records"]["NCT00000001"]
    index = json.loads(backend.files[manifest["index_path"]])
    assert sum(x["section"].startswith("Outcome") for x in index) == 120
    assert all(len(call.args[0]) <= 50 for call in backend.aupload_files.call_args_list)
    assert all(x["path"] in backend.files for x in index)
