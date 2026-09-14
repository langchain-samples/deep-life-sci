"""`middleware/tool_errors.py`: the containment that keeps one bad call from ending a run.

A PTC tool that raises does not fail one call — it fails the whole run. The exception
leaves langchain-quickjs's host-function bridge, propagates out through `eval` and every
`awrap_tool_call` above it, and errors the graph. Measured on 2026-09-03: the model wrote
`AREA[StudyType]INTERVENTAL` for `INTERVENTIONAL`, one character off in one of a dozen
calls, and a 15-minute run that had already fetched its corpus ended with nothing.

Two properties are what make the wrapper correct, and both are asserted here:

* **The payload is `{"error": ...}` and nothing else.** A failed search answering in the
  shape of an empty one reads as "no such trials exist" and gets reported as a finding.
* **A programming error still kills the run loudly.** A `TypeError` in our own code must
  not arrive at the model as a string it will try to route around.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.tools import tool

from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.sources._errors import SourceError
from deep_life_sci.sources.ctgov import ClinicalTrialsError
from deep_life_sci.sources.pmc import PMCError
from deep_life_sci.sources.pubmed import PubMedError


@tool
async def ctgov_search_stub(condition: str) -> dict:
    """A stand-in for a source tool. Raises whatever `condition` names."""
    if condition == "source":
        raise ClinicalTrialsError(
            "ClinicalTrials.gov returned HTTP 400: Unknown enum value: "
            "AREA[StudyType]INTERVENTAL"
        )
    if condition == "transport":
        raise httpx.ConnectError("connection refused")
    if condition == "bug":
        raise TypeError("'NoneType' object is not subscriptable")
    return {"count": 1, "records": [{"nct_id": "NCT03548935"}]}


@pytest.fixture
def wrapped():
    (contained,) = with_error_capture([ctgov_search_stub])
    return contained


class TestErrorCapture:
    async def test_the_success_path_is_untouched(self, wrapped):
        assert await wrapped.ainvoke({"condition": "ok"}) == {
            "count": 1,
            "records": [{"nct_id": "NCT03548935"}],
        }

    async def test_a_source_failure_comes_back_as_a_value(self, wrapped):
        result = await wrapped.ainvoke({"condition": "source"})
        assert "error" in result

    async def test_the_apis_own_message_survives_intact(self, wrapped):
        """The body names the offending token, and that is the entire value here."""
        result = await wrapped.ainvoke({"condition": "source"})
        assert "AREA[StudyType]INTERVENTAL" in result["error"]

    async def test_the_payload_carries_nothing_that_reads_as_an_empty_result(self, wrapped):
        """`count: 0` here reads as 'no such trials exist' and gets reported as a finding."""
        result = await wrapped.ainvoke({"condition": "source"})
        assert set(result) == {"error"}

    async def test_the_error_names_the_tool_that_failed(self, wrapped):
        result = await wrapped.ainvoke({"condition": "source"})
        assert result["error"].startswith("ctgov_search_stub failed:")

    async def test_a_transport_failure_below_the_retry_ladder_is_contained_too(self, wrapped):
        """One unreachable host must not take the fan-out around it with it."""
        result = await wrapped.ainvoke({"condition": "transport"})
        assert set(result) == {"error"}
        assert "ConnectError" in result["error"]

    async def test_a_programming_error_still_kills_the_run_loudly(self, wrapped):
        with pytest.raises(TypeError):
            await wrapped.ainvoke({"condition": "bug"})

    @pytest.mark.parametrize(
        "exc", [PubMedError("a"), PMCError("b"), ClinicalTrialsError("c"), SourceError("d")]
    )
    async def test_every_source_type_is_caught_through_the_shared_base(self, exc):
        """`tool_errors` must not have to import all three and go stale on a fourth."""

        @tool
        async def raiser() -> dict:
            """Raises."""
            raise exc

        (contained,) = with_error_capture([raiser])
        assert set(await contained.ainvoke({})) == {"error"}


class TestWrapperIsTheSameToolInEveryOtherRespect:
    """PTC reads name, description and args schema to build the JS signature block."""

    def test_the_name_is_preserved(self, wrapped):
        assert wrapped.name == ctgov_search_stub.name

    def test_the_description_is_preserved(self, wrapped):
        assert wrapped.description == ctgov_search_stub.description

    def test_the_args_schema_is_preserved(self, wrapped):
        assert wrapped.args == ctgov_search_stub.args

    def test_the_original_is_not_mutated(self, wrapped):
        """A copy, not a subclass and not an in-place patch."""
        assert wrapped is not ctgov_search_stub
        assert ctgov_search_stub.coroutine is not wrapped.coroutine

    def test_every_tool_handed_in_comes_back(self):
        tools = with_error_capture([ctgov_search_stub, ctgov_search_stub])
        assert len(tools) == 2

    def test_a_sync_tool_passes_through_unchanged(self):
        @tool
        def plain(x: int) -> int:
            """A sync tool; we have none, but PTC would call it the same way."""
            return x

        (result,) = with_error_capture([plain])
        assert result is plain
