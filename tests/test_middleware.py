"""The middleware that shapes what the user and the model see: progress, artifacts, cadence.

None of these run a graph. Each is tested at the seam it actually owns — a tool wrapper, a
pure parse of a shell command's output, a `ModelRequest` transform — because that is where
the behaviour lives and because a full agent needs a sandbox and a gateway key.
"""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deep_life_sci.middleware.artifacts import (
    EXTRA_MIME,
    IMAGE_SUFFIXES,
    MAX_INLINE_BYTES,
    TABLE_SUFFIXES,
    WRITER_TOOLS,
    _component_for,
    _list_command,
    _mime_for,
    _parse_listing,
)
from deep_life_sci.middleware.cadence import (
    QUIET_SECONDS,
    UpdateCadence,
    _speaks_to_user,
)
from deep_life_sci.middleware.progress import (
    STARTED,
    _arguments,
    _count,
    _trim,
    with_progress,
)
from deep_life_sci.paths import OUT_DIR

# --------------------------------------------------------------------------------
# progress.py — the run's only visible output while it works
# --------------------------------------------------------------------------------


class TestTrim:
    def test_a_short_value_is_unchanged(self):
        assert _trim("CRISPR AND liver") == "CRISPR AND liver"

    def test_whitespace_is_collapsed_so_a_multiline_query_stays_one_line(self):
        assert _trim("a\n  b\tc") == "a b c"

    def test_a_long_value_is_elided_to_the_limit(self):
        assert len(_trim("x" * 200)) == 70
        assert _trim("x" * 200).endswith("…")

    def test_none_becomes_an_empty_string_rather_than_the_word_none(self):
        assert _trim(None) == ""


class TestCount:
    @pytest.mark.parametrize("value", [["a", "b"], ("a", "b"), {"a": 1, "b": 2}])
    def test_counts_a_sized_collection(self, value):
        assert _count(value) == 2

    @pytest.mark.parametrize("value", [None, "abc", 7])
    def test_anything_else_counts_as_zero(self, value):
        assert _count(value) == 0


class TestStartedLabels:
    def test_a_search_names_what_is_being_searched_for(self):
        assert STARTED["pubmed_search"]({"term": "CRISPR"}) == "Searching PubMed for CRISPR"

    def test_a_batch_fetch_names_the_size_of_the_batch(self):
        assert STARTED["fetch_abstracts"]({"pmids": ["1", "2", "3"]}) == "Fetching 3 abstracts"

    def test_a_trial_search_falls_back_through_its_argument_names(self):
        assert "obesity" in STARTED["ctgov_search"]({"condition": "obesity"})
        assert "semaglutide" in STARTED["ctgov_search"]({"intervention": "semaglutide"})
        assert "trials" in STARTED["ctgov_search"]({})

    @pytest.mark.parametrize("name", sorted(STARTED))
    def test_every_label_survives_being_given_nothing(self, name: str):
        """A label is never worth failing a fetch over."""
        assert isinstance(STARTED[name]({}), str)


class TestArguments:
    def test_reads_keyword_arguments(self):
        async def fn(term: str, retmax: int = 50) -> None: ...

        assert _arguments(fn, (), {"term": "x"}) == {"term": "x"}

    def test_reads_positional_arguments_by_name(self):
        """The tool machinery chooses how to pass them; the label must not care."""

        async def fn(term: str, retmax: int = 50) -> None: ...

        assert _arguments(fn, ("x", 10), {}) == {"term": "x", "retmax": 10}

    def test_an_unbindable_call_degrades_to_the_keywords(self):
        async def fn(term: str) -> None: ...

        assert _arguments(fn, (1, 2, 3), {"term": "x"}) == {"term": "x"}


class TestWithProgress:
    @pytest.fixture
    def emitted(self, monkeypatch) -> list[str]:
        from deep_life_sci.middleware import progress

        lines: list[str] = []
        monkeypatch.setattr(progress, "_emit", lines.append)
        return lines

    @pytest.fixture
    def narrated(self):
        @tool
        async def pubmed_search(term: str, retmax: int = 50) -> dict:
            """A stand-in for the real search tool."""
            return {"count": 1}

        (wrapped,) = with_progress([pubmed_search])
        return wrapped

    async def test_narrates_the_call_before_running_it(self, narrated, emitted):
        await narrated.ainvoke({"term": "CRISPR"})
        assert emitted == ["Searching PubMed for CRISPR"]

    async def test_the_return_value_is_untouched(self, narrated, emitted):
        assert await narrated.ainvoke({"term": "x"}) == {"count": 1}

    async def test_an_unlabelled_tool_falls_back_to_its_own_name(self, emitted):
        @tool
        async def something_new(x: int) -> int:
            """Not in the label table."""
            return x

        (wrapped,) = with_progress([something_new])
        await wrapped.ainvoke({"x": 1})
        assert emitted == ["Running something_new"]

    async def test_a_label_that_raises_does_not_fail_the_call(self, monkeypatch, emitted):
        from deep_life_sci.middleware import progress

        monkeypatch.setitem(
            progress.STARTED, "pubmed_search", lambda a: 1 / 0
        )

        @tool
        async def pubmed_search(term: str) -> dict:
            """A stand-in."""
            return {"ok": True}

        (wrapped,) = with_progress([pubmed_search])
        assert await wrapped.ainvoke({"term": "x"}) == {"ok": True}
        assert emitted == ["Running pubmed_search"]

    async def test_an_emit_failure_never_breaks_the_run_it_describes(self, narrated):
        """No writer is configured under `uv run agent` or in evals."""
        assert await narrated.ainvoke({"term": "x"}) == {"count": 1}

    def test_the_wrapper_is_the_same_tool_in_every_other_respect(self, narrated):
        assert narrated.name == "pubmed_search"
        assert "stand-in" in narrated.description
        assert set(narrated.args) == {"term", "retmax"}


# --------------------------------------------------------------------------------
# artifacts.py — the /workspace/out sweep
# --------------------------------------------------------------------------------


class TestListCommand:
    def test_walks_the_directory_it_is_given(self):
        assert repr(OUT_DIR) in _list_command(OUT_DIR)

    def test_emits_the_three_fields_a_glob_would_not_carry(self):
        """`size` enforces the inline cap; `mtime` tells a regenerated chart from a stale one."""
        command = _list_command(OUT_DIR)
        for field in ("path", "size", "mtime"):
            assert f'"{field}"' in command

    def test_does_the_walk_in_python_so_awkward_filenames_survive(self):
        assert "os.walk" in _list_command(OUT_DIR)


class TestParseListing:
    def test_reads_one_json_object_per_line(self):
        output = '{"path": "/workspace/out/a.png", "size": 10, "mtime": 1.0}\n'
        assert _parse_listing(output) == [
            {"path": "/workspace/out/a.png", "size": 10, "mtime": 1.0}
        ]

    def test_skips_the_trailing_command_status_line(self):
        """`execute` returns stdout plus `[Command succeeded ...]`."""
        output = '{"path": "/workspace/out/a.png", "size": 1, "mtime": 1.0}\n[Command succeeded]'
        assert len(_parse_listing(output)) == 1

    def test_a_missing_out_directory_produces_no_files_rather_than_an_error(self):
        assert _parse_listing("") == []
        assert _parse_listing(None) == []

    def test_a_malformed_json_line_is_skipped(self):
        assert _parse_listing('{"path": broken}\n{"path": "/a", "size": 1}') == [
            {"path": "/a", "size": 1}
        ]


class TestComponentFor:
    @pytest.mark.parametrize("suffix", sorted(IMAGE_SUFFIXES))
    def test_an_image_renders_as_a_chart(self, suffix: str):
        assert _component_for(suffix) == "chart"

    @pytest.mark.parametrize("suffix", sorted(TABLE_SUFFIXES))
    def test_a_tabular_file_renders_as_a_table(self, suffix: str):
        assert _component_for(suffix) == "table"

    @pytest.mark.parametrize("suffix", [".md", ".json", ".pdf", ""])
    def test_anything_else_renders_as_a_file(self, suffix: str):
        assert _component_for(suffix) == "file"


class TestMimeFor:
    @pytest.mark.parametrize(("suffix", "expected"), sorted(EXTRA_MIME.items()))
    def test_the_explicit_table_wins_over_the_platforms_guess(
        self, suffix: str, expected: str
    ):
        """mimetypes doesn't know the modern Office types on every platform."""
        assert _mime_for(f"/workspace/out/x{suffix}", suffix) == expected

    def test_a_known_type_is_guessed(self):
        assert _mime_for("/workspace/out/plot.png", ".png") == "image/png"

    def test_an_unknown_type_falls_back_to_a_byte_stream(self):
        assert _mime_for("/workspace/out/x.zzz", ".zzz") == "application/octet-stream"


class TestSweepConfiguration:
    def test_eval_is_a_writer_because_ptc_calls_are_invisible_as_tool_calls(self):
        """One eval does the work of a dozen tool calls; sweeping only `execute` misses most."""
        assert "eval" in WRITER_TOOLS

    def test_every_tool_that_can_leave_a_file_behind_is_a_writer(self):
        assert {"execute", "write_file", "edit_file"} <= WRITER_TOOLS

    def test_the_inline_cap_bounds_what_rides_in_a_checkpointed_state_key(self):
        """A 50 MB xlsx would be re-serialised into every later checkpoint on the thread."""
        assert MAX_INLINE_BYTES == 8 * 1024 * 1024


# --------------------------------------------------------------------------------
# cadence.py — the seconds-since-you-last-spoke reminder
# --------------------------------------------------------------------------------


class TestSpeaksToUser:
    def test_an_ai_message_with_text_counts(self):
        assert _speaks_to_user(AIMessage("Here is what I found.")) is True

    def test_an_ai_message_carrying_only_tool_calls_does_not(self):
        """That silent turn is exactly what this middleware exists to notice."""
        message = AIMessage(
            "", tool_calls=[{"name": "eval", "args": {}, "id": "1", "type": "tool_call"}]
        )
        assert _speaks_to_user(message) is False

    def test_whitespace_only_text_does_not_count(self):
        assert _speaks_to_user(AIMessage("   \n  ")) is False

    def test_a_text_block_in_a_content_list_counts(self):
        assert _speaks_to_user(AIMessage([{"type": "text", "text": "Found it."}])) is True

    @pytest.mark.parametrize("kind", ["thinking", "reasoning"])
    def test_reasoning_blocks_do_not_reset_the_clock(self, kind: str):
        """The frontend doesn't render them, so they put no words in front of the user."""
        assert _speaks_to_user(AIMessage([{"type": kind, "text": "Hmm."}])) is False

    def test_a_human_message_is_not_the_agent_speaking(self):
        assert _speaks_to_user(HumanMessage("Hello?")) is False

    def test_an_empty_content_list_does_not_count(self):
        assert _speaks_to_user(AIMessage([])) is False


class _Request:
    """The slice of `ModelRequest` that `UpdateCadence._prepare` actually touches."""

    def __init__(self, state: dict, messages: list | None = None) -> None:
        self.state = state
        self.messages = messages or []

    def override(self, *, messages: list) -> _Request:
        return _Request(self.state, messages)


class TestUpdateCadence:
    def test_silence_past_the_threshold_injects_a_reminder(self):
        middleware = UpdateCadence(quiet_seconds=10.0)
        request = _Request({"last_update_at": time.time() - 45}, [HumanMessage("Q")])
        prepared = middleware._prepare(request)
        assert len(prepared.messages) == 2
        assert "since the user last heard from you" in prepared.messages[-1].content

    def test_the_reminder_carries_the_elapsed_seconds(self):
        middleware = UpdateCadence(quiet_seconds=10.0)
        request = _Request({"last_update_at": time.time() - 45})
        assert "45s" in middleware._prepare(request).messages[-1].content

    def test_a_short_question_never_sees_it(self):
        """Below the threshold the user has not been kept waiting; it is pure token cost."""
        middleware = UpdateCadence(quiet_seconds=20.0)
        request = _Request({"last_update_at": time.time() - 5})
        assert middleware._prepare(request) is request

    def test_an_unset_clock_injects_nothing(self):
        middleware = UpdateCadence()
        request = _Request({})
        assert middleware._prepare(request) is request

    def test_before_agent_restarts_the_clock_each_turn(self):
        """Otherwise a follow-up opens with however long the user spent reading."""
        middleware = UpdateCadence()
        result = middleware.before_agent({"messages": []}, runtime=None)
        assert result["last_update_at"] == pytest.approx(time.time(), abs=2)

    def test_after_model_restarts_the_clock_when_the_agent_spoke(self):
        middleware = UpdateCadence()
        state = {"messages": [AIMessage("Here is what I found.")]}
        assert middleware.after_model(state, runtime=None) is not None

    def test_after_model_leaves_the_clock_running_on_a_silent_turn(self):
        middleware = UpdateCadence()
        state = {
            "messages": [
                AIMessage("", tool_calls=[{"name": "eval", "args": {}, "id": "1",
                                           "type": "tool_call"}])
            ]
        }
        assert middleware.after_model(state, runtime=None) is None

    def test_an_empty_transcript_leaves_the_clock_running(self):
        assert UpdateCadence().after_model({"messages": []}, runtime=None) is None

    def test_the_default_threshold_is_roughly_one_eval_of_real_work(self):
        assert UpdateCadence().quiet_seconds == QUIET_SECONDS == 20.0

    async def test_the_async_path_is_the_one_that_runs(self):
        """Every entry point runs the graph async; the sync twin exists so the base
        class does not raise."""
        middleware = UpdateCadence(quiet_seconds=10.0)
        request = _Request({"last_update_at": time.time() - 45})
        seen: list[_Request] = []

        async def handler(prepared: _Request) -> str:
            seen.append(prepared)
            return "done"

        assert await middleware.awrap_model_call(request, handler) == "done"
        assert len(seen[0].messages) == 1
