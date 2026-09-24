"""`sources/web.py`: the digest parser for two providers' server-side search.

This module has no HTTP client of its own — the search runs inside the model provider —
so what there is to test is the block-shape handling, and it is genuinely two shapes:
Anthropic hangs `citations` off the sentence they support, OpenAI hangs `annotations` off
the whole text block, and the two name their search-call blocks differently as well.

The behaviour worth defending hardest is that **nothing here raises**. A tool exception
inside `eval` propagates out of the interpreter and kills the run, so a provider content
filter rejecting a biomedical query would destroy a completed fan-out that had nothing to
do with the web.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deep_life_sci.sources import web
from deep_life_sci.sources.web import (
    MAX_ANSWER_CHARS,
    MAX_SOURCES,
    _answer,
    _blocks,
    _failed,
    _searched,
    _sources,
    _warnings,
    web_search,
)


class TestBlocks:
    def test_a_bare_string_reply_is_normalised_to_one_text_block(self):
        """A turn with no search comes back as a string on either path."""
        assert _blocks("Just prose.") == [{"type": "text", "text": "Just prose."}]

    def test_a_list_of_blocks_passes_through(self):
        blocks = [{"type": "text", "text": "a"}]
        assert _blocks(blocks) == blocks

    def test_non_dict_members_are_dropped(self):
        assert _blocks([{"type": "text"}, "stray"]) == [{"type": "text"}]


class TestAnswer:
    def test_text_blocks_are_joined_in_order(self):
        blocks = [{"type": "text", "text": "One. "}, {"type": "text", "text": "Two."}]
        assert _answer(blocks) == "One. Two."

    @pytest.mark.parametrize("kind", ["thinking", "reasoning"])
    def test_reasoning_is_excluded_by_type_not_by_position(self, kind: str):
        """Anthropic emits `thinking`, OpenAI `reasoning`, both interleaved."""
        blocks = [
            {"type": kind, "text": "Let me think."},
            {"type": "text", "text": "The answer."},
            {"type": kind, "text": "More thinking."},
        ]
        assert _answer(blocks) == "The answer."

    def test_a_text_block_with_no_text_key_contributes_nothing(self):
        assert _answer([{"type": "text"}]) == ""


class TestSources:
    def test_reads_anthropics_per_sentence_citations(self):
        blocks = [
            {
                "type": "text",
                "text": "A claim.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://fda.gov/a",
                        "title": "FDA",
                    }
                ],
            }
        ]
        assert _sources(blocks) == [{"url": "https://fda.gov/a", "title": "FDA"}]

    def test_reads_openais_per_block_annotations(self):
        blocks = [
            {
                "type": "text",
                "text": "A claim.",
                "annotations": [
                    {"type": "url_citation", "url": "https://ema.europa.eu/b", "title": "EMA"}
                ],
            }
        ]
        assert _sources(blocks) == [{"url": "https://ema.europa.eu/b", "title": "EMA"}]

    def test_the_same_url_cited_twice_appears_once_in_first_cited_order(self):
        blocks = [
            {"type": "text", "citations": [{"url": "https://b", "title": "B"}]},
            {"type": "text", "citations": [{"url": "https://a", "title": "A"}]},
            {"type": "text", "annotations": [{"url": "https://b", "title": "B again"}]},
        ]
        assert [s["url"] for s in _sources(blocks)] == ["https://b", "https://a"]

    def test_an_uncited_answer_falls_back_to_what_the_search_returned(self):
        """Otherwise an uncited answer is attributed to nothing at all."""
        blocks = [
            {"type": "text", "text": "From memory of the results."},
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://who.int/x", "title": "WHO"}],
            },
        ]
        assert _sources(blocks) == [{"url": "https://who.int/x", "title": "WHO"}]

    def test_citations_win_over_the_fallback_so_the_list_stays_short(self):
        """Both providers retrieve several times what they end up citing."""
        blocks = [
            {"type": "text", "citations": [{"url": "https://cited", "title": "Cited"}]},
            {
                "type": "web_search_tool_result",
                "content": [{"url": "https://retrieved", "title": "Retrieved"}],
            },
        ]
        assert [s["url"] for s in _sources(blocks)] == ["https://cited"]

    def test_an_error_result_block_holding_a_dict_is_ignored(self):
        """An *error* result puts a dict where the list normally goes."""
        blocks = [
            {"type": "text", "text": "x"},
            {
                "type": "web_search_tool_result",
                "content": {"type": "web_search_tool_result_error", "error_code": "x"},
            },
        ]
        assert _sources(blocks) == []

    def test_the_list_is_capped(self):
        blocks = [
            {
                "type": "text",
                "citations": [
                    {"url": f"https://s{n}", "title": str(n)} for n in range(MAX_SOURCES + 5)
                ],
            }
        ]
        assert len(_sources(blocks)) == MAX_SOURCES

    def test_a_citation_with_no_url_is_skipped(self):
        assert _sources([{"type": "text", "citations": [{"title": "No url"}]}]) == []

    def test_a_missing_title_becomes_an_empty_string_rather_than_none(self):
        blocks = [{"type": "text", "citations": [{"url": "https://a"}]}]
        assert _sources(blocks) == [{"url": "https://a", "title": ""}]


class TestSearched:
    def test_reads_anthropics_server_tool_use_queries(self):
        blocks = [
            {
                "type": "server_tool_use",
                "name": "web_search",
                "input": {"query": "semaglutide label 2026"},
            }
        ]
        assert _searched(blocks) == ["semaglutide label 2026"]

    def test_reads_openais_agentic_actions_including_pages_opened(self):
        """OpenAI reports open_page and find_in_page alongside search."""
        blocks = [
            {"type": "web_search_call", "action": {"queries": ["a", "b"]}},
            {"type": "web_search_call", "action": {"url": "https://fda.gov/page"}},
        ]
        assert _searched(blocks) == ["a", "b", "https://fda.gov/page"]

    def test_a_singular_query_key_is_read_too(self):
        blocks = [{"type": "web_search_call", "action": {"query": "just one"}}]
        assert _searched(blocks) == ["just one"]

    def test_repeated_queries_are_deduplicated_in_order(self):
        blocks = [
            {"type": "web_search_call", "action": {"queries": ["a"]}},
            {"type": "web_search_call", "action": {"queries": ["a", "b"]}},
        ]
        assert _searched(blocks) == ["a", "b"]

    def test_a_server_tool_use_for_something_else_is_ignored(self):
        blocks = [{"type": "server_tool_use", "name": "code_execution", "input": {}}]
        assert _searched(blocks) == []


class TestWarnings:
    def test_an_answer_from_memory_is_the_failure_that_would_otherwise_look_clean(self):
        blocks = [{"type": "text", "text": "I recall that..."}]
        assert any("no search was performed" in w for w in _warnings(blocks, "x"))

    def test_a_search_that_ran_warns_about_nothing(self):
        blocks = [{"type": "web_search_call", "action": {"queries": ["a"]}}]
        assert _warnings(blocks, "short answer") == []

    def test_an_anthropic_error_block_is_reported_with_its_code(self):
        blocks = [
            {"type": "web_search_call", "action": {"queries": ["a"]}},
            {
                "type": "web_search_tool_result",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "max_uses_exceeded",
                },
            },
        ]
        assert any("max_uses_exceeded" in w for w in _warnings(blocks, "x"))

    def test_an_openai_call_that_did_not_complete_is_reported(self):
        blocks = [{"type": "web_search_call", "status": "failed", "action": {"queries": ["a"]}}]
        assert any("search failed" in w for w in _warnings(blocks, "x"))

    def test_a_completed_call_is_not_reported(self):
        blocks = [
            {"type": "web_search_call", "status": "completed", "action": {"queries": ["a"]}}
        ]
        assert _warnings(blocks, "x") == []

    def test_a_long_answer_says_it_was_truncated(self):
        blocks = [{"type": "web_search_call", "action": {"queries": ["a"]}}]
        assert any("truncated" in w for w in _warnings(blocks, "x" * (MAX_ANSWER_CHARS + 1)))


class TestFailed:
    def test_carries_the_reason_in_the_channel_the_prompt_teaches(self):
        result = _failed("q", "the model 400'd")
        assert result["answer"] == ""
        assert result["sources"] == []
        assert any("the model 400'd" in w for w in result["warnings"])

    def test_keeps_the_ordinary_digest_shape(self):
        assert set(_failed("q", "r")) == {"query", "answer", "sources", "searched", "warnings"}


class TestWebSearchTool:
    async def test_an_empty_query_returns_a_digest_rather_than_raising(self):
        result = await web_search.ainvoke({"query": "   "})
        assert result["answer"] == ""
        assert any("needs a question" in w for w in result["warnings"])

    async def test_a_model_failure_returns_a_digest_rather_than_raising(self, monkeypatch):
        """A raise here propagates out of `eval` and ends the run, not the call."""

        class Exploding:
            async def ainvoke(self, _prompt):
                raise RuntimeError("400 web_search is not supported for this model")

        monkeypatch.setattr(web, "web_search_model", lambda: Exploding())
        result = await web_search.ainvoke({"query": "what did the FDA approve?"})
        assert result["answer"] == ""
        assert any("web search unavailable" in w for w in result["warnings"])

    async def test_a_provider_400_keeps_its_words_and_the_content_filter_hint(self, monkeypatch):
        """A filtered biomedical query is a 400 too; it must not read as a config fault."""
        import httpx
        import openai

        request = httpx.Request("POST", "https://gateway.invalid/v1/responses")
        error = openai.BadRequestError(
            "flagged by the content filter", response=httpx.Response(400, request=request),
            body=None,
        )

        class Filtered:
            async def ainvoke(self, _prompt):
                raise error

        monkeypatch.setattr(web, "web_search_model", lambda: Filtered())
        result = await web_search.ainvoke({"query": "lethal dose of fentanyl in mice"})
        (warning,) = result["warnings"]
        assert "flagged by the content filter" in warning
        assert "content filter can reject a query" in warning

    async def test_a_model_that_cannot_be_built_is_contained_too(self, monkeypatch):
        def unbuildable():
            raise ValueError("reasoning_effort Input should be 'low', ...")

        monkeypatch.setattr(web, "web_search_model", unbuildable)
        result = await web_search.ainvoke({"query": "what did the FDA approve?"})
        assert any("reasoning_effort" in w for w in result["warnings"])

    async def test_a_good_search_is_assembled_into_the_documented_shape(self, monkeypatch):
        message = SimpleNamespace(
            content=[
                {"type": "web_search_call", "action": {"queries": ["fda semaglutide"]}},
                {
                    "type": "text",
                    "text": "Approved in 2021.",
                    "annotations": [{"url": "https://fda.gov/a", "title": "FDA"}],
                },
            ]
        )

        class Stub:
            async def ainvoke(self, _prompt):
                return message

        monkeypatch.setattr(web, "web_search_model", lambda: Stub())
        result = await web_search.ainvoke({"query": "when was it approved?"})
        assert result["query"] == "when was it approved?"
        assert result["answer"] == "Approved in 2021."
        assert result["sources"] == [{"url": "https://fda.gov/a", "title": "FDA"}]
        assert result["searched"] == ["fda semaglutide"]
        assert result["warnings"] == []

    async def test_the_answer_is_truncated_to_the_cap(self, monkeypatch):
        message = SimpleNamespace(
            content=[
                {"type": "web_search_call", "action": {"queries": ["q"]}},
                {"type": "text", "text": "x" * (MAX_ANSWER_CHARS + 100)},
            ]
        )

        class Stub:
            async def ainvoke(self, _prompt):
                return message

        monkeypatch.setattr(web, "web_search_model", lambda: Stub())
        result = await web_search.ainvoke({"query": "q"})
        assert len(result["answer"]) == MAX_ANSWER_CHARS
        assert any("truncated" in w for w in result["warnings"])
