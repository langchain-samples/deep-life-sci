"""Web search as a PTC tool, for the questions PubMed and the registry cannot answer.

Regulatory actions, drug labels, guideline versions, conference abstracts, prices,
software documentation: real questions from life scientists that no NCBI index holds.
This module answers them without giving up the design's core economy.

**The shape is unusual on purpose.** Both providers ship web search as a *server-side*
tool — the search runs inside the provider during a model turn, and its results come
back as content blocks in the assistant message. Bind that to the root model and every
retrieved page lands in root context, outside `eval` and outside PTC, which is precisely
what `CLAUDE.md`'s "what must never enter root context" section forbids. Measured
through this gateway on 2026-09-03: one such question cost 27.9k input tokens
(Anthropic, 1 search) and 38.6k (OpenAI, 3 searches), against a whole-run root budget
the repo tunes in the low tens of thousands of *characters*.

So the search is spent inside a tool instead. `web_search` makes one throwaway call to
the `search` role (`models.web_search_model`), which carries the provider's own search
tool, and returns a *digest* — written findings plus the URLs behind them. Being a
tool, its return value is marshalled into the JS heap like every other source, and the
root model sees only what its JavaScript chooses to keep. Same pattern as the analyst
leaves: something cheap reads, the root synthesises.

Two deliberate departures from its sibling modules:

* **No cache.** Every other source here caches, because a PMID's abstract does not
  change and NCBI's rate limit is the binding constraint. Neither holds: an approval
  date or a label revision is exactly the kind of fact where a stale hit is a *wrong
  answer*, and the provider meters the searching, so there is no local budget to
  protect. `paths.py` gets no new directory.
* **No `Throttle`, but a concurrency cap.** There is no documented per-second limit to
  pace against because the requests are not ours. `_SEMAPHORE` is about cost and
  latency instead: a `Promise.all` over twenty questions is twenty model calls, each of
  which does its own multi-step browsing.

The two providers' block shapes differ and both are handled by `_digest`; the probe
output they were written against is in the module's git history rather than a notes
directory, since this surface has no API of its own to characterise.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from langchain_core.tools import tool

from research_agent.models import describe, web_search_model

# Cost and latency, not politeness — see the module docstring. Eight concurrent searches
# is already a wide question; more usually means the model is enumerating rather than
# searching, and it should narrow instead.
MAX_CONCURRENCY = 8
_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)

# A digest that runs longer than this is no longer a digest. Measured output for a
# single question was ~7k chars (Anthropic) and ~3.5k (OpenAI), so this rarely bites;
# when it does, the caller is told rather than left to wonder.
MAX_ANSWER_CHARS = 8_000

# Sources actually cited, deduplicated by URL. Both providers cite far fewer than they
# retrieve, which is the point — this is the attribution list, not the search log.
MAX_SOURCES = 15

_PROMPT = """\
Search the web and answer the question for a research scientist. Today is {today}.

- Ground every claim in a page you actually retrieved. Where sources conflict or are
  thin, say so rather than resolving it yourself.
- Prefer primary and regulatory sources — FDA/EMA labels and approval letters, guideline
  bodies, registries, journal pages — over news write-ups and secondary summaries.
- Date anything that changes over time: approvals, label revisions, guideline versions.
- Findings only. No preamble, no restatement of the question, no closing summary.
- If the search turns up nothing usable, say that plainly instead of reasoning from
  memory.

Question: {query}
"""


def _failed(query: str, reason: str) -> dict:
    """The tool's return shape for a search that could not run at all.

    **This must not raise.** A tool exception inside `eval` is not handed back to the
    JavaScript — it propagates out of the interpreter and kills the whole run, so one
    failed search destroys a completed fan-out that had nothing to do with the web. That
    is not hypothetical: a provider bio-risk filter (`code: bio_policy`) rejected a query
    about a bacterial toxin's NLS and took down an otherwise clean run, on a question
    PubMed alone could answer.

    So a failure comes back as an ordinary digest with an empty `answer` and the reason in
    `warnings`, which is the channel the prompt already teaches the model to check. The
    root model can then route around the web and answer from the tool-backed sources.

    The reason worth spelling out is a misconfigured `search` role: a model that does not
    support its path's server-side search answers with a 400 from deep inside the SDK,
    which reads like an outage rather than a setting. The model id travels with the
    warning so the fix is obvious.
    """
    return {
        "query": query,
        "answer": "",
        "sources": [],
        "searched": [],
        "warnings": [f"web search unavailable: {reason}"],
    }


def _blocks(content: Any) -> list[dict]:
    """The message's content as a list of block dicts.

    Both paths return a list of blocks for a search-bearing turn, but a plain-text reply
    (no search performed) comes back as a bare string on either one, so that case is
    normalised rather than special-cased downstream.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)]


def _answer(blocks: list[dict]) -> str:
    """The written digest: text blocks only, in order.

    Reasoning blocks are excluded by type rather than by position — Anthropic emits
    `thinking` and OpenAI `reasoning`, both interleaved with the text, and neither is
    part of the answer.
    """
    return "".join(b.get("text") or "" for b in blocks if b.get("type") == "text").strip()


def _sources(blocks: list[dict]) -> list[dict]:
    """The pages the answer is attributed to, deduplicated, in first-cited order.

    Citations live in different keys per path and in the same place — on the text blocks:
    Anthropic hangs `citations` (`web_search_result_location`) off the sentence they
    support, OpenAI hangs `annotations` (`url_citation`) off the whole text block. Both
    carry `url` and `title`, so one loop covers them.

    The fallback matters. When a model searches but cites nothing, the citation lists are
    empty while `web_search_tool_result` still holds everything the search returned, so
    an uncited answer is attributed to its search results rather than to nothing at all.
    Preferring citations when they exist is what keeps this list short: both providers
    retrieve several times what they end up citing.
    """
    found: dict[str, str] = {}

    def add(entry: Any) -> None:
        if isinstance(entry, dict) and (url := entry.get("url")):
            found.setdefault(url, entry.get("title") or "")

    for block in blocks:
        if block.get("type") == "text":
            for cite in (block.get("citations") or []) + (block.get("annotations") or []):
                add(cite)
    if not found:
        for block in blocks:
            if block.get("type") == "web_search_tool_result":
                # An *error* result puts a dict here where a list normally goes; `add`
                # ignores the non-dict members that iterating one would yield.
                for result in block.get("content") or []:
                    add(result)
    return [{"url": u, "title": t} for u, t in list(found.items())[:MAX_SOURCES]]


def _searched(blocks: list[dict]) -> list[str]:
    """What the model actually did — queries it ran and pages it opened, deduplicated.

    Worth returning because it is the cheapest way for the caller to see that a digest
    saying "nothing found" came from a bad query rather than from an empty web. OpenAI's
    search is agentic and reports `open_page` and `find_in_page` actions alongside
    `search`, so pages opened are listed here too.
    """
    out: list[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "server_tool_use" and block.get("name") == "web_search":
            out.append((block.get("input") or {}).get("query") or "")
        elif kind == "web_search_call":
            action = block.get("action") or {}
            out.extend(action.get("queries") or [action.get("query") or ""])
            out.append(action.get("url") or "")
    return list(dict.fromkeys(q for q in out if q))


def _warnings(blocks: list[dict], answer: str) -> list[str]:
    """Ways the search did not run as asked. Empty means it did.

    Three things can go wrong without raising: Anthropic replaces a result block with a
    `web_search_tool_result_error` (its `max_uses_exceeded` is reachable from here, since
    `WEB_SEARCH_SPECS` sets that cap), OpenAI marks a `web_search_call` with a status
    other than `completed`, and either path can answer from memory without searching at
    all — which is the one failure that would otherwise look like a clean result.
    """
    out = []
    for block in blocks:
        if block.get("type") == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, dict) and "error" in (content.get("type") or ""):
                out.append(f"search error: {content.get('error_code') or 'unknown'}")
        elif block.get("type") == "web_search_call":
            status = block.get("status")
            if status not in (None, "completed"):
                out.append(f"search {status}")
    if not _searched(blocks):
        out.append("no search was performed; the answer is unsourced")
    if len(answer) > MAX_ANSWER_CHARS:
        out.append(f"answer truncated to {MAX_ANSWER_CHARS} chars")
    return out


@tool
async def web_search(query: str) -> dict:
    """Search the web and return a written digest with the URLs behind it.

    For what PubMed and ClinicalTrials.gov structurally cannot answer: regulatory
    actions and drug labels, clinical guidelines, conference abstracts, company
    announcements, prices, methods and software documentation. Anything about a paper or
    a registered trial belongs to those tools instead, which cite by PMID and NCT id.

    A cheap model does the searching and the reading; you get its findings, not the
    pages. Ask a full question rather than keywords — the search is agentic and will
    issue several queries and open pages of its own to answer one. One call per question;
    `Promise.all` several if you have several, up to about eight at a time.

    Args:
        query: The question, in natural language. Include the timeframe if it matters —
            'as of 2026' rather than 'current' — since a retrieved page may be old.

    Returns:
        query: the question as asked, for the record
        answer: the digest, with claims attributed inline
        sources: [{url, title}] the answer is attributed to, in first-cited order
        searched: queries run and pages opened, so a thin answer can be diagnosed
        warnings: ways the search did not run as asked. Empty means it ran clean; a
            'no search was performed' warning means the answer came from the model's
            memory and should not be used. A search that could not run at all returns
            this same shape with an empty `answer` and a 'web search unavailable'
            warning — never an exception, which would kill the run.
    """
    if not (query := query.strip()):
        return _failed(query, "web_search needs a question; an empty query is not one")

    model = web_search_model()
    try:
        async with _SEMAPHORE:
            message = await model.ainvoke(
                _PROMPT.format(today=date.today().isoformat(), query=query)
            )
    except Exception as exc:  # noqa: BLE001 - a failed search must not kill the run
        return _failed(
            query,
            f"{describe('search')} failed: {exc}. Not every model supports its provider's "
            "server-side web search, and a provider content filter can reject a query "
            "outright; if this is a 400, one of those is the likely cause.",
        )

    blocks = _blocks(message.content)
    answer = _answer(blocks)
    return {
        "query": query,
        "answer": answer[:MAX_ANSWER_CHARS],
        "sources": _sources(blocks),
        "searched": _searched(blocks),
        "warnings": _warnings(blocks, answer),
    }
