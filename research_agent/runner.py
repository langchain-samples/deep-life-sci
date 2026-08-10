"""Run one question to completion and return the result as data.

This is the seam evaluators attach to. `cli.py` streams to a terminal and `graph.py`
serves a socket; neither gives you a value you can score. Without a third entry point,
every evaluator would re-implement sandbox setup and answer extraction, and they would
drift apart.

What it returns is chosen by what evaluators actually need to judge:

* `answer` — the synthesised text, for grounding and correctness judges.
* `artifacts` — what reached `/workspace/out`. "Did it produce the plot it was asked
  for" is a question about deliverables, and the bytes never appear in `answer`.
* `messages` — the root transcript, for trajectory checks. Whether the agent fanned out
  or ground through papers one at a time is invisible in the final answer and is exactly
  the regression that matters here.
* `root_context_chars` — the number CLAUDE.md tracks as the cost proxy. A prompt change
  that quietly starts reading PNGs back into context shows up here first.

Sandbox isolation is per-call by default. An evaluator scoring twenty questions against
one shared container would let question three's leftover files be swept and published
during question four, and `artifacts` would score the wrong run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from research_agent.agent import build_agent
from research_agent.sandbox import sandbox_session


@dataclass
class RunResult:
    """One completed run, in the shape evaluators want."""

    question: str
    answer: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    root_context_chars: int = 0
    root_turns: int = 0
    tool_calls: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """LangSmith-friendly view. Messages are dropped — they don't serialise cleanly
        and evaluators that want them take the `RunResult` itself."""
        return {
            "answer": self.answer,
            "artifact_names": [a.get("name") for a in self.artifacts],
            "artifact_count": len(self.artifacts),
            "root_context_chars": self.root_context_chars,
            "root_turns": self.root_turns,
            "tool_calls": self.tool_calls,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _final_answer(messages: list[Any]) -> str:
    """The last AI message with actual text in it.

    Not simply `messages[-1]`: a run can end on a `ToolMessage` when the artifact sweep
    fires after the model's closing turn, and `.text` on an AI message that only made
    tool calls is empty.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage) and (text := message.text):
            return text
    return ""


def _summarise(messages: list[Any]) -> tuple[int, int, list[str]]:
    """Root-transcript size, AI turn count, and the tool calls made, in one pass."""
    chars = 0
    turns = 0
    calls: list[str] = []
    for message in messages:
        chars += len(str(getattr(message, "content", "") or ""))
        if isinstance(message, AIMessage):
            turns += 1
            calls.extend(tc.get("name", "?") for tc in (message.tool_calls or []))
    return chars, turns, calls


async def run_once(question: str, *, backend=None, quiet: bool = True) -> RunResult:
    """Answer one question and return everything worth scoring.

    Args:
        question: The user turn.
        backend: An existing sandbox backend. Pass one to reuse a container across
            questions — cheaper, but the runs are then no longer independent, so don't
            do it inside an eval whose artifacts you intend to score.
        quiet: Suppress sandbox boot chatter. On by default because the caller here is
            usually a harness, not a person.
    """
    if backend is not None:
        return await _run(question, backend)
    async with sandbox_session(quiet=quiet) as owned:
        return await _run(question, owned)


async def _run(question: str, backend) -> RunResult:
    started = time.monotonic()
    agent = build_agent(backend)
    state = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    elapsed = time.monotonic() - started

    messages = state.get("messages", [])
    chars, turns, calls = _summarise(messages)
    return RunResult(
        question=question,
        answer=_final_answer(messages),
        # `ui` is where ArtifactMiddleware publishes; absent when the run wrote nothing.
        artifacts=list(state.get("ui", []) or []),
        messages=messages,
        root_context_chars=chars,
        root_turns=turns,
        tool_calls=calls,
        duration_seconds=elapsed,
    )
