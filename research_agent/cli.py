"""One-shot CLI: ask the demo question, stream the answer, exit.

    uv run agent                        # default `anthropic` profile
    MODEL_PROFILE=mixed uv run agent    # switch model pair (see models.py PROFILES)

Traces export to LangSmith automatically when LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set in .env.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk

# override=True so .env wins over ambient shell values. Without it a LANGSMITH_PROJECT
# already exported in the shell silently captures this project's traces.
#
# But override=True is too blunt for settings you want to vary per run, so anything
# passed explicitly on the command line is captured first and restored afterwards.
# Without this, `MODEL_PROFILE=openai uv run agent` is silently overwritten by the
# MODEL_PROFILE in .env and you get the default profile with no indication why.
_CLI_OVERRIDES = {k: v for k in ("MODEL_PROFILE",) if (v := os.environ.get(k))}
load_dotenv(override=True)
os.environ.update(_CLI_OVERRIDES)

# Imported after load_dotenv on purpose: `sandbox.py` reads SANDBOX_SNAPSHOT_NAME at
# import time, so importing it first would bake in the pre-.env value.
from research_agent.agent import build_agent  # noqa: E402
from research_agent.middleware.perf import install_logging  # noqa: E402
from research_agent.models import check_gateway_config, describe  # noqa: E402
from research_agent.sandbox import sandbox_session  # noqa: E402

# Exercises both surfaces on purpose: the fan-out answers the reading-comprehension half,
# Python answers the quantitative half.
DEMO_QUESTION = (
    "Find recent papers on base editing in the liver and tell me which ones used "
    "in vivo mouse models. Then use Python to plot the distribution of publication "
    "years and report the median year."
)


async def stream_answer(agent, question: str) -> None:
    """Run the agent, printing the final answer as it is generated.

    Synthesis is the single longest span in a run — 23.4s and 32.2s in the two traces
    from thread 019fde6d-d25c-77b3-a751-56c6b7aa4ead, against a measured 48-60 tok/s
    for Sonnet with a 1.6s time-to-first-token. `ainvoke` returns nothing until that
    span completes, so the user waits the full ~30s staring at a blank terminal for
    text that was ready to show after 1.6s. Streaming does not make the run shorter;
    it removes almost all of the *perceived* latency of its slowest part.

    Two filters, both load-bearing:

    * **`AIMessageChunk` only.** `stream_mode="messages"` also emits `ToolMessage`s, and
      a `ToolMessage`'s `.text` is the tool's *result* — an `eval` result runs up to
      max_result_chars (40k), so printing those dumps the entire fan-out payload into
      the terminal ahead of the answer that summarises it.
    * **Root graph only.** Subagent model tokens do not currently reach this stream at
      all, because `task` invokes its subagent rather than streaming it. If that
      changes, 18 analysts interleaved token-by-token would be unreadable, so nested
      namespaces are dropped rather than relied upon to stay empty.
    """
    printed = False
    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        # Root nodes are a single segment ("model:<uuid>"); anything running inside a
        # subgraph carries its parent's segments ahead of its own, joined by "|".
        if "|" in (metadata or {}).get("langgraph_checkpoint_ns", ""):
            continue
        # `.text` is the natural-language part only — tool-call arguments stream as
        # separate content blocks and are not what the user is waiting to read.
        if text := chunk.text:
            print(text, end="", flush=True)
            printed = True
    if printed:
        print()


async def main() -> None:
    # Fail on a bad model config before paying to boot a sandbox.
    check_gateway_config()
    install_logging()
    print(f"[models] {describe()}\n")

    # The CLI owns exactly one sandbox for the life of the block and has nothing to hand
    # back, so no `reacquire` is wired; `sandbox_session` also does the `aclose()` that
    # releases the lazily-built async connection pool.
    async with sandbox_session() as backend:
        await stream_answer(build_agent(backend), DEMO_QUESTION)


def run() -> None:
    """Console-script entry point (`uv run agent`)."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
