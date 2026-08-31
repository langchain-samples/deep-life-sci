"""One-shot CLI: ask a question, stream the answer, exit.

    uv run agent                                   # the demo question, default pair
    uv run agent "does X bind Y?"                  # your own question
    ROOT_MODEL=openai/gpt-5.6-terra uv run agent   # swap one role; SUBAGENT_/JUDGE_ too
    ROOT_EFFORT=high uv run agent                  # same pair, more thinking

Traces export to LangSmith automatically when LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set in .env.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk

# override=True so .env wins over ambient shell values. Without it a LANGSMITH_PROJECT
# already exported in the shell silently captures this project's traces.
#
# But override=True is too blunt for settings you want to vary per run, so anything
# passed explicitly on the command line is captured first and restored afterwards.
# Without this, `ROOT_MODEL=... uv run agent` is silently overwritten by the ROOT_MODEL
# in .env and you get the default pair with no indication why.
#
# The names come from models.py rather than a copy here, so a new axis is preserved by
# adding it there and nowhere else. Importing that module this early is safe *because* it
# reads the environment only inside functions — unlike `sandbox.py` below.
from research_agent.models import ENV_VARS

_CLI_OVERRIDES = {k: v for k in ENV_VARS if (v := os.environ.get(k))}
load_dotenv(override=True)
os.environ.update(_CLI_OVERRIDES)

# Imported after load_dotenv on purpose: `sandbox.py` reads SANDBOX_SNAPSHOT_NAME at
# import time, so importing it first would bake in the pre-.env value.
from research_agent.agent import build_agent  # noqa: E402
from research_agent.middleware.perf import install_logging  # noqa: E402
from research_agent.models import check_gateway_config, describe  # noqa: E402
from research_agent.sandbox import sandbox_session  # noqa: E402
from research_agent.sources import cache_io  # noqa: E402

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


async def main(question: str) -> None:
    # Fail on a bad model config before paying to boot a sandbox.
    check_gateway_config()
    install_logging()
    print(f"[models] {describe()}\n")

    # Drop expired cache entries before the run rather than after: a one-shot process has
    # no "after". Measured at ~11ms for a 1800-file cache, so it stays off the critical
    # path in the only sense that matters.
    await cache_io.sweep_if_due()

    # The CLI owns exactly one sandbox for the life of the block and has nothing to hand
    # back, so no `reacquire` is wired; `sandbox_session` also does the `aclose()` that
    # releases the lazily-built async connection pool.
    async with sandbox_session() as backend:
        await stream_answer(build_agent(backend), question)


def run() -> None:
    """Console-script entry point (`uv run agent`).

    Joined rather than taken as argv[1] so an unquoted question still works — the
    shell has already split it into words by the time it arrives here, and the
    alternative is a confusing "unexpected argument" for a natural way to type it.
    """
    required = ("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        # Without this the first model call dies as an SDK auth error a long way from its
        # cause. The launcher used to check this in shell; it lives here so the check
        # survives being invoked as `uv run agent` directly, which is the Windows path.
        raise SystemExit(
            f"[agent] not set up — {', '.join(missing)} missing from .env.\n"
            "[agent] run:  uv run scripts/setup.py"
        )
    asyncio.run(main(" ".join(sys.argv[1:]).strip() or DEMO_QUESTION))


if __name__ == "__main__":
    run()
