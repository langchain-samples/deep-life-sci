"""PubMed research assistant — a Deep Agent for biologists.

The agent has two ways to run code, and they do different jobs:

* **`eval` (QuickJS)** — the orchestration layer. It has no network, filesystem, or
  shell of its own; it reaches the PubMed tools through programmatic tool calling and
  dispatches subagents with `task()`. This is what lets a whole workflow — search,
  batch-fetch, fan out across 30 abstracts, collect — happen in one step.
* **`execute` (sandbox shell)** — real Python in an isolated Linux container, for
  statistics and plots over data the agent has already fetched.

They compose: `execute` is exposed inside the interpreter, so a single `eval` call can
run search -> fetch -> write -> compute -> collect. See `prompts.py` for the reference
snippets that teach the pattern.

The sandbox starts empty and is deleted when the run ends. `data/` on the host stays the
durable abstract cache that `pubmed.py` owns; the agent never sees it. Anything the agent
wants to compute over, it writes into the sandbox itself.

Traces export to LangSmith automatically when LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set in .env.
"""

import asyncio
import os
import time

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_quickjs import CodeInterpreterMiddleware
from langsmith.sandbox import SandboxClient

# override=True so .env wins over ambient shell values. Without it a LANGSMITH_PROJECT
# already exported in the shell silently captures this project's traces.
#
# But override=True is too blunt for settings you want to vary per run, so anything
# passed explicitly on the command line is captured first and restored afterwards.
# Without this, `MODEL_PROFILE=openai uv run agent.py` is silently overwritten by the
# MODEL_PROFILE in .env and you get the default profile with no indication why.
_CLI_OVERRIDES = {k: v for k in ("MODEL_PROFILE",) if (v := os.environ.get(k))}
load_dotenv(override=True)
os.environ.update(_CLI_OVERRIDES)

from artifacts import ArtifactMiddleware  # noqa: E402
from models import check_gateway_config, describe, root_model, subagent_model  # noqa: E402
from perf import LoopLagProbe, install_logging  # noqa: E402
from pmc import fetch_full_text, make_sandbox_tools, pmc_locate  # noqa: E402
from prompts import (  # noqa: E402
    ABSTRACT_ANALYST,
    FIGURE_ANALYST,
    FULL_TEXT_ANALYST,
    SYSTEM_PROMPT,
)
from pubmed import fetch_abstracts, pubmed_search  # noqa: E402
from resilience import ResilientSandbox  # noqa: E402

# Where the agent works inside the sandbox. Mirrored in the system prompt.
WORKSPACE = "/workspace"

# The sandbox is deleted on context exit, but a `finally` doesn't survive SIGKILL — and
# billing doesn't care why the process died. This is the server-side backstop.
IDLE_TTL_SECONDS = 600

# The default sandbox image is bare Python 3.12 — no numpy/pandas/scipy/matplotlib —
# and installing them costs ~30s. `build_snapshot.py` bakes them into a named snapshot
# instead; booting from it takes ~1s. Run that once, then every run starts warm.
SNAPSHOT_NAME = os.environ.get("SANDBOX_SNAPSHOT_NAME", "pubmed-py")

# Fallback for a clone that hasn't built the snapshot yet. Same package set, paid per run.
PROVISION = (
    f"mkdir -p {WORKSPACE}/out && "
    "pip install --break-system-packages --quiet "
    "numpy pandas scipy matplotlib openpyxl python-docx python-pptx 2>&1 | tail -2"
)


def find_snapshot(client) -> str | None:
    """Return SNAPSHOT_NAME if that snapshot exists, else None.

    A failed lookup falls back to installing at runtime rather than killing the run.
    """
    try:
        snapshots = client.list_snapshots(name_contains=SNAPSHOT_NAME)
    except Exception as exc:  # noqa: BLE001 - a missing snapshot is not a fatal error
        print(f"[sandbox] snapshot lookup failed ({exc}); will install at runtime")
        return None
    return next((s.name for s in snapshots if s.name == SNAPSHOT_NAME), None)


def provision(sandbox) -> None:
    """Install the scientific Python stack into a sandbox that didn't ship with it."""
    t0 = time.monotonic()
    result = sandbox.run(PROVISION, timeout=600)
    if result.exit_code != 0:
        msg = f"sandbox provisioning failed (exit {result.exit_code}): {result.stderr}"
        raise RuntimeError(msg)
    print(f"[sandbox] python ready in {time.monotonic() - t0:.1f}s")


def build_agent(backend):
    """Assemble the agent against a backend. In practice the backend is the sandbox."""
    # fetch_figures and fetch_supplementary need the backend to upload bytes straight
    # into the sandbox, so they're built per-run rather than imported as module globals.
    # That upload is the whole point: an image has to exist as a real file on a real
    # path before a subagent can read_file it and actually see it.
    fetch_figures, fetch_supplementary = make_sandbox_tools(backend)

    # Subagents inherit the parent's tools unless they declare their own, so every leaf
    # sets `tools: []` explicitly. That governs the *parent's* tools — the PubMed ones —
    # and is necessary but not sufficient.
    #
    # `middleware: []` does NOT mean "no middleware". deepagents unconditionally prepends
    # FilesystemMiddleware + summarization + PatchToolCalls + prompt caching to whatever
    # a spec declares (graph.py:667), so a leaf built with `middleware: []` still ends up
    # holding the default filesystem toolset:
    #
    #     ['delete', 'edit_file', 'execute', 'glob', 'grep', 'ls', 'read_file',
    #      'write_file']
    #
    # — `execute` included, i.e. a shell into the shared sandbox. The documented way to
    # narrow it is to pass a *configured FilesystemMiddleware instance*, which deepagents
    # substitutes for the default one instead of stacking on top. Both leaf kinds do that.

    def text_leaf(spec: dict) -> dict:
        """An analyst whose payload arrives in its prompt. Nothing it needs is on disk.

        Their prompts say "you have no tools and cannot retrieve anything". `tools: []`
        alone did not make that true — see the note above — so this narrows the default
        filesystem to the single tool FilesystemMiddleware refuses to drop
        (filesystem.py:1648 requires `read_file`). One tool the leaf has no use for is
        the floor; the point is that `execute`, `grep`, `write_file` and the rest are
        gone.

        The cost of a leaf that can reach the filesystem is measured, not hypothetical.
        In trace 019fde70-69b6-7190-a969-a6a60e52894d three of nine abstract-analysts
        called `read_file` on paths they invented — `/dev/null` twice, and
        `/tmp/pubmed_abstract.txt` — looking for an abstract that was already in the
        prompt. Each error bought a second model turn, and those three were the slowest
        analysts in the fan-out (21.0s, 21.5s, 24.0s against 17.3s for the clean ones).
        In 019fe907-abed-70c0-b589-2cfcb3ef5d2b, with the full default toolset in hand,
        6 of 30 analysts called tools and one spent 36.1s in a single `grep` — a 57.0s
        task against a 16.1s median, setting the critical path the whole `eval` waited on.

        Every such call also opens a sandbox WebSocket, which is the same connection that
        returned HTTP 502 and destroyed a completed 18-way fan-out (see resilience.py).
        """
        return {
            **spec,
            "model": subagent_model(),
            "tools": [],
            "middleware": [FilesystemMiddleware(backend=backend, tools=["read_file"])],
        }

    def image_leaf(spec: dict) -> dict:
        """An analyst that must open a file to do its job.

        Same restriction as text_leaf, for the opposite reason: figure-analyst is handed
        a sandbox path, not an image, and `read_file` on that path is what turns it into
        something the model can actually see.
        """
        return {
            **spec,
            "model": subagent_model(),
            "tools": [],
            "middleware": [FilesystemMiddleware(backend=backend, tools=["read_file"])],
        }

    return create_deep_agent(
        model=root_model(),
        tools=[
            pubmed_search,
            fetch_abstracts,
            pmc_locate,
            fetch_full_text,
            fetch_figures,
            fetch_supplementary,
        ],
        system_prompt=SYSTEM_PROMPT,
        subagents=[
            text_leaf(ABSTRACT_ANALYST),
            text_leaf(FULL_TEXT_ANALYST),
            image_leaf(FIGURE_ANALYST),
        ],
        backend=backend,
        middleware=[
            CodeInterpreterMiddleware(
                # Tools reach JS camelCased: pubmed_search -> tools.pubmedSearch.
                # `execute` is what lets one eval call finish a whole analysis.
                ptc=[
                    "pubmed_search",
                    "fetch_abstracts",
                    "pmc_locate",
                    "fetch_full_text",
                    "fetch_figures",
                    "fetch_supplementary",
                    "execute",
                    "read_file",
                    "write_file",
                    "ls",
                    "glob",
                ],
                # The default is 5 seconds. A fan-out across a few dozen abstracts runs for
                # minutes, so leaving this at the default would kill every real query.
                timeout=900.0,
                # Enough room for the collected fan-out results to survive to synthesis.
                max_result_chars=40_000,
                max_ptc_calls=512,
            ),
            # Carries anything the agent writes to /workspace/out out of the sandbox
            # and into the `ui` state key, so a frontend can render it. Ordered after
            # the interpreter because it sweeps once the tool call it wraps has
            # returned, and `eval` is the tool that does most of the writing.
            ArtifactMiddleware(backend),
            # Last, so it is the innermost wrapper and its wall time is the `eval`
            # itself rather than the artifact sweep that follows it. Measures event-loop
            # lag during the fan-out; see perf.py for what the number distinguishes.
            LoopLagProbe(),
        ],
    )


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

    client = SandboxClient()
    snapshot = find_snapshot(client)
    if snapshot is None:
        print(
            f"[sandbox] no snapshot named {SNAPSHOT_NAME!r} — installing at runtime "
            "(~30s). Run `uv run build_snapshot.py` once to skip this."
        )

    t0 = time.monotonic()
    with client.sandbox(
        snapshot_name=snapshot, idle_ttl_seconds=IDLE_TTL_SECONDS
    ) as sandbox:
        print(f"[sandbox] up in {time.monotonic() - t0:.1f}s ({snapshot or 'base image'})")
        if snapshot is None:
            provision(sandbox)
        # No `reacquire`: the CLI owns exactly one sandbox for the life of the `with`
        # block and has nothing to hand back. Retries still cover a blinking socket,
        # which is the common case.
        backend = ResilientSandbox(sandbox=sandbox)
        try:
            await stream_answer(build_agent(backend), DEMO_QUESTION)
        finally:
            # ainvoke lazily builds a cached async client with its own connection pool;
            # aclose() is the only thing that closes it.
            await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
