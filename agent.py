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
from deepagents.backends import LangSmithSandbox
from deepagents.middleware.filesystem import FilesystemMiddleware
from dotenv import load_dotenv
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
from pmc import fetch_full_text, make_sandbox_tools, pmc_locate  # noqa: E402
from prompts import (  # noqa: E402
    ABSTRACT_ANALYST,
    FIGURE_ANALYST,
    FULL_TEXT_ANALYST,
    SYSTEM_PROMPT,
)
from pubmed import fetch_abstracts, pubmed_search  # noqa: E402

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
    "numpy pandas scipy matplotlib openpyxl 2>&1 | tail -2"
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

    # Every leaf analyst gets the same shape: no PubMed tools, no shell, read_file only.
    # Subagents inherit the parent's tools unless they declare their own, so without
    # these two keys every analyst would get the PubMed tools and a shell into the
    # shared sandbox — contradicting its own description and the no-I/O promise the
    # fan-out depends on. read_file is the floor; FilesystemMiddleware rejects a tools
    # list that omits it, and figure-analyst genuinely needs it: read_file on an image
    # is what turns a staged path into something the model can see.
    def leaf(spec: dict) -> dict:
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
            leaf(ABSTRACT_ANALYST),
            leaf(FULL_TEXT_ANALYST),
            leaf(FIGURE_ANALYST),
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
        ],
    )


# Exercises both surfaces on purpose: the fan-out answers the reading-comprehension half,
# Python answers the quantitative half.
DEMO_QUESTION = (
    "Find recent papers on base editing in the liver and tell me which ones used "
    "in vivo mouse models. Then use Python to plot the distribution of publication "
    "years and report the median year."
)


async def main() -> None:
    # Fail on a bad model config before paying to boot a sandbox.
    check_gateway_config()
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
        backend = LangSmithSandbox(sandbox=sandbox)
        try:
            agent = build_agent(backend)
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": DEMO_QUESTION}]}
            )
            print(result["messages"][-1].content)
        finally:
            # ainvoke lazily builds a cached async client with its own connection pool;
            # aclose() is the only thing that closes it.
            await backend.aclose()


if __name__ == "__main__":
    asyncio.run(main())
