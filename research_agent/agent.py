"""Agent assembly. Nothing here boots a sandbox or runs a question.

The agent has two ways to run code, and they do different jobs:

* **`eval` (QuickJS)** — the orchestration layer. It has no network, filesystem, or
  shell of its own; it reaches the source tools through programmatic tool calling and
  dispatches subagents with `task()`. This is what lets a whole workflow — search,
  batch-fetch, fan out across 30 abstracts, collect — happen in one step.
* **`execute` (sandbox shell)** — real Python in an isolated Linux container, for
  statistics and plots over data the agent has already fetched.

They compose: `execute` is exposed inside the interpreter, so a single `eval` call can
run search -> fetch -> write -> compute -> collect. See `prompts/system.py` for the
reference snippets that teach the pattern.

The sandbox starts empty. `data/` on the host stays the durable abstract cache that
`sources/pubmed.py` owns; the agent never sees it. Anything the agent wants to compute
over, it writes into the sandbox itself.

`build_agent` takes a backend rather than making one, which is what lets the same
assembly serve three different sandbox lifetimes — `cli.py`'s single block, `graph.py`'s
thread-keyed container, and `evals/`'s per-example throwaway. See `sandbox.py`.
"""

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from research_agent.middleware.artifacts import ArtifactMiddleware
from research_agent.middleware.perf import LoopLagProbe
from research_agent.models import root_model, subagent_model
from research_agent.prompts import (
    ABSTRACT_ANALYST,
    FIGURE_ANALYST,
    FULL_TEXT_ANALYST,
    TRIAL_ANALYST,
    build_system_prompt,
)
from research_agent.sources.ctgov import ctgov_fetch, ctgov_search
from research_agent.sources.pmc import fetch_full_text, make_sandbox_tools, pmc_locate
from research_agent.sources.pubmed import fetch_abstracts, pubmed_search


def build_agent(backend):
    """Assemble the agent against a backend. In practice the backend is the sandbox."""
    # fetch_figures and fetch_supplementary need the backend to upload bytes straight
    # into the sandbox, so they're built per-run rather than imported as module globals.
    # That upload is the whole point: an image has to exist as a real file on a real
    # path before a subagent can read_file it and actually see it.
    fetch_figures, fetch_supplementary = make_sandbox_tools(backend)

    # Subagents inherit the parent's tools unless they declare their own, so every leaf
    # sets `tools: []` explicitly. That governs the *parent's* tools — the PubMed and
    # registry ones — and is necessary but not sufficient. It is also what keeps
    # ClinicalTrials.gov's ~1 req/sec limit survivable: a fan-out of leaves that could
    # each fetch would trip it at twelve.
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
    # substitutes for the default one instead of stacking on top.

    def analyst_leaf(spec: dict) -> dict:
        """A leaf on the cheap model, narrowed to `read_file` and nothing else.

        The two leaf kinds arrive at the same restriction from opposite directions, which
        is why one function serves both:

        * **The text analysts** (abstract, full-text) have their payload in the prompt and
          need nothing on disk. Their prompts say "you have no tools and cannot retrieve
          anything". `tools: []` alone did not make that true — see the note above — so
          this narrows the default filesystem to the single tool FilesystemMiddleware
          refuses to drop (filesystem.py:1648 requires `read_file`). One tool the leaf has
          no use for is the floor; the point is that `execute`, `grep`, `write_file` and
          the rest are gone.
        * **The figure analyst** is handed a sandbox path rather than an image, and
          `read_file` on that path is what turns it into something the model can see. For
          it the floor is exactly the tool it needs.

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
        returned HTTP 502 and destroyed a completed 18-way fan-out (see sandbox.py).
        """
        return {
            **spec,
            "model": subagent_model(),
            "tools": [],
            "middleware": [FilesystemMiddleware(backend=backend, tools=["read_file"])],
        }

    agent = create_deep_agent(
        model=root_model(),
        tools=[
            pubmed_search,
            fetch_abstracts,
            pmc_locate,
            fetch_full_text,
            fetch_figures,
            fetch_supplementary,
            ctgov_search,
            ctgov_fetch,
        ],
        system_prompt=build_system_prompt(),
        subagents=[
            analyst_leaf(ABSTRACT_ANALYST),
            analyst_leaf(FULL_TEXT_ANALYST),
            analyst_leaf(FIGURE_ANALYST),
            analyst_leaf(TRIAL_ANALYST),
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
                    "ctgov_search",
                    "ctgov_fetch",
                    "execute",
                    "read_file",
                    "write_file",
                    # Without this, the only way to change one line of a Python script
                    # written from JS is to re-emit the whole script. In thread
                    # 019fe982-9296-7a23-836a-bd3ae24605a1 a single bad `RandomState`
                    # seed cost 1,845 output tokens and 26s of retyping a 4.5k-char
                    # matplotlib script that was already on disk.
                    "edit_file",
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
            # lag during the fan-out; see middleware/perf.py for what the number
            # distinguishes.
            LoopLagProbe(),
        ],
    )
    # A large fan-out plus subagent turns can exceed LangGraph's default limit of 25
    # super-steps well before anything is actually wrong; this is a ceiling against a
    # genuine runaway loop, not a tuning knob for normal runs.
    return agent.with_config(recursion_limit=200)
