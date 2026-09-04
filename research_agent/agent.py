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

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from research_agent.middleware.artifacts import ArtifactMiddleware
from research_agent.middleware.cadence import UpdateCadence
from research_agent.middleware.perf import LoopLagProbe
from research_agent.middleware.progress import with_progress
from research_agent.middleware.tool_errors import with_error_capture
from research_agent.middleware.uploads import UploadMiddleware
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
from research_agent.sources.web import web_search


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

    # deepagents auto-adds a `general-purpose` subagent (graph.py:751) built with
    # `"model": model` — the *root* model object, ROOT_TIMEOUT and all — plus the
    # unrestricted default filesystem, `execute` included. Both halves are wrong here:
    #
    # * ROOT_TIMEOUT's `read=10.0` is an inter-chunk watchdog, safe only because the root
    #   streams. A subagent's inner agent calls non-streaming, so the same 10s becomes a
    #   ceiling on the whole response. In trace 01a04aca-c0cf-7a21-9db6-ae9180cefcd0 that
    #   was three attempts at ~10s, then APITimeoutError — 31.7s spent to fail a
    #   ClinicalTrials.gov triage `trial-analyst` would have finished.
    # * The unrestricted toolset is the same hazard `analyst_leaf` exists to close.
    #
    # Nothing here should ever route to it: over the project's whole history it has taken
    # 106 dispatches against `trial-analyst`'s zero, and the ctgov work it absorbed is
    # exactly what the leaves are for.
    #
    # Registration is keyed by provider or `provider:model`, and we hand `create_deep_agent`
    # a pre-built model, so the lookup falls back to the bare provider key. Both providers
    # are registered because ROOT_PROVIDER is an env axis (models.py:ENV_VARS) and a
    # profile keyed to today's default would silently stop applying after a swap. The
    # registry is process-global, so this also covers any other deepagents agent built on
    # these providers in the same process — `graph.py` builds only ours.
    #
    # The `task` tool survives this: it disappears only when no synchronous subagents
    # remain, and the four leaves below are synchronous.
    _NO_GENERAL_PURPOSE = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    for _provider in ("openai", "anthropic"):
        register_harness_profile(_provider, _NO_GENERAL_PURPOSE)

    agent = create_deep_agent(
        model=root_model(),
        # Two wrappers, both because the PTC bridge calls `tool.arun` directly and no
        # middleware hook sees a call made inside `eval`. Inner: each call emits a line the
        # frontend can show (`middleware/progress.py`). Outer: a source failure returns
        # `{error}` instead of raising, since a raise here ends the whole run rather than
        # the one call (`middleware/tool_errors.py`). Both are the same tools in every
        # respect PTC inspects.
        tools=with_error_capture(with_progress([
            pubmed_search,
            fetch_abstracts,
            pmc_locate,
            fetch_full_text,
            fetch_figures,
            fetch_supplementary,
            ctgov_search,
            ctgov_fetch,
            web_search,
        ])),
        system_prompt=build_system_prompt(),
        subagents=[
            analyst_leaf(ABSTRACT_ANALYST),
            analyst_leaf(FULL_TEXT_ANALYST),
            analyst_leaf(FIGURE_ANALYST),
            analyst_leaf(TRIAL_ANALYST),
        ],
        backend=backend,
        middleware=[
            # First, because its `before_agent` is what puts the user's own files on disk
            # under /workspace/uploads before any tool can look for them — and because it
            # strips the upload payload out of the human message, which has to happen
            # before the first model call rather than before the first tool call.
            UploadMiddleware(backend),
            # Nothing below the tool boundary can prompt the model — one `eval` runs for
            # minutes with no turn boundary in it — so this reports the silence at the
            # next moment the model can actually speak. See middleware/cadence.py.
            UpdateCadence(),
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
                    # Provider-side search, spent inside the tool so its pages land in
                    # the JS heap instead of root context. `sources/web.py` says why
                    # that indirection exists; binding the provider's own search tool to
                    # the root model here would undo it.
                    "web_search",
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
