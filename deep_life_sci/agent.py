"""Agent assembly. Nothing here boots a sandbox or runs a question.

`build_agent` takes a backend rather than making one, so the same assembly serves three
sandbox lifetimes: `cli.py`'s single block, `graph.py`'s thread-keyed container, and
`evals/`'s per-example throwaway.
"""

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_quickjs import CodeInterpreterMiddleware

from deep_life_sci.middleware.artifacts import ArtifactMiddleware
from deep_life_sci.middleware.cadence import UpdateCadence
from deep_life_sci.middleware.model_errors import ModelSettingErrors
from deep_life_sci.middleware.perf import LoopLagProbe
from deep_life_sci.middleware.progress import with_progress
from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.middleware.uploads import UploadMiddleware
from deep_life_sci.models import root_model, subagent_model
from deep_life_sci.prompts import (
    ABSTRACT_ANALYST,
    DOCUMENT_ANALYST,
    FIGURE_ANALYST,
    FULL_TEXT_ANALYST,
    TRIAL_ANALYST,
    build_system_prompt,
)
from deep_life_sci.sources.ctgov import ctgov_search
from deep_life_sci.sources.pmc import fetch_full_text, make_sandbox_tools, pmc_locate
from deep_life_sci.sources.pubmed import fetch_abstracts, pubmed_search
from deep_life_sci.sources.trial_files import make_trial_fetch
from deep_life_sci.sources.web import web_search


def build_agent(backend):
    """Assemble the agent against a backend. In practice the backend is the sandbox."""
    # Built per-run: these upload bytes into the sandbox, so an image exists as a real
    # file on a real path before a subagent can read_file it.
    fetch_figures, fetch_supplementary = make_sandbox_tools(backend)
    ctgov_fetch = make_trial_fetch(backend)

    def analyst_leaf(spec: dict) -> dict:
        """Read-only leaves; the trial analyst can also search staged sections.

        `tools: []` drops the parent's tools. It does not drop deepagents' own prepended
        FilesystemMiddleware, which includes `execute` — a shell into the shared sandbox.
        Passing a configured instance substitutes for the default rather than stacking,
        and `read_file` is the floor it refuses to drop.
        """
        filesystem_tools = ["read_file"]
        if spec["name"] == "trial-analyst":
            filesystem_tools.append("grep")
        return {
            **spec,
            "model": subagent_model(),
            "tools": [],
            "middleware": [
                FilesystemMiddleware(backend=backend, tools=filesystem_tools),
                ModelSettingErrors("subagent"),
            ],
        }

    # Disable deepagents auto-added `general-purpose` subagent
    _NO_GENERAL_PURPOSE = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    for _provider in ("openai", "anthropic"):
        register_harness_profile(_provider, _NO_GENERAL_PURPOSE)

    agent = create_deep_agent(
        model=root_model(),
        # The PTC bridge calls `tool.arun` directly, so no middleware hook sees a call
        # made inside `eval`. Inner: progress lines. Outer: source failures return
        # `{error}` instead of raising, since a raise ends the run, not the call.
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
            # Reads a path rather than being handed text, like figure-analyst and for the
            # same reason: an uploaded PDF's extracted text is a file in the sandbox
            # (`middleware/upload_probe.py`) and far too large for a task description.
            analyst_leaf(DOCUMENT_ANALYST),
        ],
        backend=backend,
        middleware=[
            # First: its `before_agent` strips the upload payload out of the human
            # message, which must happen before the first model call.
            UploadMiddleware(backend),
            # A refused model call names the models.yaml setting to fix.
            ModelSettingErrors("root"),
            UpdateCadence(),
            CodeInterpreterMiddleware(
                # Tools reach JS camelCased: pubmed_search -> tools.pubmedSearch.
                ptc=[
                    "pubmed_search",
                    "fetch_abstracts",
                    "pmc_locate",
                    "fetch_full_text",
                    "fetch_figures",
                    "fetch_supplementary",
                    "ctgov_search",
                    "ctgov_fetch",
                    # Provider-side search, spent inside the tool so its pages land in the
                    # JS heap instead of root context. Binding the provider's own search
                    # tool to the root model would undo that.
                    "web_search",
                    "execute",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "ls",
                    "glob",
                ],
                # The 5s default kills every real fan-out.
                timeout=900.0,
                max_result_chars=40_000,
                max_ptc_calls=512,
            ),
            # After the interpreter: it sweeps once the tool call it wraps has returned,
            # and `eval` does most of the writing to /workspace/out.
            ArtifactMiddleware(backend),
            # Last, so its wall time is the `eval` itself rather than the artifact sweep.
            LoopLagProbe(),
        ],
    )
    # A ceiling against a runaway loop, not a tuning knob: a large fan-out exceeds
    # LangGraph's default 25 super-steps well before anything is wrong.
    return agent.with_config(recursion_limit=200)
