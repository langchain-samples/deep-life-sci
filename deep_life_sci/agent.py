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
from deepagents._models import get_model_provider
from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from langchain_quickjs import CodeInterpreterMiddleware

from deep_life_sci.middleware.artifacts import ArtifactMiddleware
from deep_life_sci.middleware.cadence import UpdateCadence
from deep_life_sci.middleware.model_errors import SurfaceModelErrors
from deep_life_sci.middleware.perf import LoopLagProbe
from deep_life_sci.middleware.progress import with_progress
from deep_life_sci.middleware.tool_errors import with_error_capture
from deep_life_sci.middleware.uploads import UploadMiddleware
from deep_life_sci.models import (
    CHAT_ROLES,
    report_web_search_problem,
    root_model,
    subagent_model,
    takes_system_midway,
    validate,
)
from deep_life_sci.paths import SKILLS_DIR
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
    # Every chat role, including search, which is otherwise first resolved inside a tool
    # call: a setting that cannot work fails here, before the run, naming what to fix.
    # A search model that cannot search is the exception: it takes out only web search,
    # so it is logged rather than refused, and each web search returns the reason.
    validate(*CHAT_ROLES)
    report_web_search_problem()
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
                SurfaceModelErrors("subagent"),
            ],
        }

    root = root_model()
    leaves = [
        analyst_leaf(ABSTRACT_ANALYST),
        analyst_leaf(FULL_TEXT_ANALYST),
        analyst_leaf(FIGURE_ANALYST),
        analyst_leaf(TRIAL_ANALYST),
        # Reads a path rather than being handed text, like figure-analyst and for the
        # same reason: an uploaded PDF's extracted text is a file in the sandbox
        # (`middleware/upload_probe.py`) and far too large for a task description.
        analyst_leaf(DOCUMENT_ANALYST),
    ]

    # Disable deepagents auto-added `general-purpose` subagent. Its profile is looked up by
    # the provider the built model reports, which is not always the gateway path's name:
    # Bedrock with the user's own keys reports "openai-mantle" or "anthropic-bedrock", and
    # a provider with no profile gets the general-purpose leaf back, sources and shell too.
    _NO_GENERAL_PURPOSE = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    built = (get_model_provider(model) for model in (root, *(leaf["model"] for leaf in leaves)))
    for _provider in {"openai", "anthropic", *filter(None, built)}:
        register_harness_profile(_provider, _NO_GENERAL_PURPOSE)

    # The source-specific prompt sections are skills, read on demand. They are package
    # files rather than sandbox files, so `/skills/` routes to the host and everything
    # else to the sandbox; the deny rule keeps the agent from editing them.
    skills_route = "/skills/"
    root_backend = CompositeBackend(
        default=backend,
        routes={skills_route: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )

    agent = create_deep_agent(
        model=root,
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
        subagents=leaves,
        backend=root_backend,
        skills=[(skills_route, "Research")],
        permissions=[
            FilesystemPermission(operations=["write"], paths=[f"{skills_route}**"], mode="deny")
        ],
        middleware=[
            # First: its `before_agent` strips the upload payload out of the human
            # message, which must happen before the first model call.
            UploadMiddleware(backend),
            # A gateway error reaches the user in the provider's words, not as "internal error".
            SurfaceModelErrors("root"),
            UpdateCadence(system_turns=takes_system_midway(root)),
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
