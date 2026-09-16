# Repository guidance

Instructions for any coding agent working in this repository. Keep this file under
200 lines: retain commands, project-specific constraints, and pitfalls; put detailed
rationale in module docstrings. `CLAUDE.md` is a symlink to this file.

## This is the `engine-demo` branch — do not fix the planted bug

`models.py:WEB_SEARCH_SPECS` folds Anthropic's `name` and `max_uses` keys into the OpenAI
spec, so every `web_search` call 400s and comes back contained as an empty digest. That is
**deliberate**: it is the failure this branch exists to demonstrate to LangSmith Engine.
Read `ENGINE_WORKSHOP.md` before touching `models.py`, `sources/web.py`,
`scripts/engine_demo.py` or `engine_workshop/`. Never merge this branch to `main`.

## Project and local guidance

A life-science research assistant built with Deep Agents: PubMed/PMC literature,
ClinicalTrials.gov, web search, specialist subagents, and sandboxed analysis.

- Read `tests/test_invariants.py` for contracts that span multiple files.
- Before changing source clients, read `deep_life_sci/sources/CLAUDE.md`.
- Before changing setup or the chat stack, read `scripts/CLAUDE.md`; for overlay
  components, also read `chat-ui-overlay/CLAUDE.md`. These files apply to any coding agent.
- Keep `README.md` focused on human setup. Module docstrings explain implementation;
  avoid duplicating them here. See `tests/__init__.py` and `evals/README.md` for test scope.

## Commands

Run from the repository root:

```bash
uv run scripts/setup.py                # initial setup: environment, deps, snapshot, UI
uv run scripts/dev.py                  # start API and chat UI; NO_BROWSER=1 skips browser
uv run agent ["question"]              # one-shot CLI; no question uses the demo
uv run scripts/build_snapshot.py       # rebuild the scientific Python sandbox image
uv run langgraph dev                   # API only, port 2024
uv run --group test pytest             # offline tests
uv run --group test pytest tests/test_pubmed.py  # example focused check
uv run ruff check .                    # repository lint configuration in pyproject.toml
uv run python -m evals.run --structural --limit 3  # live evaluation, no judge model
uv run python -m evals.sync             # publish dataset seeds to LangSmith
```

- Run tests relevant to the change and lint. Runtime changes should pass the full
  offline suite; support both Python 3.12 and 3.13. Do not add live calls to `tests/`.
- `tests/` verifies code; `evals/` measures research quality using external services.
  Keep evaluation code outside the deployed package and invoke its entry points with `-m`.
- Follow `pyproject.toml` for style. Do not reflow or strip whitespace from prompt strings:
  JavaScript examples and Markdown line breaks can be semantically significant.

## Configuration and setup

- `scripts/setup.py` owns `.env` writes. Add a setup prompt and an `.env.example` entry
  for any new required setting. `scripts/dev.py` verifies setup; starting the app must
  not trigger dependency installation.
- The gateway authenticates with a LangSmith key. Provider credentials belong in the
  workspace's provider integrations. See `models.py:gateway_key()` for precedence.
- Model roles and environment axes live in `models.py:ENV_VARS`. Entry points import
  that list; do not copy it. Preserve explicitly empty values across dotenv loading:
  empty effort disables the parameter for models that do not support it.
- Keep model/provider validation and provider-specific routing in `models.py`. Search
  models must support the provider's web-search tool. Root streaming and its timeout
  belong together: the read timeout measures gaps between streamed chunks.
- Define host paths in `paths.py`, anchored to the repository root. Preserve
  `DEEP_LIFE_SCI_DATA_DIR` overrides and scope cache sweeps to the named cache roots.
- `.chat-ui/` is an ignored upstream clone patched by setup. Make persistent UI changes
  in `chat-ui-overlay/` or the setup patches, not only in the generated clone.

## Agent and tool contracts

- All entry points share `agent.py:build_agent(backend)`. Assembly must not acquire a
  sandbox or perform external I/O; callers own sandbox lifetime.
- `eval` is QuickJS orchestration, with no filesystem, shell, or network of its own.
  `execute` runs real Python/shell commands in the LangSmith sandbox.
- Adding a source tool requires assembly registration, the `ptc` allowlist, its camelCase
  usage in `prompts/system.py`, and progress/error wrapping. Preserve the interpreter's
  explicit limits; default timeouts are too short for research fan-outs.
- Contain expected source/transport failures as `{error}` through `with_error_capture`.
  Exceptions escaping the PTC bridge can end the entire run. Keep programming errors
  visible rather than disguising them as empty results.
- Build figure and supplementary tools per backend with `make_sandbox_tools`: returned
  paths must refer to bytes actually staged in that sandbox.
- Batch-fetch before dispatching analyst subagents; leaves must not fetch independently.
  Set `tools: []` and narrow their filesystem middleware to `read_file`. Empty middleware
  alone does not disable inherited filesystem tools. Keep the general-purpose leaf disabled.
- Treat prompts as production code. Keep instructions concise and general; explain
  pitfalls instead of adding unnecessary procedures or long tutorials.

## Context and file boundaries

- Keep large source payloads in QuickJS or analyst contexts. Use `pmc_locate` to triage
  before full-text retrieval. Root-visible output should summarize results, not dump them.
- Web search belongs in the search-role PTC tool. Binding provider search directly to
  the root model places retrieved pages in root context.
- Host `data/` is a cache, not a filesystem the research agent can access. Materialize
  required files in the sandbox explicitly.
- `/workspace/out/` contains user deliverables, published by `ArtifactMiddleware` through
  the `ui` state channel. Preserve the prompt's prohibition on reading deliverables back
  into root context. Record artifact fingerprints only after successful publication.
- Uploads live under `/workspace/uploads/`, outside deliverables. Strip attachment bytes
  before the first model call; keep durable copies per thread in the LangGraph store.
  Manifests carry shapes, identifiers, and sidecar paths, not document bodies.
- An upload format needs entries in `UPLOAD_KINDS`, the probe dispatcher, the root prompt,
  and the composer allowlist. New reader libraries also require a snapshot rebuild.
- Keep common upload probes on the standard library to avoid cold import latency.
  PDF, GenBank, and fallback image decoding use lazy library imports; keep those local
  to their format-specific paths. New attachments must overwrite same-name sandbox files
  even when their byte counts match.

## Lifecycle and deployment pitfalls

- Graph runs reuse a sandbox by `thread_id`. Graph reads, including reads with a thread
  ID, use `_UnboundSandbox`; inspecting history must not boot a container.
- Blocking filesystem and SDK calls in async paths must run through `asyncio.to_thread`.
  Otherwise they stall concurrent runs and can fail under the development server.
- Preserve per-source rate limits and API guards; they prevent silent wrong answers.
  Read their rationale in the source module before changing them.
- Cache expiry and sandbox idle lifetime share `paths.py:IDLE_TTL_SECONDS`. Evaluation
  runs can disable cache expiry with `DEEP_LIFE_SCI_CACHE_TTL=off`.
- `ResilientSandbox` retries assume idempotent commands. Route operations that must run
  exactly once around the retry wrapper. Keep cleanup on failed boots and session exits.
- Provision scientific libraries through the snapshot. Without one, startup provisioning
  is slower; model-issued runtime installs remain blocked.
- Artifact UI scripts use host-relative URLs. Frontends must proxy `/ui/*` through their
  own origin. Check `/ui/<graph>/entrypoint.js` when artifact cards fail to render.
- Thread lists must request metadata projections rather than full state: QuickJS snapshots
  can make each thread response megabytes. Preserve the `patch_thread_search` behavior.
- QuickJS middleware and dynamic subagent APIs are beta. Exercise the real compiled-agent
  integration test when changing their wiring or upgrading those dependencies.
