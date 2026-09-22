"""Workshop harness for running the ADLC loop on this agent with LangSmith Engine.

    uv run python -m engine_workshop.upload_traces    # replay the reference batch
    uv run python -m engine_workshop.download_traces  # re-capture that batch
    uv run python -m engine_workshop.eval             # score a dataset

A package with `-m` entry points rather than loose scripts, for the same reason `evals/`
is one: the repo root has to be on `sys.path` for `deep_life_sci` to import, and that is
what `-m` from the root gives you. Deliberately outside `deep_life_sci/` — this measures
and demonstrates the agent, it is not part of what the agent ships.

`scripts/engine_demo.py` is the third entry point and the only one that calls a model:
it drives the live agent over the workshop's prompts. Everything here works off saved
traces instead, so a room full of attendees gets the same clusters without a provider key
and without waiting on twelve sandboxes.
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta

# Where replayed traces land, and where Engine is pointed. Derived from whatever the
# healthy agent uses so the two never share a project — see ENGINE_WORKSHOP.md.
DEMO_SUFFIX = "-engine-demo"


def demo_project() -> str:
    """The workshop's tracing project: the base project plus `-engine-demo`.

    Resolved per call, not captured at import, because every entry point loads `.env`
    after importing this package — a module-level constant would read the environment one
    step too early and miss the file.
    """
    base = os.environ.get("LANGSMITH_PROJECT", "").strip() or "deep-life-sci"
    return base if base.endswith(DEMO_SUFFIX) else base + DEMO_SUFFIX


# How far back `query_runs` looks. `runs.query` defaults to the last day and rejects any
# window over 401 days, so every call names one; this is the widest the API allows.
QUERY_LOOKBACK = timedelta(days=400)


def query_runs(client, project_id, **filters) -> list:
    """Every run in `project_id` matching `filters`, via `client.runs.query`.

    The replacement for the deprecated `list_runs`, with two things the SDK leaves to the
    caller. `runs.query` is async even on the sync `Client` (by design, per the SmithDB
    migration guide), so the pages are drained under `asyncio.run` here. And the query
    defaults to the last 24 hours, which would miss a batch uploaded yesterday, so the
    window is always set explicitly.

    Returns the v2 `Run` model: ids are strings, select names are upper-case (`"ID"`), and
    there is no `parent_run_id` — the direct parent is the last of `parent_run_ids`.
    """
    filters.setdefault("min_start_time", datetime.now(UTC) - QUERY_LOOKBACK)

    async def drain() -> list:
        pages = client.runs.query(project_ids=[str(project_id)], page_size=1000, **filters)
        return [run async for run in pages]

    return asyncio.run(drain())
