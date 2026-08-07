"""Server entry point: the same agent as `agent.py`, exposed over LangGraph.

`agent.py` runs one question and exits, so it can own a sandbox for the length of a
`with` block. A server can't — a thread is a conversation, the user comes back to it,
and the second turn has to see the files the first turn wrote. So the sandbox is keyed
to the thread and reused across turns.

Lifetime is handled by the server side rather than by us: `idle_ttl_seconds` deletes a
sandbox that has gone quiet, which is the only cleanup that survives the process being
killed. We never explicitly delete one, because "the user might come back to this
thread" is true right up until it isn't.

Point a chat UI at this:

    uv run langgraph dev                     # this graph, on :2024
    npx create-agent-chat-app                # UI, on :5173 -> deployment http://localhost:2024
"""

from __future__ import annotations

import asyncio
import re

from deepagents.backends import LangSmithSandbox
from langchain_core.runnables import RunnableConfig
from langsmith.sandbox import SandboxClient

from agent import IDLE_TTL_SECONDS, SNAPSHOT_NAME, build_agent, find_snapshot, provision

_client = SandboxClient()

# thread_id -> sandbox name. The sandbox itself is looked up fresh each turn, because
# the TTL may have reaped it while the user was away and a cached handle would then be
# pointing at nothing.
_sandbox_names: dict[str, str] = {}


def _sandbox_name(thread_id: str) -> str:
    """A stable, name-safe sandbox id derived from the thread id."""
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", thread_id)[:40].strip("-")
    return f"pubmed-{slug or 'default'}"


def _acquire(thread_id: str):
    """Return this thread's sandbox, rebooting it if the TTL already reaped it."""
    name = _sandbox_names.get(thread_id) or _sandbox_name(thread_id)

    try:
        sandbox = _client.get_sandbox(name)
    except Exception:  # noqa: BLE001 - absent, expired, or never created
        sandbox = None

    if sandbox is None:
        snapshot = find_snapshot(_client)
        sandbox = _client.create_sandbox(
            snapshot_name=snapshot,
            name=name,
            idle_ttl_seconds=IDLE_TTL_SECONDS,
        )
        if snapshot is None:
            # No snapshot: pay the ~30s install once for this thread's sandbox.
            print(
                f"[sandbox] no snapshot named {SNAPSHOT_NAME!r} — installing at "
                "runtime. Run `uv run build_snapshot.py` once to skip this."
            )
            provision(sandbox)

    _sandbox_names[thread_id] = name
    return sandbox


class _UnboundSandbox:
    """Placeholder for a graph that is being inspected rather than run.

    Studio draws its diagram by calling the graph factory and reading the compiled
    structure, which is identical for every thread. Handing it a real sandbox would
    bill a container per page load, so it gets this instead — the shape of the graph
    doesn't depend on the backend, and nothing here is ever called.
    """

    def __getattr__(self, name: str):
        msg = (
            f"sandbox is unbound (tried to call .{name}). This graph was built for "
            "inspection, without a thread_id — it can't run tools."
        )
        raise RuntimeError(msg)


async def make_graph(config: RunnableConfig):
    """Build the agent bound to this thread's sandbox.

    Called per run by the LangGraph server, which is what lets the sandbox — and so
    the files in `/workspace/out` — follow the thread rather than the process.

    Acquiring a sandbox is synchronous HTTP. Under `langgraph dev` that runs inside the
    server's event loop, where blockbuster raises `BlockingError` on any blocking
    socket call, so it has to go through a worker thread. Keeping `_acquire` itself
    synchronous also keeps it usable from the CLI, which has no loop to protect.
    """
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if not thread_id:
        return build_agent(LangSmithSandbox(sandbox=_UnboundSandbox()))

    sandbox = await asyncio.to_thread(_acquire, str(thread_id))
    return build_agent(LangSmithSandbox(sandbox=sandbox))
