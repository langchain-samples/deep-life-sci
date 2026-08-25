"""Server entry point: the same agent as `cli.py`, exposed over LangGraph.

`cli.py` runs one question and exits, so it can own a sandbox for the length of a
`with` block. A server can't — a thread is a conversation, the user comes back to it,
and the second turn has to see the files the first turn wrote. So the sandbox is keyed
to the thread and reused across turns.

Lifetime is handled by the server side rather than by us: `idle_ttl_seconds` deletes a
sandbox that has gone quiet, which is the only cleanup that survives the process being
killed. We never explicitly delete one, because "the user might come back to this
thread" is true right up until it isn't.

Point a chat UI at this — `uv run scripts/dev.py` starts both halves, or by hand:

    uv run langgraph dev                     # this graph, on :2024
    cd ../agent-chat-ui && pnpm dev          # UI, on :3000 -> http://localhost:2024

The UI must serve `/ui/*` from its own origin — a `next.config.mjs` rewrite to :2024.
`/ui/{graph}` hands back a script tag with a host-relative `src`, so the browser resolves
it against the page, not the API. Cross-origin (hosted Agent Chat, Studio) it 404s and
`ArtifactMiddleware`'s components render as an empty div with the run otherwise intact.

**Clients should request `streamMode: ["messages", "updates"]`.** Synthesis is the
longest single span in a run — 23-32s in the traces from thread
019fde6d-d25c-77b3-a751-56c6b7aa4ead — and with `updates` alone the answer appears all
at once when that span ends. `messages` streams it token by token instead, at a measured
1.6s to first token. The run takes the same time either way; the wait stops being blank.
`updates` is worth keeping alongside it for the tool-call and `ui` artifact events.
"""

from __future__ import annotations

import asyncio
import re

from langchain_core.runnables import RunnableConfig
from langsmith.sandbox import SandboxClient

from research_agent.agent import build_agent
from research_agent.middleware.perf import install_logging
from research_agent.paths import IDLE_TTL_SECONDS
from research_agent.sandbox import (
    SNAPSHOT_NAME,
    ResilientSandbox,
    find_snapshot,
    provision,
)
from research_agent.sources import cache_io

install_logging()

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
            # No snapshot: pay the ~95s install once for this thread's sandbox.
            print(
                f"[sandbox] no snapshot named {SNAPSHOT_NAME!r} — installing at "
                "runtime. Run `uv run scripts/build_snapshot.py` once to skip this."
            )
            provision(sandbox)

    _sandbox_names[thread_id] = name
    return sandbox


class _UnboundSandbox:
    """Placeholder for a graph that is being read rather than run.

    The server calls the factory for reads as well as runs — Studio drawing its
    diagram, the chat UI listing a thread's history — and the compiled structure it
    wants back is identical for every thread. Handing those a real sandbox would bill
    a container per page load, so they get this instead: the shape of the graph doesn't
    depend on the backend, and nothing here is ever called.
    """

    def __getattr__(self, name: str):
        msg = (
            f"sandbox is unbound (tried to call .{name}). This graph was built to be "
            "read rather than run — no thread_id, or not a run — so it can't run tools."
        )
        raise RuntimeError(msg)


async def make_graph(config: RunnableConfig):
    """Build the agent bound to this thread's sandbox.

    Called per request by the LangGraph server — per *run* is what binds a sandbox, and
    that is what lets it, and so the files in `/workspace/out`, follow the thread rather
    than the process. Reads get an unbound one; see below.

    Acquiring a sandbox is synchronous HTTP. Under `langgraph dev` that runs inside the
    server's event loop, where blockbuster raises `BlockingError` on any blocking
    socket call, so it has to go through a worker thread. Keeping `_acquire` itself
    synchronous also keeps it usable from the CLI, which has no loop to protect.
    """
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    # `langgraph_api` sets this on every factory call, True only for `threads.create_run`.
    # A read — `/threads/{id}/history`, `/threads/{id}/state`, Studio's diagram — has a
    # thread_id like a run does, so without this the sandbox lookup below runs on every
    # one: 470ms of synchronous HTTP to render a transcript that needs no container at
    # all, and, if the TTL already reaped the thread's sandbox, `_acquire` boots a fresh
    # one just to page back through an old conversation.
    #
    # Absent is treated as a run, not a read. The key is private to the server and may
    # move between releases; the failure that guesses wrong in that direction is a
    # `_UnboundSandbox` reaching a real run and raising on the first tool call.
    is_read = configurable.get("__is_for_execution__") is False
    if not thread_id or is_read:
        return build_agent(ResilientSandbox(sandbox=_UnboundSandbox()))

    key = str(thread_id)
    # Self-gated to once per TTL window, so this is a no-op on all but one turn in ten
    # minutes. This is the long-lived process of the three, and the only one where an
    # unbounded cache would accumulate across many threads rather than one run.
    await cache_io.sweep_if_due()
    sandbox = await asyncio.to_thread(_acquire, key)
    # `_acquire` is both the initial lookup and the recovery path: it re-creates the
    # sandbox under the same thread-derived name if the container is gone. Handing it to
    # the backend lets a connection failure mid-run be repaired without the model ever
    # seeing an error string — which is exactly what it could not do in trace
    # 019fde6d-d267-70f0-924b-e0cccae622be, where one 502 cost ~46s of a 101s run.
    #
    # Files written before the container died do not come back. That is a real loss, but
    # a strictly smaller one than failing the tool call: /workspace/out is swept and
    # published after every writing call, so anything already delivered is already out.
    return build_agent(ResilientSandbox(sandbox=sandbox, reacquire=lambda: _acquire(key)))
