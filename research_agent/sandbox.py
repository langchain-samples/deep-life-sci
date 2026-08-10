"""The sandbox: getting one, and keeping it alive under a flaky socket.

Two concerns, deliberately in one module because they are two halves of one question —
"is there a working container behind this handle?"

1. **Lifecycle** (`find_snapshot`, `provision`, `sandbox_session`). Three callers need a
   sandbox and each needs it differently: `cli.py` owns one for a `with` block,
   `graph.py` keys one to a `thread_id` and reuses it across turns, and `evals/` wants a
   disposable one per example. That used to live in `agent.py`, which meant the server
   imported the CLI to boot a container. It lives here now so all three import the same
   thing and nobody imports an entry point.
2. **Resilience** (`ResilientSandbox`). Retrying transport failures below the tool
   boundary, for the reasons the next paragraphs give.

Every async filesystem operation in `deepagents` — `read_file`, `ls`, `write_file`,
`edit_file` — funnels through `LangSmithSandbox.aexecute`, which opens a WebSocket to
the sandbox dataplane. When that upgrade is rejected the SDK raises
`SandboxConnectionError`, and the exception propagates all the way out of the `eval`
tool as a string the model has to read and reason about.

That is the wrong layer to handle it. Observed in trace
`019fde6d-d267-70f0-924b-e0cccae622be`: an `eval` that had already fanned out to 18
`abstract-analyst` subagents and collected all 18 answers was killed by a single
`HTTP 502` on a WebSocket upgrade, 15.5s in. Every answer was discarded. The model
then spent five more turns re-deriving state it had lost — `1 + 1` to probe whether
the interpreter was alive, then a hardcoded PMID list because it no longer trusted
its own variables — and re-ran the same fan-out three more times. Roughly 46s of a
101s run.

A rejected upgrade is transient: the dataplane is recycling, or the router briefly has
no healthy backend. The SDK treats plain `SandboxConnectionError` as permanent (only
`SandboxConnectTimeoutError` is retried internally) because it cannot know whether the
command was already accepted. This wrapper takes the other side of that trade,
deliberately:

**Retried operations are assumed idempotent.** For this agent they are — the
filesystem tools are read/write/list against a workspace, and `execute` runs stats and
plots that are safe to re-run. If you add a tool whose sandbox command must run
exactly once (appending to a file, incrementing a counter, POSTing somewhere), route
it around this wrapper. The alternative is what the trace shows: throwing away minutes
of completed work because a socket blinked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from deepagents.backends import LangSmithSandbox
from deepagents.backends.protocol import ExecuteResponse
from langsmith.sandbox import SandboxClient, SandboxConnectionError

from research_agent.paths import WORKSPACE

logger = logging.getLogger(__name__)

# The sandbox is deleted on context exit, but a `finally` doesn't survive SIGKILL — and
# billing doesn't care why the process died. This is the server-side backstop.
IDLE_TTL_SECONDS = 600

# The default sandbox image is bare Python 3.12 — no numpy/pandas/scipy/matplotlib —
# and installing them costs ~30s. `scripts/build_snapshot.py` bakes them into a named
# snapshot instead; booting from it takes ~1s. Run that once, then every run starts warm.
SNAPSHOT_NAME = os.environ.get("SANDBOX_SNAPSHOT_NAME", "pubmed-py")

# Fallback for a clone that hasn't built the snapshot yet. Same package set, paid per run.
PROVISION = (
    f"mkdir -p {WORKSPACE}/out && "
    "pip install --break-system-packages --quiet "
    "numpy pandas scipy matplotlib openpyxl python-docx python-pptx 2>&1 | tail -2"
)

# Four attempts over ~7s of backoff. A dataplane that is recycling comes back inside
# that window; one that is genuinely gone will not come back in any window worth
# blocking a tool call for.
DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 4.0

# `execute()`/`aexecute()` is the one place every shell command the model can issue
# passes through (see class docstring), which makes it the only place a package
# install can be blocked for good instead of "for as long as the model reads the
# prompt telling it not to." Observed in trace 019fe937-1bf6-7d61-9ca8-658c53dfd8ae:
# the model tried `pip install openpyxl`, hit PEP 668's externally-managed-environment
# error, retried with `--break-system-packages`, and only then got what
# `build_snapshot.py` was already supposed to have baked in. A stale or incomplete
# snapshot should fail loudly and tell the model to say so — not send it down a
# pip-install detour that burns a network round trip and can silently drift the
# sandbox away from the pinned versions in `build_snapshot.py`.
_INSTALL_COMMAND_RE = re.compile(
    r"(?i)\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|easy_install|"
    r"conda\s+install|mamba\s+install|apt(?:-get)?\s+install)\b"
)

_INSTALL_BLOCKED_MESSAGE = (
    "Blocked: installing packages at runtime is disabled. This sandbox is "
    "pre-provisioned (numpy, pandas, scipy, matplotlib, openpyxl, python-docx, "
    "python-pptx) by build_snapshot.py — use one of those instead of installing a "
    "substitute. If the task genuinely needs something outside that list, say so "
    "in your final answer rather than trying to install it."
)


def _blocked_execute_response() -> ExecuteResponse:
    return ExecuteResponse(
        output=f"{_INSTALL_BLOCKED_MESSAGE}\n\n[Command failed with exit code 1]",
        exit_code=1,
    )


class ResilientSandbox(LangSmithSandbox):
    """A `LangSmithSandbox` that retries connection failures instead of raising.

    Args:
        sandbox: The LangSmith sandbox to wrap.
        reacquire: Optional zero-argument callable returning a fresh `Sandbox` for
            this thread. Called (in a worker thread — it is synchronous HTTP) once
            the first retry has failed, which is the point at which the likely cause
            stops being "the socket blinked" and becomes "the container is gone and
            the TTL reaped it". `graph.py` passes its thread-keyed `_acquire`; the
            CLI has nothing to re-acquire and passes nothing.
        attempts: Total tries per operation, including the first.
        base_delay: Seconds before the first retry; doubles each attempt.
        max_delay: Ceiling on the backoff.
    """

    def __init__(
        self,
        sandbox: Any,
        *,
        reacquire: Callable[[], Any] | None = None,
        attempts: int = DEFAULT_ATTEMPTS,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> None:
        super().__init__(sandbox)
        self._reacquire = reacquire
        self._attempts = max(1, attempts)
        self._base_delay = base_delay
        self._max_delay = max_delay
        # A fan-out puts many operations in flight at once, and a dead dataplane
        # fails all of them together. Without this lock, 18 concurrent read_files
        # would each independently re-acquire the sandbox and the last writer would
        # win — 18 containers requested to replace one.
        self._rebind_lock = asyncio.Lock()
        self._rebind_generation = 0

    async def _arebind(self, seen_generation: int) -> bool:
        """Swap in a freshly acquired sandbox. Returns whether one is now bound.

        `seen_generation` is the generation the caller observed before it failed.
        If it has already moved, another coroutine in the same fan-out rebound while
        this one was waiting on the lock, and this caller should just retry against
        the new sandbox rather than acquire a second one.
        """
        if self._reacquire is None:
            return False

        async with self._rebind_lock:
            if self._rebind_generation != seen_generation:
                return True

            try:
                fresh = await asyncio.to_thread(self._reacquire)
            except Exception:  # noqa: BLE001 - re-acquire failing is not fatal; retry the old handle
                logger.warning("sandbox re-acquire failed; retrying existing handle", exc_info=True)
                return False

            stale_client = self._async_client
            # Drop the cached async sandbox/client so `_aget_sandbox` rebuilds them
            # against the new container instead of the dead one's connection pool.
            self._async_sandbox = self._async_client = None
            self._sandbox = fresh
            self._rebind_generation += 1

            if stale_client is not None:
                try:
                    await stale_client.aclose()
                except Exception:  # noqa: BLE001 - the pool is already unusable
                    logger.debug("closing stale sandbox client failed", exc_info=True)

            logger.warning("re-acquired sandbox %r after connection failure", getattr(fresh, "name", "?"))
            return True

    async def aexecute(self, command: str, *, timeout: int | None = None) -> Any:  # noqa: ASYNC109
        """Run a command, retrying transport failures.

        This is the single choke point for the async filesystem tools — `als`,
        `aread`, `awrite` and `aedit` all delegate here — so wrapping it covers
        every sandbox operation a subagent or the interpreter can reach. It is also
        where package installs are blocked; see `_INSTALL_COMMAND_RE`.
        """
        if _INSTALL_COMMAND_RE.search(command):
            return _blocked_execute_response()
        delay = self._base_delay
        for attempt in range(1, self._attempts + 1):
            generation = self._rebind_generation
            try:
                return await super().aexecute(command, timeout=timeout)
            except SandboxConnectionError as exc:
                if attempt == self._attempts:
                    logger.error("sandbox unreachable after %d attempts: %s", attempt, exc)
                    raise
                logger.warning(
                    "sandbox connection failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self._attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_delay)
                # First failure is treated as a blip and simply retried. From the
                # second on, assume the container itself is gone.
                if attempt >= 2:
                    await self._arebind(generation)
        # Unreachable: the loop either returns or raises on the final attempt.
        raise AssertionError("aexecute retry loop exited without result")

    def execute(self, command: str, *, timeout: int | None = None) -> Any:
        """Synchronous `execute` with the same retry policy.

        Used by `build_snapshot.py` and by provisioning, which run before there is
        an event loop. Re-acquire is deliberately not attempted here: the sync path
        runs at startup, where a failure is better surfaced loudly than papered over.

        Note: `scripts/build_snapshot.py` and `provision()` below call `sandbox.run()`
        on the raw (unwrapped) sandbox directly, not this method, so the install guard
        below never blocks the provisioning step that's supposed to install packages.
        """
        if _INSTALL_COMMAND_RE.search(command):
            return _blocked_execute_response()
        delay = self._base_delay
        for attempt in range(1, self._attempts + 1):
            try:
                return super().execute(command, timeout=timeout)
            except SandboxConnectionError as exc:
                if attempt == self._attempts:
                    raise
                logger.warning(
                    "sandbox connection failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self._attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, self._max_delay)
        raise AssertionError("execute retry loop exited without result")


def find_snapshot(client: SandboxClient) -> str | None:
    """Return SNAPSHOT_NAME if that snapshot exists, else None.

    A failed lookup falls back to installing at runtime rather than killing the run.
    """
    try:
        snapshots = client.list_snapshots(name_contains=SNAPSHOT_NAME)
    except Exception as exc:  # noqa: BLE001 - a missing snapshot is not a fatal error
        print(f"[sandbox] snapshot lookup failed ({exc}); will install at runtime")
        return None
    return next((s.name for s in snapshots if s.name == SNAPSHOT_NAME), None)


def provision(sandbox: Any) -> None:
    """Install the scientific Python stack into a sandbox that didn't ship with it."""
    t0 = time.monotonic()
    result = sandbox.run(PROVISION, timeout=600)
    if result.exit_code != 0:
        msg = f"sandbox provisioning failed (exit {result.exit_code}): {result.stderr}"
        raise RuntimeError(msg)
    print(f"[sandbox] python ready in {time.monotonic() - t0:.1f}s")


@asynccontextmanager
async def sandbox_session(*, quiet: bool = False):
    """Yield a `ResilientSandbox` for the life of the block, then delete the container.

    This is the single-owner case: one caller, one sandbox, torn down on exit. `cli.py`
    uses it for a run, and `evals/` uses it per example — an evaluator scoring twenty
    questions must not let question three's leftover `/workspace/out` files be swept and
    published during question four, which is exactly what sharing one container does.

    `graph.py` does *not* use this. A server thread outlives any block: the user comes
    back to it and the second turn has to see the first turn's files, so there the
    sandbox is keyed to `thread_id` and reaped by `idle_ttl_seconds` instead.

    No `reacquire` is wired here for the same reason: a single-owner caller has nothing
    to hand back a fresh container to mid-block. Retries still cover a blinking socket,
    which is the common case.

    Async because closing is: `ainvoke` lazily builds a cached async client with its own
    connection pool, and `aclose()` is the only thing that closes it. Making that the
    context manager's job rather than every caller's is the point — the CLI got it right
    by hand, and an eval loop that leaked one pool per example would not have.

    Acquiring and deleting the container are blocking HTTP, done here on the calling
    thread. That is fine for the standalone processes this serves and is why `graph.py`,
    which runs inside an ASGI loop that blockbuster polices, goes through
    `asyncio.to_thread` instead of using this.

    Args:
        quiet: Suppress the boot/provision progress lines. Evals run many of these and
            the per-example noise buries the scores.
    """
    client = SandboxClient()
    snapshot = find_snapshot(client)
    if snapshot is None and not quiet:
        print(
            f"[sandbox] no snapshot named {SNAPSHOT_NAME!r} — installing at runtime "
            "(~30s). Run `uv run scripts/build_snapshot.py` once to skip this."
        )

    t0 = time.monotonic()
    with client.sandbox(snapshot_name=snapshot, idle_ttl_seconds=IDLE_TTL_SECONDS) as sandbox:
        if not quiet:
            print(f"[sandbox] up in {time.monotonic() - t0:.1f}s ({snapshot or 'base image'})")
        if snapshot is None:
            provision(sandbox)
        backend = ResilientSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            await backend.aclose()
