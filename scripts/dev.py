"""Start the local stack: the agent server and the chat UI that renders its artifacts.

    uv run scripts/dev.py                    # both, logs interleaved and prefixed
    NO_BROWSER=1 uv run scripts/dev.py       # don't open a browser tab
    AGENT_CHAT_UI=~/src/acu uv run …         # a chat UI checkout of your own

Both halves are required. :2024 serves graph.py; :3000 serves the frontend, and is the
side carrying the `/ui/*` rewrite that lets artifact components load at all (see the
same-origin invariant in CLAUDE.md). Running only one gets you an API with no window, or a
window pointed at nothing.

A server already listening on its port is left alone and reused, so this is safe to run
alongside a `langgraph dev` you started yourself. Only what this script starts is what it
stops.

Python rather than bash because this is the one command Windows users cannot avoid, and
the three things it needs — a port check, a browser, and killing a process tree on Ctrl-C —
are exactly the three with no portable shell spelling.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import REPO_ROOT, chat_ui_dir, die, listening, require_setup, say, tool

UI_URL = "http://localhost:3000"
WINDOWS = os.name == "nt"

_started: list[tuple[str, subprocess.Popen[str]]] = []


def spawn(name: str, cwd, argv: list[str]) -> None:
    """Start a server, its own process group, output pumped through a prefixing thread."""
    exe = tool(argv[0])
    if exe is None:
        die("dev", f"{argv[0]} is not on your PATH.")
    # Its own group either way, which is what makes the whole tree killable below: a
    # Ctrl-C otherwise reaches this process and leaves `next dev` orphaned, holding :3000
    # against the next run.
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if WINDOWS
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        [exe, *argv[1:]],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line buffered: the prefix appears as the server logs it, not at exit
        errors="replace",
        **group,
    )
    _started.append((name, proc))
    threading.Thread(target=_pump, args=(name, proc), daemon=True).start()


def _pump(name: str, proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}", flush=True)


def stop_all() -> None:
    """Kill each server and everything it spawned (pnpm -> next dev)."""
    if not _started:
        # Nothing started means nothing to stop. Staying quiet matters: announcing a
        # shutdown would make the "everything was already up" exit read as though this
        # had torn down servers it never owned.
        return
    print()
    say("dev", "shutting down")
    for _, proc in _started:
        if proc.poll() is not None:
            continue
        try:
            if WINDOWS:
                # No process groups to signal on Windows in the POSIX sense; taskkill /T
                # is what walks the tree. This is the piece that has no shell equivalent
                # under Git Bash, whose PIDs are not the ones taskkill wants.
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    for _, proc in _started:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def open_ui_when_ready() -> None:
    """Open the window once :3000 answers.

    Polled rather than opened straight away: `next dev` needs a few seconds to bind, and a
    browser that arrives first shows a connection error the user has to reload past.
    `webbrowser` covers macOS, Linux, Windows and WSL in one call, which is what the
    open/xdg-open/wslview trio was doing by hand.
    """
    if os.environ.get("NO_BROWSER"):
        return

    def wait_and_open() -> None:
        for _ in range(120):
            if listening(3000):
                webbrowser.open(UI_URL)
                return
            time.sleep(0.5)

    threading.Thread(target=wait_and_open, daemon=True).start()


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    require_setup("dev")
    # Checked up front rather than at spawn time, where a missing pnpm would surface only
    # after the agent server has already claimed its port.
    if tool("pnpm") is None:
        die("dev", "pnpm is not on your PATH. Run:  uv run scripts/setup.py")

    ui_dir = chat_ui_dir()
    if not ui_dir.is_dir():
        say("dev", f"no chat UI at {ui_dir}")
        die("dev", "install it (clone, /ui/* rewrite, pnpm install) with:  uv run scripts/setup.py")

    if listening(2024):
        say("dev", ":2024 already serving — reusing it")
    else:
        # --n-jobs-per-worker: `langgraph dev` defaults this to 1, so a single run occupies
        # the only worker and every later run queues behind it. That bites hardest across
        # restarts: the server persists its queue to .langgraph_api/, so a run abandoned by
        # a Ctrl-C is resumed on the next boot, takes the worker, and starves the query you
        # just typed — which sits `pending` forever, produces no LangSmith trace, and looks
        # like a hang while the console streams the *old* run's progress.
        #
        # This makes the asyncio.to_thread rule in sources/cache_io.py load-bearing rather
        # than theoretical: runs share one event loop, so a blocking call stalls neighbours.
        spawn(
            "agent",
            REPO_ROOT,
            ["uv", "run", "--group", "dev", "langgraph", "dev",
             "--no-browser", "--n-jobs-per-worker", "5"],
        )

    if listening(3000):
        say("dev", ":3000 already serving — reusing it")
        if not os.environ.get("NO_BROWSER"):
            webbrowser.open(UI_URL)
    else:
        spawn("ui", ui_dir, ["pnpm", "dev"])
        open_ui_when_ready()

    # Both already up. Nothing to supervise and nothing this script may stop — the servers
    # belong to whoever started them — so say so and get out of the way.
    if not _started:
        say("dev", "both halves already running — nothing to start.")
        say("dev", f"chat UI -> {UI_URL}")
        return 0

    say("dev", f"chat UI -> {UI_URL}   (Ctrl-C stops what this script started)")
    # Every signal explicitly — the `trap cleanup INT TERM` the shell version had, plus the
    # two it was missing. Each default disposition skips the teardown below in its own way,
    # and the cost is identical every time: both servers outlive us still holding their
    # ports, and the *next* launch finds them listening and reuses them, so a stale stack
    # silently serves the run.
    #
    # SIGTERM, SIGHUP and SIGQUIT because Python's default handler exits without unwinding.
    # SIGHUP is the one that actually bites: closing the terminal signals the foreground
    # process group, and the servers are deliberately in sessions of their own (see
    # `spawn`), so the signal reaches only us — exactly the process whose job was to kill
    # them. SIGINT because its default handler is not guaranteed to be installed at all: a
    # process started in the background by a non-interactive shell inherits SIGINT as
    # SIG_IGN, and Python keeps that disposition rather than raising KeyboardInterrupt.
    #
    # Looked up by name rather than named directly: SIGHUP and SIGQUIT do not exist on
    # Windows, and `signal.SIGHUP` there is an AttributeError raised while building the
    # loop's sequence, i.e. before any suppression inside it can apply.
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _raise_interrupt)
    try:
        while True:
            for name, proc in _started:
                if proc.poll() is not None:
                    say("dev", f"{name} exited ({proc.returncode})")
                    return proc.returncode or 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
