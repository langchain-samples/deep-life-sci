"""Start the local stack: the agent server and the chat UI that renders its artifacts.

    uv run scripts/dev.py                    # both, logs interleaved and prefixed
    uv run scripts/dev.py --remote <URL>     # only the chat UI, against a deployment
    NO_BROWSER=1 uv run scripts/dev.py       # don't open a browser tab

Both halves are required. :2024 serves graph.py; :3000 serves the frontend, and is the
side carrying the `/ui/*` rewrite that lets artifact components load at all (see the
same-origin invariant in CLAUDE.md). Running only one gets you an API with no window, or a
window pointed at nothing.

A server already serving on its port is reused unless you say to stop it: what is holding
the port is named, with the directory it is running from, and killing it is offered. With
no tty to answer the prompt, reuse is what happens — so a CI job or a backgrounded launch
still gets the old behaviour of never stopping what this script did not start.

Python rather than bash because this is the one command Windows users cannot avoid, and
the three things it needs — a port check, a browser, and killing a process tree on Ctrl-C —
are exactly the three with no portable shell spelling.

`--remote` is for a server made by `scripts/deploy.py`. The UI then talks to it through
the upstream app's own API passthrough (`src/app/api/[..._path]`), which adds this .env's
LangSmith key server-side so the key never reaches the browser. That makes the passthrough
as good as the key, so it serves this machine's own page and nothing else: the UI listens on
127.0.0.1 only, and the route refuses requests from any other origin, so neither another
machine on the network nor another website open in the browser can borrow it. The `/ui/*`
rewrite follows the same `LANGGRAPH_API_URL`; the server serves those assets without
authentication. This is for your own use from localhost, not a way to publish the UI.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (
    REPO_ROOT,
    answering,
    deps_current,
    die,
    env_value,
    frontend_dir,
    listening,
    pnpm_or_die,
    require_setup,
    say,
    tool,
)

# `hideToolCalls` is a nuqs query param, so opening the tab with it set is a default
# without a code change: nuqs keeps it across thread switches, and the in-app toggle still
# turns it back off. Our root turns are almost all thinking + tool_use, so left on the
# upstream default the transcript is mostly collapsed tool cards.
UI_URL = "http://localhost:3000?hideToolCalls=true"
WINDOWS = os.name == "nt"

_started: list[tuple[str, subprocess.Popen[str]]] = []


def spawn(name: str, cwd, argv: list[str], env: dict[str, str] | None = None) -> None:
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
        env={**os.environ, **env} if env else None,
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
            # The group again, not `proc.kill()`: that reaps the direct child (pnpm) and
            # leaves `next dev` behind it holding :3000 — an orphan the next launch finds
            # listening and adopts. A server busy enough to miss its SIGTERM window is
            # exactly the one that must not survive this.
            if WINDOWS:
                proc.kill()
            else:
                with contextlib.suppress(OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


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


def _ask(question: str, *, default: bool = False) -> bool:
    """A yes/no prompt that answers itself when nobody is there.

    `False` without a tty, whatever the default: killing a server nobody was asked about
    is the one outcome this must never produce on its own.
    """
    if not sys.stdin.isatty():
        return False
    hint = "[Y/n]" if default else "[y/N]"
    answer = input(f"[dev] {question} {hint} ").strip().lower()
    return default if not answer else answer.startswith("y")


def _port_holder(port: int) -> tuple[int, str] | None:
    """The pid and working directory of whatever is listening on `port`.

    The directory is what a port number cannot tell you: which checkout this server came
    from. `lsof` is absent on Windows, so the answer is `None` there and the caller reuses,
    which is what every platform did before.
    """
    if WINDOWS or not tool("lsof"):
        return None
    listeners = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    if not listeners:
        return None
    pid = int(listeners[0])
    # -Fn is the parseable form: one `n`-prefixed line carrying the path.
    fields = subprocess.run(
        ["lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn"],
        capture_output=True, text=True, check=False,
    ).stdout
    cwd = next((line[1:] for line in fields.splitlines() if line.startswith("n")), "")
    return pid, cwd


def _kill_holder(port: int, pid: int) -> bool:
    """Stop the process holding `port`. True once the port is free.

    The process *group*, for the reason `stop_all` gives: a `pnpm` reaped on its own leaves
    `next dev` behind it still holding :3000, and an orphan is exactly what the next launch
    adopts. Falls back to the bare pid when the group cannot be read.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (OSError, ProcessLookupError):
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, sig)
        for _ in range(40):
            if not listening(port):
                say("dev", f"stopped pid {pid}; :{port} is free")
                return True
            time.sleep(0.25)
    say("dev", f"warning: :{port} is still held after SIGKILL — reusing it")
    return False


def _take_over(port: int, label: str, healthy: bool, kill_hint: str) -> bool:
    """Whether to start our own `label` on `port`. False means reuse what is there."""
    holder = _port_holder(port)
    if holder:
        say("dev", f":{port} is held by pid {holder[0]}"
                   + (f" in {holder[1]}" if holder[1] else ""))

    if not healthy:
        # A wedged server cannot be reused at all, so the only question is who stops it.
        say("dev", f":{port} is held by a server that is not responding.")
        if (
            holder
            and _ask(f"kill pid {holder[0]} and start a fresh {label}?", default=True)
            and _kill_holder(port, holder[0])
        ):
            return True
        die("dev", f"stop it and run this again:  {kill_hint}")

    if not holder:
        return False
    if not _ask(f"kill it and start {label} from this checkout?"):
        return False
    return _kill_holder(port, holder[0])


def remote_ui_env(url: str) -> dict[str, str]:
    """What the chat UI needs to talk to a deployment instead of :2024.

    Environment rather than `.env.local`, which setup wrote for the local server and which
    Next reads only for names the environment has not already set.
    """
    return {
        "LANGGRAPH_API_URL": url.rstrip("/"),
        "LANGSMITH_API_KEY": env_value("LANGSMITH_API_KEY"),
        "NEXT_PUBLIC_API_URL": "http://localhost:3000/api",
        "NEXT_PUBLIC_ASSISTANT_ID": "agent",
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="uv run scripts/dev.py")
    parser.add_argument("--remote", metavar="URL",
                        help="start only the chat UI, against this deployment")
    args = parser.parse_args()
    if args.remote and not args.remote.startswith(("http://", "https://")):
        die("dev", f"--remote wants the deployment's URL, starting https://; got {args.remote!r}")

    require_setup("dev")
    ui_dir = frontend_dir()
    # Checked rather than installed: starting the app must not install anything. A checkout
    # from before frontend/ was in the repo lands here after its first pull, and so does one
    # whose pull changed a lockfile since setup last installed from it. The artifact
    # components are bundled by the local agent server, so `--remote` needs only the UI's.
    if not (ui_dir / "node_modules").is_dir():
        die("dev", "the chat UI's dependencies are not installed. Run:  uv run scripts/setup.py")
    needed = [(ui_dir / "node_modules", ui_dir / "pnpm-lock.yaml")]
    if not args.remote:
        needed.append((REPO_ROOT / "node_modules", REPO_ROOT / "package-lock.json"))
    stale = [lockfile for modules, lockfile in needed if not deps_current(modules, lockfile)]
    if stale:
        names = " and ".join(str(lockfile.relative_to(REPO_ROOT)) for lockfile in stale)
        die("dev", f"{names} changed since setup installed from it. Run:  uv run scripts/setup.py")

    # Resolved up front rather than at spawn time, where an unrunnable pnpm would surface
    # only after the agent server had already claimed its port. Re-resolved on every launch
    # rather than recorded by setup: it is cheap once the first rung answers, and a path
    # cached here would go stale the next time the user reinstalls node.
    pnpm = pnpm_or_die("dev")

    # Reused only if it *answers*. `langgraph dev` can wedge — bound to the port, alive in
    # the process table, serving nothing — and a port check adopts that as a working
    # server: every request from the chat UI then hangs, so the window shows its thinking
    # indicator forever with no error to explain it. `_take_over` asks before stopping
    # anything, so a :2024 of your own still survives a run of this.
    if args.remote:
        say("dev", f"agent server: {args.remote}")
    elif listening(2024) and not _take_over(
        2024, "the agent server", answering(2024),
        'pkill -f "langgraph dev"   '
        '(add -9 if it survives; Windows: taskkill /F /IM langgraph.exe)',
    ):
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

    # Same wedge, same rule as :2024 above — `next dev` can end up bound to the port and
    # answering nothing, and adopting that paints a window that never fills in. No /ok
    # here, so the root document is the liveness check, and it gets longer than the default
    # timeout: Next compiles that route on first request, so a healthy but cold server can
    # take well over 5s, and calling it dead sends the user off to kill a server that was
    # about to work.
    ui_kill_hint = (
        "lsof -ti tcp:3000 | xargs kill   (add -9 if it survives; Windows: npx kill-port 3000)"
    )
    if listening(3000) and not _take_over(
        3000, "the chat UI", answering(3000, path="/", timeout=20.0), ui_kill_hint,
    ):
        if args.remote:
            # Reusing it would chat with whichever server it was started against.
            die("dev", f"a chat UI is already on :3000; stop it to use --remote:  {ui_kill_hint}")
        say("dev", ":3000 already serving — reusing it")
        if not os.environ.get("NO_BROWSER"):
            webbrowser.open(UI_URL)
    else:
        spawn("ui", ui_dir, [*pnpm.argv, "dev"],
              env=remote_ui_env(args.remote) if args.remote else None)
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
