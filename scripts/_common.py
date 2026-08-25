"""Helpers shared by `setup.py` and `dev.py`.

These were duplicated between the two bash launchers so either could run alone. In Python
they are one import away, so the copies are gone and the `.env` placeholder rule — the one
that had to stay identical in both — is now a single function.

Nothing here imports `research_agent`: these run before `uv sync` has necessarily happened.
"""

from __future__ import annotations

import http.client
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def say(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def die(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def tool(name: str) -> str | None:
    """Absolute path to an executable, or None.

    Windows needs this rather than a bare name: npm and pnpm are `.cmd` shims there, and
    `subprocess` does not consult PATHEXT the way the shell does — a bare "pnpm" raises
    FileNotFoundError on a machine that has pnpm installed and working.
    """
    return shutil.which(name)


def run(argv: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    """Run a command, letting its output through to the terminal."""
    exe = tool(argv[0])
    if exe is None:
        raise FileNotFoundError(argv[0])
    completed = subprocess.run([exe, *argv[1:]], cwd=cwd)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def listening(port: int) -> bool:
    """True if something accepts connections on the loopback port.

    Replaces `lsof`, which is absent on Windows and not guaranteed on a minimal Linux
    image. Like the shell version this answers "something is there", not "something of
    ours is there" — on loopback the distinction has not come up.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def answering(port: int, path: str = "/ok", timeout: float = 5.0) -> bool:
    """True if the loopback port answers an HTTP GET with a 2xx.

    `listening` proves only that something holds the port. A `langgraph dev` that has
    wedged — still bound, still in the process table, answering nothing — passes that
    check, and reusing it points the chat UI at a server whose every request hangs: the
    window sits on its thinking indicator forever, with no error anywhere naming the
    cause. Asking for a *response* rather than a socket is what tells the two apart.

    Generous timeout on purpose. A healthy server can be slow to this if its event loop is
    momentarily blocked (the asyncio.to_thread rule in sources/cache_io.py is about exactly
    that), and calling a working server dead is the more annoying error of the two.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return 200 <= r.status < 300
    except (OSError, http.client.HTTPException):
        # urllib.error.URLError and HTTPError are both OSError subclasses, so this covers
        # refused, timed out and 4xx/5xx alike; HTTPException covers a truncated reply.
        return False


def env_value(key: str) -> str:
    """The value of `key` in .env — empty if unset, or still the placeholder.

    `.env.example` ships `KEY=lsv2_sk_...` as a hint, so a trailing `...` counts as unset.
    """
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            value = line[len(key) + 1 :].strip()
            return "" if value.endswith("...") else value
    return ""


def set_env(key: str, value: str) -> None:
    """Rewrite the `key=` line in .env, or append it, leaving comments untouched."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chat_ui_dir() -> Path:
    """Where the frontend lives: inside the repo, or wherever AGENT_CHAT_UI points."""
    override = os.environ.get("AGENT_CHAT_UI")
    return Path(override).expanduser() if override else REPO_ROOT / ".chat-ui"


def require_setup(tag: str) -> None:
    """Fail with the fix rather than the symptom.

    Each of these is a setup step that did not happen, and each otherwise fails later in a
    way that does not name the cause: no .env is an SDK auth error deep in a run, no
    virtualenv is uv silently installing mid-launch. The snapshot is deliberately *not*
    checked — a missing one only makes runs slower (sandbox.py installs at runtime), and
    confirming it costs a LangSmith round trip on every launch.
    """
    if not ENV_FILE.exists():
        die(tag, f"no .env in {REPO_ROOT}. Run:  uv run scripts/setup.py")
    if not (REPO_ROOT / ".venv").is_dir():
        die(tag, f"no virtualenv in {REPO_ROOT}. Run:  uv run scripts/setup.py")
    for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not env_value(key):
            die(tag, f"{key} is not set in .env. Run:  uv run scripts/setup.py")
