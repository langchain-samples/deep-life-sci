"""Helpers shared by `setup.py` and `dev.py`.

These were duplicated between the two bash launchers so either could run alone. In Python
they are one import away, so the copies are gone and the `.env` placeholder rule — the one
that had to stay identical in both — is now a single function.

Nothing here imports `research_agent`: these run before `uv sync` has necessarily happened.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

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


class Pnpm(NamedTuple):
    """How to run pnpm here. `argv` is a command *prefix*, not a path: on a machine with no
    pnpm of its own it is `["corepack", "pnpm"]` or `["npm", "exec", ...]`, so callers must
    splice it (`[*found.argv, "install"]`) rather than treat element 0 as the executable.
    """

    argv: list[str]
    version: str
    how: str


def pinned_pnpm() -> str | None:
    """The chat UI's `packageManager` pin, e.g. "pnpm@10.5.1" — or None if it has none.

    Reading this ourselves is what makes the ladder below robust. Corepack is the tool that
    normally reads this field, but it is bundled with node under a long-standing plan to
    unbundle it, so a setup that *depends* on corepack inherits that clock. Holding the pin
    as a string instead means every rung can honour it and corepack becomes a convenience.
    """
    manifest = chat_ui_dir() / "package.json"
    if not manifest.is_file():
        return None
    try:
        field = json.loads(manifest.read_text(encoding="utf-8")).get("packageManager")
    except (OSError, ValueError):
        return None
    if not isinstance(field, str) or not field.startswith("pnpm@"):
        return None
    # "pnpm@10.5.1+sha512.<hash>" — corepack's optional integrity suffix, which `npm exec`
    # does not understand.
    return field.split("+", 1)[0]


def _probe(argv: list[str], cwd: Path | None, timeout: float) -> str | None:
    """Run `argv` for its stdout, or None if it fails in any way whatsoever.

    Every rung is probed by *attempting* it rather than by detecting whether it ought to
    work, because the ways it can fail are not enumerable from here: corepack unbundled,
    no network, a filtering proxy, a registry that needs auth, a half-written cache. The
    attempt is the only honest test, and `--version` exercises the whole path — resolve the
    pin, fetch, cache, execute — for the cost of the download the real call needs anyway.

    Bounded, because a rung stalls rather than fails behind a proxy that blackholes rather
    than refuses. `subprocess.run(timeout=)` rather than the `timeout` binary, which macOS
    does not ship.
    """
    exe = tool(argv[0])
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [exe, *argv[1:]], cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def pnpm_command(tag: str = "setup") -> Pnpm | None:
    """Find a way to run the pinned pnpm, or None having tried every one we know.

    Three rungs, cheapest first, all landing on the same version:

      1. `pnpm` on PATH        — no download. pnpm 9+ re-execs itself at the pin, so this
                                 is reproducible too (verified: a global 11.x runs as 10.5.1
                                 inside the clone). An older pnpm reports its own version,
                                 fails the match below, and falls through.
      2. `corepack pnpm`       — bundled with node, downloads the pin to a per-user cache.
      3. `npm exec --yes <pin>` — the rung that survives corepack being unbundled. Caches
                                 under ~/.npm, so it needs no writable node prefix either.

    Deliberately *not* a rung: `npm install -g pnpm`. Its global prefix is the node install
    root, which is root-owned on any node from a .pkg/apt/system image, so it is the one
    option that fails on a permission error rather than on anything about pnpm — and
    `sudo`-ing it leaves root-owned files in ~/.npm that break the user's next npm command.
    """
    # Corepack asks an interactive y/n the first time it fetches a version. Our children
    # inherit the terminal, so that prompt would surface inside setup.py's own questions and
    # read as a hang; `capture_output` in the probe would hide it as one outright.
    os.environ.setdefault("COREPACK_ENABLE_DOWNLOAD_PROMPT", "0")

    pin = pinned_pnpm()
    want = pin.split("@", 1)[1] if pin else None
    ui = chat_ui_dir()
    # The pin lives in the clone, so rung 1 must run *there* to self-correct onto it.
    cwd = ui if ui.is_dir() else None

    rungs: list[tuple[str, list[str], float]] = [
        ("PATH", ["pnpm"], 60),
        ("corepack", ["corepack", "pnpm"], 300),
    ]
    if pin:
        # Needs the pin spelled out: without a version `npm exec` resolves whatever is
        # latest, which is the reproducibility the other two rungs get for free.
        # Ten minutes because this one measured over two on a cold ~/.npm and only runs at
        # all once both rungs above have failed — a spurious timeout here is the difference
        # between a working UI and none, while the wait is paid once and then cached.
        rungs.append(("npm exec", ["npm", "exec", "--yes", pin, "--"], 600))

    for how, argv, timeout in rungs:
        if how == "npm exec" and tool("npm") is not None:
            # The probe captures output, so without this the slowest rung is also the one
            # that looks like a hang.
            say(tag, f"fetching {pin} with `npm exec` — a few minutes, once.")
        version = _probe([*argv, "--version"], cwd, timeout)
        # A rung can answer with the wrong pnpm rather than fail: an older `pnpm` on PATH
        # predates the self-correcting `packageManager` support, and corepack outside a
        # project with a pin serves whatever is latest (both verified). Neither is the
        # version the lockfile was written by, so treat a mismatch as a failed rung.
        if version is None or (want and version != want):
            continue
        return Pnpm(argv, version, how)
    return None


def pnpm_or_die(tag: str) -> Pnpm:
    """`pnpm_command`, reported. Which rung answered is the first thing worth knowing about
    a machine where the UI would not install, and it is invisible afterwards.
    """
    found = pnpm_command(tag)
    if found is not None:
        say(tag, f"pnpm {found.version} via {found.how}")
        return found
    say(tag, "no pnpm: not on PATH, and neither corepack nor `npm exec` could fetch it.")
    say(tag, "if you are online and this persists, install it yourself with one of:")
    say(tag, "  curl -fsSL https://get.pnpm.io/install.sh | sh -")
    say(tag, f"  npm install -g {pinned_pnpm() or 'pnpm'}   "
             "# only if `npm config get prefix` is writable — never with sudo")
    die(tag, "then re-run this script.")
    raise AssertionError("unreachable")  # die() exits; this is for the type checker


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
    for key in ("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"):
        if not env_value(key):
            die(tag, f"{key} is not set in .env. Run:  uv run scripts/setup.py")
