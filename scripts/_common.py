"""Helpers shared by `setup.py` and `dev.py`.

These were duplicated between the two bash launchers so either could run alone. In Python
they are one import away, so the copies are gone and the `.env` placeholder rule — the one
that had to stay identical in both — is now a single function.

Nothing here imports `deep_life_sci`: these run before `uv sync` has necessarily happened.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
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


# A Node that setup unpacked into the repo, for a machine with none it can use (see
# `install_node` in setup.py). Ignored by git and docker.
LOCAL_NODE_DIR = REPO_ROOT / ".node"


def local_node_bin() -> Path:
    """Where that Node's executables live: `bin/` in the POSIX tarballs, the root on Windows."""
    return LOCAL_NODE_DIR if os.name == "nt" else LOCAL_NODE_DIR / "bin"


def use_local_node() -> None:
    """Put the repo-local Node first on PATH, if setup installed one.

    PATH rather than a path threaded through the callers, because everything that needs
    node finds it there: `tool()`, npm and corepack resolving their own `node`, pnpm's
    scripts, and every child `dev.py` spawns. That is also why a Node unpacked here is not
    the "visible to this process alone" install `install_node` warns about: the UI only
    ever runs through `setup.py` and `dev.py`, and both import this module, which calls
    this at import so neither can forget to.
    """
    bin_dir = local_node_bin()
    path = os.environ.get("PATH", "")
    if bin_dir.is_dir() and str(bin_dir) not in path.split(os.pathsep):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + path


use_local_node()


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
    that), and calling a working server dead is the worse of the two errors.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return 200 <= r.status < 300
    except (OSError, http.client.HTTPException):
        # urllib.error.URLError and HTTPError are both OSError subclasses, so this covers
        # refused, timed out and 4xx/5xx alike; HTTPException covers a truncated reply.
        return False


# One .env assignment, read the way python-dotenv reads it, since that is how the server, the
# CLI and `langgraph deploy` read the same file: leading whitespace and `export ` are allowed,
# and spaces around `=`; a value may be quoted; an unquoted value ends at a ` #` comment.
# A line uncommented by deleting only its `#` keeps its indent, and must still count.
_ASSIGNMENT = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)")
_QUOTED = {"'": re.compile(r"'((?:\\'|[^'])*)'"), '"': re.compile(r'"((?:\\"|[^"])*)"')}
_ESCAPES = {"'": re.compile(r"\\([\\'])"), '"': re.compile(r'\\([\\"nt])')}


def _assignment(line: str) -> tuple[str, str] | None:
    """`(key, value)` if the line assigns one, else None."""
    match = _ASSIGNMENT.fullmatch(line)
    if not match:
        return None
    key, raw = match.groups()
    quoted = _QUOTED.get(raw[:1])
    found = quoted.match(raw) if quoted else None
    if found is None:
        return key, re.split(r"\s+#", raw, maxsplit=1)[0].strip()
    unescape = {"n": "\n", "t": "\t"}
    return key, _ESCAPES[raw[0]].sub(lambda m: unescape.get(m[1], m[1]), found[1])


def env_value(key: str) -> str:
    """The value of `key` in .env — empty if unset, or still the placeholder.

    `.env.example` ships `KEY=lsv2_sk_...` as a hint, so a trailing `...` counts as unset. A
    key assigned twice has its last value, as python-dotenv gives it.
    """
    if not ENV_FILE.exists():
        return ""
    value = ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        found = _assignment(line)
        if found and found[0] == key:
            value = found[1]
    return "" if value.endswith("...") else value


def set_env(key: str, value: str) -> None:
    """Rewrite the line assigning `key` in .env (the last, which is the one that counts), or
    append one, leaving comments untouched."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    for i in reversed(range(len(lines))):
        found = _assignment(lines[i])
        if found and found[0] == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def frontend_dir() -> Path:
    """The chat UI, a Next app kept in this repo (see frontend/UPSTREAM.md)."""
    return REPO_ROOT / "frontend"


# Written into a node_modules by setup once an install from the lockfile beside it succeeds:
# that lockfile's digest. A pull that changes the lockfile no longer matches it, which is how
# setup knows to install again and dev.py knows to say so, rather than either trusting a
# node_modules that merely exists and starting a UI that fails on a module it lacks.
DEPS_STAMP = ".deep-life-sci-lockfile"


def _lockfile_digest(lockfile: Path) -> str:
    return hashlib.sha256(lockfile.read_bytes()).hexdigest()


def deps_current(node_modules: Path, lockfile: Path) -> bool:
    """Whether `node_modules` was installed by setup from `lockfile` as it is now."""
    try:
        stamp = (node_modules / DEPS_STAMP).read_text(encoding="utf-8").strip()
        return stamp == _lockfile_digest(lockfile)
    except OSError:
        return False


def stamp_deps(node_modules: Path, lockfile: Path) -> None:
    (node_modules / DEPS_STAMP).write_text(_lockfile_digest(lockfile) + "\n", encoding="utf-8")


class Pnpm(NamedTuple):
    """How to run pnpm here. `argv` is a command *prefix*, not a path: on a machine with no
    pnpm of its own it is `["corepack", "pnpm"]` or `["npm", "exec", ...]`, so callers must
    splice it (`[*found.argv, "install"]`) rather than treat element 0 as the executable.
    """

    argv: list[str]
    version: str
    how: str


def pinned_pnpm() -> str | None:
    """frontend/'s `packageManager` pin, e.g. "pnpm@10.5.1" — or None if it has none.

    Reading this ourselves is what makes the ladder below robust. Corepack is the tool that
    normally reads this field, but it is bundled with node under a long-standing plan to
    unbundle it, so a setup that *depends* on corepack inherits that clock. Holding the pin
    as a string instead means every rung can honour it and corepack becomes a convenience.
    """
    manifest = frontend_dir() / "package.json"
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
                                 is reproducible too: a newer global pnpm runs as the pinned
                                 version inside frontend/. An older pnpm reports its own
                                 version, fails the match below, and falls through.
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
    ui = frontend_dir()
    # The pin lives in frontend/, so rung 1 must run *there* to self-correct onto it.
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
            say(tag, f"fetching {pin} with `npm exec` — a few minutes.")
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
    # Under MODEL_ACCESS=direct the gateway key goes unused, and which provider keys are
    # needed depends on models.yaml: the server checks those as it starts.
    keys = ["LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"]
    if env_value("MODEL_ACCESS").lower() == "direct":
        keys.remove("LANGSMITH_GATEWAY_API_KEY")
    for key in keys:
        if not env_value(key):
            die(tag, f"{key} is not set in .env. Run:  uv run scripts/setup.py")
