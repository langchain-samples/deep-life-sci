"""One-time setup. Run this once per clone, then ask questions with the chat UI.

    uv run scripts/setup.py          # prompt for the API key, install everything
    uv run scripts/setup.py --yes    # never prompt; for CI and containers

Four steps, in the order they depend on each other:

    1. .env        — the LangSmith API key, prompted for and written here
    2. uv sync     — the virtualenv
    3. a snapshot  — sandbox image with the scientific Python stack baked in
    4. the chat UI — the frontend, and the deps for the components it renders

Step 3 needs step 2 (it imports langsmith) and step 1 (it calls LangSmith), which is why
this is a script rather than a list in the README. Every step is skipped when already done,
so re-running after a `git pull` is the cheap way to catch up.

uv itself is not a step: you are already running under it. Installing it is the one-liner
in the README, and it brings its own Python, so it stays the only prerequisite.

The chat UI is not optional and has no flag: it is how this agent is meant to be used, and
a headless-only install is the unusual case. It comes last so that a machine without Node
still ends up with a working `uv run agent`.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (
    ENV_FILE,
    LOCAL_NODE_DIR,
    REPO_ROOT,
    deps_current,
    die,
    env_value,
    frontend_dir,
    pnpm_or_die,
    run,
    say,
    set_env,
    stamp_deps,
    tool,
    use_local_node,
)

TAG = "setup"
WINDOWS = os.name == "nt"
# frontend/ is on Next 16, which refuses to build below 20.9. Checked as a major so a
# .0 release of 20 is not read as too old; Next's own check catches 20.0-20.8.
NODE_MIN_MAJOR = 20
# The Node setup installs into the repo when the machine has none it can use: the current
# LTS, pinned so every install builds the UI with the same Node, and checked against nodejs.org's
# published SHA-256 before anything is unpacked. To bump: change the version, paste the
# matching lines from https://nodejs.org/dist/v<version>/SHASUMS256.txt, `rm -rf .node`,
# re-run setup. An existing `.node` is never replaced, so the bump reaches a clone only then.
NODE_VERSION = "24.21.0"
NODE_SHA256 = {
    "node-v24.21.0-darwin-arm64.tar.gz":
        "bed7eea5325e1108f32ce5228ddd6a5f0f08a499ee42aa7442aea583702f6057",
    "node-v24.21.0-darwin-x64.tar.gz":
        "1462cb3b3046b815cf8ea436d3da450ec1a9f11dac7e5a46b0ada5305d7e8097",
    "node-v24.21.0-linux-arm64.tar.gz":
        "724282c3b43aec998aa9527380465b45d229e021b58035f5f4f63095eabfe5d5",
    "node-v24.21.0-linux-x64.tar.gz":
        "6e1db87ef58b8819e5d5402eff1536491b18edd8eb7bee5ef7897876e88dc5ff",
    "node-v24.21.0-win-arm64.zip":
        "8779b1bde1d39f8d420e3b57aa657b39891af434d3de44a919044cec06785921",
    "node-v24.21.0-win-x64.zip":
        "158f7685b44de51f6c0df1d153526cbcd3e1bc739a8dfc607721cef75de9e541",
}
_assume_yes = False


def interactive() -> bool:
    """A prompt only makes sense with someone there to answer it."""
    return not _assume_yes and sys.stdin.isatty()


def confirm(question: str) -> bool:
    if not interactive():
        return True
    return (input(f"[{TAG}] {question} [Y/n] ").strip() or "y").lower().startswith("y")


# --- 1. .env ----------------------------------------------------------------------


def ask_secret(prompt: str) -> str:
    """Read a credential without echoing it.

    `input()` would print the key into the terminal, from where it reaches scrollback, a
    `script` log, and any screen share or pasted transcript. Only ever called behind
    `interactive()`, which has already established a tty, so getpass cannot fall back to
    its unmasked echoing path.

    Whitespace is stripped rather than rejected because nothing is echoed to proofread: a
    key pasted with a trailing newline or a stray space looks identical to a clean one.
    """
    return "".join(getpass.getpass(f"[{TAG}] {prompt}: ").split())


def ask_key(key: str, prompt: str, prefix: str) -> None:
    """Required, loops until answered."""
    if env_value(key):
        return
    if not interactive():
        die(TAG, f"{key} is not set in .env and there is no terminal to ask on. "
                 "Add it and re-run.")
    while True:
        reply = ask_secret(prompt)
        if not reply:
            continue
        # A wrong-but-plausible key is the failure this catches: a provider key pasted
        # here looks right and dies deep in the SDK at the first model call, because the
        # gateway takes only a LangSmith key. Queried rather than rejected — key formats
        # belong to LangSmith.
        if prefix and not reply.startswith(prefix):
            # Echoing the first few characters is the only proofreading available now that
            # the key itself never appears — enough to spot a provider key or a truncated
            # paste, short enough not to be the secret.
            say(TAG, f"that starts '{reply[:8]}…', not '{prefix}' — "
                     "see the note in .env.example.")
            if not input(f"[{TAG}] use it anyway? [y/N] ").strip().lower().startswith("y"):
                continue
        set_env(key, reply)
        return


def ask_optional(key: str, prompt: str, *, secret: bool = False) -> None:
    """`secret` for a credential, which must not echo; the default suits an email."""
    if not interactive():
        return
    ask = ask_secret if secret else lambda p: "".join(input(f"[{TAG}] {p}: ").split())
    reply = ask(f"{prompt} (optional, Enter to skip)")
    if reply:
        set_env(key, reply)


def migrate_gateway_key() -> None:
    """Rename the gateway key where an older clone still calls it OPENAI_API_KEY.

    Renamed because the name said OpenAI while the value is a LangSmith key, which is the
    single most confusing thing about the setup. Nothing reads the old name any more, so
    an unmigrated .env would look unconfigured; the whole file is rewritten so the
    explanatory comment above the key follows it.
    """
    if not ENV_FILE.exists():
        return
    text = ENV_FILE.read_text(encoding="utf-8")
    if "OPENAI_API_KEY" not in text or "LANGSMITH_GATEWAY_API_KEY" in text:
        return
    ENV_FILE.write_text(
        text.replace("OPENAI_API_KEY", "LANGSMITH_GATEWAY_API_KEY"), encoding="utf-8"
    )
    say(TAG, "renamed OPENAI_API_KEY to LANGSMITH_GATEWAY_API_KEY in .env")


def ensure_langsmith_key() -> None:
    """One LangSmith key, prompted for once and written to both names that read it.

    `LANGSMITH_API_KEY` authenticates tracing and sandbox provisioning;
    `LANGSMITH_GATEWAY_API_KEY` authenticates model calls at the LLM gateway. For almost
    everyone that is the same key, because the gateway takes a *LangSmith* key and resolves
    the provider credential from the workspace's Provider Secrets — a real `sk-...` is
    rejected with a 403 before it reaches a provider. Asking twice for one value was the
    most confusing thing in setup.

    Written to two names rather than one because the gateway key is a documented override:
    a workspace-scoped key carrying `gateway:invoke` can bill model calls somewhere other
    than the personal key doing the tracing. Editing `LANGSMITH_GATEWAY_API_KEY` in .env by
    hand is how that is done, and it is only ever filled in when empty, so a hand-edited
    one survives every later `setup.py`.

    An exported `LC_GATEWAY_KEY` is adopted rather than prompted for. A deployment that
    issues a per-machine gateway *service* key — so model spend stays capped and
    attributable — already holds the value in the environment, and copying the tracing key
    into the gateway slot instead 403s at the first model call, which reads like a broken
    install rather than a wrong key.
    """
    ask_key(
        "LANGSMITH_API_KEY",
        "LangSmith API key, for tracing and sandboxes (lsv2_...)",
        "lsv2_",
    )
    if env_value("LANGSMITH_GATEWAY_API_KEY"):
        return
    if gateway := os.environ.get("LC_GATEWAY_KEY", "").strip():
        set_env("LANGSMITH_GATEWAY_API_KEY", gateway)
        say(TAG, "using LC_GATEWAY_KEY from the environment for model calls "
                 "(tracing keeps LANGSMITH_API_KEY)")
        return
    set_env("LANGSMITH_GATEWAY_API_KEY", env_value("LANGSMITH_API_KEY"))


# The `tool` every E-utilities request carries. NCBI enforces its rate limits against this
# string as well as the IP, so a value shared by every clone of a public repo is one where
# any single runaway loop gets *everyone* throttled or blocked — and the person who caused
# it is the one person NCBI cannot identify. A per-install suffix makes the installs
# distinguishable. It is a courtesy identifier and not a credential: nothing reads it back,
# and it is deliberately not derived from the machine or the user.
NCBI_TOOL_BASE = "deep_life_sci"


def ensure_ncbi_tool() -> None:
    """Give this install its own `tool` identifier, once.

    Replaces the bare shared name as well as an empty value, so a clone made before this
    existed stops sharing on its next setup. Any *other* value is left alone — an operator
    who set a meaningful name for their own group has said what they want.
    """
    current = env_value("NCBI_TOOL")
    if current and current != NCBI_TOOL_BASE:
        return
    set_env("NCBI_TOOL", f"{NCBI_TOOL_BASE}_{secrets.token_hex(4)}")


def ensure_env() -> None:
    fresh = not ENV_FILE.exists()
    if fresh:
        shutil.copy(REPO_ROOT / ".env.example", ENV_FILE)
        say(TAG, "created .env from .env.example")
    else:
        migrate_gateway_key()

    ensure_langsmith_key()
    ensure_ncbi_tool()

    # Only on a first run: these are genuinely optional, so re-asking every time would be
    # nagging someone who already decided to skip them.
    if fresh:
        say(TAG, "NCBI credentials are optional: they raise PubMed's rate limit "
                 "from 3 to 10 req/s.")
        ask_optional("NCBI_API_KEY", "NCBI API key", secret=True)
        ask_optional("NCBI_EMAIL", "contact email for NCBI (their usage policy asks for one)")


# --- 2. dependencies --------------------------------------------------------------


def ensure_deps() -> None:
    """The dev group too, unconditionally: it is only langgraph-cli, and syncing it here is
    what keeps the chat UI from stalling on an install after it has claimed the ports."""
    say(TAG, "syncing dependencies…")
    # `--frozen` installs uv.lock exactly. Every dependency is an unbounded `>=`,
    # and two of the APIs this repo builds on are beta (see CLAUDE.md), so a
    # re-lock on a fresh clone is how someone lands on a moved API rather than the
    # versions this was tested against.
    run(["uv", "sync", "--frozen", "--group", "dev", "--quiet"], cwd=REPO_ROOT)


# --- 3. sandbox snapshot ----------------------------------------------------------


# A workspace with no sandbox entitlement fails here rather than at the key check, and the
# platform's wording names neither the setting nor where to change it. The README does, so
# send them there rather than to the key they just typed correctly.
#
# Two messages because the platform gives two failures for one cause. Only the first is
# unambiguous; a bare 401 is equally what a wrong key looks like, and telling someone to
# go enable a setting they already have is its own dead end, so that one names both.
_ENTITLEMENT = "sandbox feature is not enabled"
_REJECTED = ("unauthorized", "401", "403")

# Hand-wrapped to `die()`'s hanging indent, so each is written out in full rather than
# sharing a fragment that can only line up under one of them.
SANDBOX_DISABLED_HINT = (
    "sandboxes are not enabled for this LangSmith workspace, which this agent needs.\n"
    "        Enable them under the Sandboxes tab in the LangSmith console — see the\n"
    "        setup section of README.md — then re-run this script."
)

SANDBOX_REJECTED_HINT = (
    "LangSmith rejected that key. Either it is wrong, or sandboxes are not enabled for\n"
    "        the workspace — this agent needs them. Enable them under the Sandboxes tab\n"
    "        in the LangSmith console — see the setup section of README.md — then\n"
    "        re-run this script."
)


def sandbox_hint(exc: Exception) -> str | None:
    """The hint `exc` calls for, or None when it is not about sandbox access.

    Matched on the message because the SDK raises one exception type for all of these.
    """
    text = str(exc).lower()
    if _ENTITLEMENT in text:
        return SANDBOX_DISABLED_HINT
    if any(sign in text for sign in _REJECTED):
        return SANDBOX_REJECTED_HINT
    return None


def ensure_snapshot() -> None:
    """Optional in the sense that a missing snapshot is slow rather than broken —
    sandbox.py falls back to a ~95s pip install per run — but ~100s once is the better
    trade. It is also why the launcher does not check for it: a per-run LangSmith round
    trip to re-learn something setup already guaranteed.

    Imported rather than re-read from .env, so this can never check for a different name
    than the agent boots from. In bash this was a heredoc piped to `uv run python -`; here
    it is an import, and the whole bash-3.2 parser workaround around it is gone.
    """
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)
    try:
        from deep_life_sci.sandbox import SNAPSHOT_NAME, make_client

        names = {s.name for s in make_client().list_snapshots(name_contains=SNAPSHOT_NAME)}
    except Exception as exc:  # noqa: BLE001 - any failure here is the same user-facing problem
        # Reaching LangSmith at all failed, so this is a credentials or connectivity
        # problem and every run would have it too. Fail here, where the cause is visible.
        print(exc, file=sys.stderr)
        die(TAG, sandbox_hint(exc)
            or "could not reach LangSmith. Check LANGSMITH_API_KEY in .env.")

    if SNAPSHOT_NAME in names:
        say(TAG, "sandbox snapshot ready.")
        return
    # The estimate lives in build_snapshot.py, beside the step it actually times; two
    # "~100s" lines in a row read as two waits.
    say(TAG, "building the sandbox snapshot (once)…")
    run(["uv", "run", "scripts/build_snapshot.py"], cwd=REPO_ROOT)


# --- 4. chat UI -------------------------------------------------------------------
#
# The frontend is this repo's own `frontend/`, a Next app that began as agent-chat-ui (see
# frontend/UPSTREAM.md). Setup only installs its dependencies and points it at the local
# server; the deploy image builds it for itself (scripts/deploy.py).


def node_major() -> int | None:
    """Major version of the node on PATH, or None when there is not a usable one.

    `npm` alone was the old check, and it answers the wrong question: node and npm are
    installed together, so npm's presence proves nothing about the version behind it.
    """
    exe = tool("node")
    if exe is None or tool("npm") is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    except OSError:
        return None
    found = re.match(r"v?(\d+)", out.stdout.strip())
    return int(found.group(1)) if found else None


def node_archive() -> str | None:
    """The nodejs.org archive for this machine, or None where there is no official build."""
    system = {"darwin": "darwin", "linux": "linux", "win32": "win"}.get(sys.platform)
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine().lower()
    )
    ext = "zip" if system == "win" else "tar.gz"
    name = f"node-v{NODE_VERSION}-{system}-{arch}.{ext}"
    return name if name in NODE_SHA256 else None


def install_node() -> None:
    """Unpack the pinned Node into `.node/`, or die naming how to get one. Only ever called
    with node missing or too old.

    Repo-local rather than through a package manager, because it is the one route that needs
    nothing of the machine: no admin rights (Homebrew and the nodejs.org installers both
    want them, and a managed laptop may not grant them), no Homebrew, no shell-profile edit.
    A Node unpacked by a script is normally visible to that process alone, so the UI would
    install and then vanish; this one is not, because `_common.use_local_node` puts it on
    PATH for `setup.py`, `dev.py` and everything they spawn.

    Unpacked beside its destination and renamed into place, so an interrupted download never
    leaves a `.node` that looks installed.
    """
    name = node_archive()
    if name is None:
        say(TAG, f"no official Node build for this machine; install Node {NODE_MIN_MAJOR}+ "
                 "from https://nodejs.org")
        die(TAG, "then re-run this script to add the UI.")
    url = f"https://nodejs.org/dist/v{NODE_VERSION}/{name}"
    partial = LOCAL_NODE_DIR.with_name(".node.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    archive = partial / name

    say(TAG, f"installing Node {NODE_VERSION} into {LOCAL_NODE_DIR.name}/ (~50 MB, once)…")
    try:
        urllib.request.urlretrieve(url, archive)
    except OSError as exc:
        shutil.rmtree(partial, ignore_errors=True)
        die(TAG, f"could not download {url}: {exc}")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != NODE_SHA256[name]:
        shutil.rmtree(partial, ignore_errors=True)
        die(TAG, f"{name} does not match its published checksum; nothing was installed.")

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(partial)
    else:
        # "data" keeps the tree inside `partial` and drops setuid bits; npm's own symlinks
        # (bin/npm -> ../lib/...) are relative and pass it.
        with tarfile.open(archive) as bundle:
            bundle.extractall(partial, filter="data")
    unpacked = partial / name.removesuffix(".zip").removesuffix(".tar.gz")
    shutil.rmtree(LOCAL_NODE_DIR, ignore_errors=True)
    unpacked.rename(LOCAL_NODE_DIR)
    shutil.rmtree(partial, ignore_errors=True)

    use_local_node()
    if (node_major() or 0) < NODE_MIN_MAJOR:
        die(TAG, f"Node unpacked into {LOCAL_NODE_DIR.name}/ but will not run here. Install "
                 f"Node {NODE_MIN_MAJOR}+ from https://nodejs.org and re-run this script.")


def ensure_node() -> None:
    """Node is a real prerequisite, not a nicety: it builds the frontend and installs the
    artifact components. A machine's own Node is used when it is new enough; otherwise
    `install_node` puts a private one in the repo. Reached only after the steps above, so
    the message can truthfully say the headless path already works.

    Says nothing about pnpm, which `ensure_frontend` resolves against frontend/'s own pin.
    """
    major = node_major()
    if major is None or major < NODE_MIN_MAJOR:
        say(TAG, 'everything else is ready — ask questions now with:  uv run agent "your question"')
        found = "none is installed" if major is None else f"yours is {major}"
        say(TAG, f"the chat UI needs Node {NODE_MIN_MAJOR}+ — {found}.")
        install_node()


def ensure_frontend() -> None:
    ui_dir = frontend_dir()
    # Without this the UI opens on a form asking for a deployment URL and assistant id.
    # `.env.local` because Next reads it ahead of `.env` and frontend/.gitignore ignores
    # `*.local`. The id is the graph name in langgraph.json.
    local_env = ui_dir / ".env.local"
    if not local_env.exists():
        local_env.write_text(
            "NEXT_PUBLIC_API_URL=http://localhost:2024\nNEXT_PUBLIC_ASSISTANT_ID=agent\n",
            encoding="utf-8",
        )
        say(TAG, "pointed the UI at localhost:2024")

    if (REPO_ROOT / ".chat-ui").is_dir():
        say(TAG, "note: .chat-ui/ is the old cloned chat UI and is no longer used; "
                 "delete it whenever you like.")

    # Installed again whenever the lockfile has changed since the last install, not only
    # when there is none: a pull that adds a dependency must reach node_modules too.
    modules, lockfile = ui_dir / "node_modules", ui_dir / "pnpm-lock.yaml"
    if deps_current(modules, lockfile):
        return
    pnpm = pnpm_or_die(TAG)
    say(TAG, "updating frontend dependencies…" if modules.is_dir()
        else "installing frontend dependencies (~1 min)…")
    run([*pnpm.argv, "install", "--frozen-lockfile", "--silent"], cwd=ui_dir)
    stamp_deps(modules, lockfile)


def ensure_artifact_deps() -> None:
    """The artifact components in ui/ are bundled by the *graph server*, not by the
    frontend, so their dependencies are part of wanting a UI at all rather than of
    frontend/. They fail silently when missing: the bundler logs `Could not resolve
    "xlsx"`, still answers /ui/<graph>/entrypoint.js with a 200, and the chart is simply
    absent — indistinguishable from the missing-rewrite failure. `npm ci` rather than
    `npm install` because package-lock.json is tracked for exactly this reason.

    At the repo root, where ui/ is an npm workspace: a deployment's build runs its install
    beside langgraph.json and nowhere else, so that is where the lockfile has to be for the
    image to get these too.
    """
    modules, lockfile = REPO_ROOT / "node_modules", REPO_ROOT / "package-lock.json"
    if deps_current(modules, lockfile):
        return
    # `--silent` below prints nothing at all until it finishes, so without the duration
    # this is a dead terminal for minutes at the very last step of setup.
    say(TAG, "installing artifact component dependencies (a few minutes)…")
    run(["npm", "ci", "--silent"], cwd=REPO_ROOT)
    stamp_deps(modules, lockfile)


def main() -> int:
    global _assume_yes
    parser = argparse.ArgumentParser(
        prog="uv run scripts/setup.py",
        description="One-time setup for the PubMed/PMC research agent. Safe to re-run: "
        "every step is skipped when already done.",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="never prompt; for CI and containers"
    )
    _assume_yes = parser.parse_args().yes

    ensure_env()
    ensure_deps()
    ensure_snapshot()
    ensure_node()
    ensure_frontend()
    ensure_artifact_deps()

    say(TAG, "setup complete.")
    say(TAG, "open the chat UI with:  uv run scripts/dev.py")
    say(TAG, 'or ask one question headlessly:  uv run agent "your question"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
