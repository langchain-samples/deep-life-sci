"""One-time setup. Run this once per clone, then ask questions with the chat UI.

    uv run scripts/setup.py          # prompt for the two API keys, install everything
    uv run scripts/setup.py --yes    # never prompt; for CI and containers

Four steps, in the order they depend on each other:

    1. .env        — two API keys, prompted for and written here
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
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (
    ENV_FILE,
    REPO_ROOT,
    chat_ui_dir,
    die,
    env_value,
    run,
    say,
    set_env,
    tool,
)

TAG = "setup"
WINDOWS = os.name == "nt"
UI_REPO = "https://github.com/langchain-ai/agent-chat-ui.git"

_assume_yes = False


def interactive() -> bool:
    """A prompt only makes sense with someone there to answer it."""
    return not _assume_yes and sys.stdin.isatty()


def confirm(question: str) -> bool:
    if not interactive():
        return True
    return (input(f"[{TAG}] {question} [Y/n] ").strip() or "y").lower().startswith("y")


# --- 1. .env ----------------------------------------------------------------------


def ask_key(key: str, prompt: str, prefix: str) -> None:
    """Required, loops until answered."""
    if env_value(key):
        return
    if not interactive():
        die(TAG, f"{key} is not set in .env and there is no terminal to ask on. "
                 "Add it and re-run.")
    while True:
        reply = "".join(input(f"[{TAG}] {prompt}: ").split())
        if not reply:
            continue
        # A wrong-but-plausible key is the failure this catches: a real OpenAI key in
        # OPENAI_API_KEY looks right and dies deep in the SDK at the first model call.
        # Queried rather than rejected — key formats belong to the gateway and may change.
        if prefix and not reply.startswith(prefix):
            say(TAG, f"that doesn't start with '{prefix}' — see the note in .env.example.")
            if not input(f"[{TAG}] use it anyway? [y/N] ").strip().lower().startswith("y"):
                continue
        set_env(key, reply)
        return


def ask_optional(key: str, prompt: str) -> None:
    if not interactive():
        return
    reply = "".join(input(f"[{TAG}] {prompt} (optional, Enter to skip): ").split())
    if reply:
        set_env(key, reply)


def ensure_env() -> None:
    fresh = not ENV_FILE.exists()
    if fresh:
        shutil.copy(REPO_ROOT / ".env.example", ENV_FILE)
        say(TAG, "created .env from .env.example")

    # Two keys, both from LangSmith (https://smith.langchain.com/settings), and easy to mix
    # up: model calls are billed and authenticated as *gateway* compute under a service
    # key, while tracing and sandbox provisioning use the personal API key.
    ask_key(
        "OPENAI_API_KEY",
        "LangSmith gateway service key for model calls (lsv2_sk_..., NOT an OpenAI key)",
        "lsv2_sk_",
    )
    ask_key(
        "LANGSMITH_API_KEY",
        "LangSmith API key for tracing and sandboxes (lsv2_pt_...)",
        "lsv2_",
    )

    # Only on a first run: these are genuinely optional, so re-asking every time would be
    # nagging someone who already decided to skip them.
    if fresh:
        say(TAG, "NCBI credentials are optional: they raise PubMed's rate limit "
                 "from 3 to 10 req/s.")
        ask_optional("NCBI_API_KEY", "NCBI API key")
        ask_optional("NCBI_EMAIL", "contact email for NCBI (their usage policy asks for one)")


# --- 2. dependencies --------------------------------------------------------------


def ensure_deps() -> None:
    """The dev group too, unconditionally: it is only langgraph-cli, and syncing it here is
    what keeps the chat UI from stalling on an install after it has claimed the ports."""
    say(TAG, "syncing dependencies…")
    run(["uv", "sync", "--group", "dev", "--quiet"], cwd=REPO_ROOT)


# --- 3. sandbox snapshot ----------------------------------------------------------


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
        from langsmith.sandbox import SandboxClient

        from research_agent.sandbox import SNAPSHOT_NAME

        names = {s.name for s in SandboxClient().list_snapshots(name_contains=SNAPSHOT_NAME)}
    except Exception as exc:  # noqa: BLE001 - any failure here is the same user-facing problem
        # Reaching LangSmith at all failed, so this is a credentials or connectivity
        # problem and every run would have it too. Fail here, where the cause is visible.
        print(exc, file=sys.stderr)
        die(TAG, "could not reach LangSmith. Check LANGSMITH_API_KEY in .env.")

    if SNAPSHOT_NAME in names:
        say(TAG, "sandbox snapshot ready.")
        return
    say(TAG, "building the sandbox snapshot (~100s, once)…")
    run(["uv", "run", "scripts/build_snapshot.py"], cwd=REPO_ROOT)


# --- 4. chat UI -------------------------------------------------------------------
#
# Vendored *inside* the repo, at .chat-ui, rather than beside it: a sibling directory is
# outside what the user cloned and is not necessarily writable. The cost is that
# `langgraph build` uses the repo root as its Docker context, so .chat-ui has to be listed
# in .dockerignore or a dev-only frontend ships in the deploy image.

REWRITE = """  // setup: artifact components load /ui/* from the page origin, so this proxy is
  // what makes them render at all. See CLAUDE.md.
  async rewrites() {
    return [
      { source: "/ui/:path*", destination: "http://localhost:2024/ui/:path*" },
    ];
  },
"""


def patch_next_config() -> None:
    """Reapplied on every run. It is load-bearing — without it the artifact components
    silently render nothing (see CLAUDE.md) — and it is a one-key insert. The other patch
    CLAUDE.md documents, the ai.tsx whitespace fix, is cosmetic and edits a component body,
    so it stays a manual choice rather than something rewritten underneath whoever made it.
    """
    cfg = chat_ui_dir() / "next.config.mjs"
    if not cfg.is_file():
        say(TAG, f"warning: no next.config.mjs in {chat_ui_dir()}; skipped the /ui/* rewrite.")
        return
    text = cfg.read_text(encoding="utf-8")
    if "/ui/:path" in text:
        return

    # Upstream currently has no `rewrites` key and one `const nextConfig = {` to insert
    # after. If either stops being true, print the snippet instead of guessing: a second
    # `rewrites` key would silently shadow the first rather than fail.
    anchor = "const nextConfig = {"
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if "rewrites" in text or len(hits) != 1:
        say(TAG, f"warning: {cfg} is not the shape expected. Add this to its config by hand:")
        print(REWRITE)
        return
    lines.insert(hits[0] + 1, REWRITE.rstrip())
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(TAG, "added the /ui/* rewrite to next.config.mjs")


def ensure_node() -> None:
    """Node is a real prerequisite, not a nicety: pnpm builds the frontend and npm installs
    the artifact components. pnpm is offered automatically because npm can install it in
    one command; Node itself is left to the user, since installing a language runtime
    unasked is a larger liberty. Reached only after the steps above, so the message can
    truthfully say the headless path already works.
    """
    if tool("npm") is None:
        say(TAG, 'everything else is ready — ask questions now with:  uv run agent "your question"')
        say(TAG, "the chat UI needs Node 20+.")
        # nodejs.org ships an .msi that wants administrator rights, which is exactly what a
        # managed laptop withholds — and this is the only step in setup that does. The
        # per-user managers below install into the user profile and need no elevation, so
        # naming them here is the difference between "add the UI later" and "cannot".
        if WINDOWS:
            say(TAG, "  with admin rights:  winget install OpenJS.NodeJS.LTS")
            say(TAG, "  without:            winget install Schniz.fnm  &&  fnm install 22")
            say(TAG, "  or unzip the Windows binary from https://nodejs.org onto your PATH")
        else:
            say(TAG, "  https://nodejs.org, or `brew install node` on a Mac")
        die(TAG, "install it and re-run this script to add the UI.")
    if tool("pnpm") is not None:
        return
    # agent-chat-ui pins pnpm as its package manager and ships only a pnpm lockfile, so
    # this is not a substitutable choice between equivalent tools. `-g` reads as
    # machine-wide but is not: npm's global prefix is ~/.npm-global or %APPDATA%\npm,
    # inside the user profile, so this needs no elevation either.
    if not confirm("the chat UI needs pnpm. install it with `npm install -g pnpm` ?"):
        die(TAG, "install pnpm yourself (https://pnpm.io/installation), then re-run.")
    run(["npm", "install", "-g", "pnpm"])
    if tool("pnpm") is None:
        die(TAG, "pnpm installed but not on PATH — open a new terminal and re-run this script.")


def ensure_chat_ui() -> None:
    if tool("git") is None:
        die(TAG, "the chat UI needs git.")
    ui_dir = chat_ui_dir()
    if not ui_dir.is_dir():
        say(TAG, f"cloning agent-chat-ui into {ui_dir}…")
        run(["git", "clone", "--depth", "1", "--quiet", UI_REPO, str(ui_dir)])

    patch_next_config()

    # Without this the UI opens on a form asking for a deployment URL and assistant id.
    # `.env.local` because Next reads it ahead of `.env` and upstream ignores `*.local`.
    # The id is the graph name in langgraph.json.
    local_env = ui_dir / ".env.local"
    if not local_env.exists():
        local_env.write_text(
            "NEXT_PUBLIC_API_URL=http://localhost:2024\nNEXT_PUBLIC_ASSISTANT_ID=agent\n",
            encoding="utf-8",
        )
        say(TAG, "pointed the UI at localhost:2024")

    if (ui_dir / "node_modules").is_dir():
        return
    say(TAG, "installing frontend dependencies (~1 min, once)…")
    run(["pnpm", "install", "--silent"], cwd=ui_dir)


def ensure_artifact_deps() -> None:
    """The artifact components in ui/ are bundled by the *graph server*, not by the
    frontend, so their dependencies are part of wanting a UI at all rather than of the
    clone above. They fail silently when missing: the bundler logs `Could not resolve
    "xlsx"`, still answers /ui/<graph>/entrypoint.js with a 200, and the chart is simply
    absent — indistinguishable from the missing-rewrite failure. `npm ci` rather than
    `npm install` because ui/package-lock.json is tracked for exactly this reason.
    """
    if (REPO_ROOT / "ui" / "node_modules").is_dir():
        return
    say(TAG, "installing artifact component dependencies…")
    run(["npm", "ci", "--silent"], cwd=REPO_ROOT / "ui")


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
    ensure_chat_ui()
    ensure_artifact_deps()

    say(TAG, "setup complete.")
    say(TAG, "open the chat UI with:  uv run scripts/dev.py")
    say(TAG, 'or ask one question headlessly:  uv run agent "your question"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
