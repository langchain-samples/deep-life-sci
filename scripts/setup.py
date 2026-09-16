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
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (
    ENV_FILE,
    REPO_ROOT,
    chat_ui_dir,
    die,
    env_value,
    pnpm_or_die,
    run,
    say,
    set_env,
    tool,
)

TAG = "setup"
WINDOWS = os.name == "nt"
# agent-chat-ui is on Next 16, which refuses to build below 20.9. Checked as a major so a
# .0 release of 20 is not read as too old; Next's own check catches 20.0-20.8.
NODE_MIN_MAJOR = 20
UI_REPO = "https://github.com/langchain-ai/agent-chat-ui.git"
# Pinned to a commit, never to a branch. Every patch below is anchored on an exact string
# in upstream's source, so an upstream refactor half-patches every *new* clone at once —
# and the clone is gitignored, so there is no diff anywhere that would show it. A pin
# turns that from "breaks for everyone on the day upstream lands a rename" into "breaks
# when someone deliberately bumps this line".
#
# Verified against this commit: from a virgin clone all 17 patches apply and
# `unapplied_patches()` is empty afterwards. To bump: change the SHA, `rm -rf .chat-ui`,
# re-run setup, and check it printed no warnings.
UI_REPO_REF = "d96e46f365f55f3e23ef1036fe05de7c53064897"

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
        from langsmith.sandbox import SandboxClient

        from deep_life_sci.sandbox import SNAPSHOT_NAME

        names = {s.name for s in SandboxClient().list_snapshots(name_contains=SNAPSHOT_NAME)}
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
# Vendored *inside* the repo, at .chat-ui, rather than beside it: a sibling directory is
# outside what the user cloned and is not necessarily writable. The cost is that
# `langgraph build` uses the repo root as its Docker context, so .chat-ui has to be listed
# in .dockerignore or a dev-only frontend ships in the deploy image.

# What each patch leaves behind, and where. Every patch is idempotent because it looks for
# its own mark before doing anything, and `unapplied_patches()` asks the same question from
# outside — which is what lets `dev.py` verify the patches are *present* rather than that
# setup once *ran*. The clone is gitignored, so nothing else records what the frontend is:
# a hand-edited or half-upgraded .chat-ui otherwise sits un-patched while the repo believes
# the feature shipped. One table rather than a literal in each function, so the two answers
# cannot drift apart.
#
# Mapped to False where the patch works by *removing* something.
# The name this agent goes by in the UI. Upstream ships its own throughout, and a
# half-renamed app is worse than an unrenamed one, so every occurrence moves together.
APP_NAME = "Deep Life Sci"

# Names this agent used to go by. The rename patch is a search-and-replace on upstream's own
# product name, so a clone patched under an earlier name has no `Agent Chat` left to replace
# and would sit there half-renamed with nothing saying so — the clone is gitignored, so its
# only history is this list. Append here whenever APP_NAME changes.
PRIOR_APP_NAMES = ("Life Sci Agent",)

# The empty state's logo and heading, stacked rather than side by side. Its own patch rather
# than another edit inside patch_app_name, so a clone already carrying the rename picks it up.
HOME_HEADING_OLD = '<div className="flex items-center gap-3">'
HOME_HEADING_NEW = '<div className="flex flex-col items-center gap-3">'

# The logo, cropped to the LangChain mark. Upstream's asset is one viewBox carrying both the
# mark (x 0-488) and the `LangChain` wordmark (x 591-2999), and all three places it is drawn
# set the product name directly beside or under it. Cropped to the square it is a brand mark
# next to a product name, which is a lockup. It also un-squashes the header, where
# `width={32} height={32}` on a 5.4:1 viewBox drew the whole logo into a 32px smudge.
LOGO_VIEWBOX_OLD = 'viewBox="0 0 3000 554"'
LOGO_VIEWBOX_NEW = 'viewBox="0 0 488 488"'

# The composer's attach label. Defined up here because `PATCH_MARKS` marks on it; the patch
# itself is `patch_attach_label`, which is separate from `patch_uploads` because that one only
# runs from the upstream baseline and early-outs on a clone already carrying its mark — a
# rename folded into it would never reach one. Both prior wordings are accepted so every clone
# converges on the same label.
ATTACH_LABEL_BASELINES = ("Attach a file", "Upload PDF or Image")
ATTACH_LABEL = "Attach files"

# The header logo, sized to the icons it sits between. Cropping the viewBox turned what was a
# 32px-wide smudge into a 32px *square* mark, which then towered over the 20px panel toggle on
# one side and the name on the other. Its own patch rather than part of `patch_logo_mark`:
# a different file, and a clone already carrying the crop still picks this up.
HEADER_LOGO_OLD = """<LangGraphLogoSVG
                    width={32}
                    height={32}
                  />"""
HEADER_LOGO_NEW = HEADER_LOGO_OLD.replace("{32}", "{20}")

# Keep the upstream component's local name so its dimension patches remain valid.
# The product mark itself belongs in the persistent overlay.
DEEP_HELIX_IMPORT_OLD = 'import { LangGraphLogoSVG } from "../icons/langgraph";\n'
DEEP_HELIX_IMPORT_NEW = (
    'import { DeepHelixSVG as LangGraphLogoSVG } from "../icons/deep-helix";\n'
)

PATCH_MARKS = {
    "rewrite": ("next.config.mjs", "/ui/:path", True),
    "dev-indicator": ("next.config.mjs", "devIndicators", True),
    "svg": ("src/components/icons/langgraph.tsx", "clip-path=", False),
    "logo-mark": ("src/components/icons/langgraph.tsx", LOGO_VIEWBOX_NEW, True),
    "header-logo": ("src/components/thread/index.tsx", HEADER_LOGO_NEW, True),
    "github-link": ("src/components/thread/index.tsx", "OpenGitHubRepo", False),
    "app-name": ("src/components/thread/index.tsx", APP_NAME, True),
    "home-heading": ("src/components/thread/index.tsx", HOME_HEADING_NEW, True),
    "deep-helix": ("src/components/thread/index.tsx", DEEP_HELIX_IMPORT_NEW, True),
    "empty-turns": ("src/components/thread/messages/ai.tsx", "hasCustomComponents", True),
    "uploads": ("src/hooks/use-file-upload.tsx", "isSupportedUpload", True),
    "attach-label": ("src/components/thread/index.tsx", ATTACH_LABEL, True),
    "upload-kinds": ("src/lib/multimodal-utils.ts", "isSandboxUpload", True),
    "progress-events": ("src/providers/Stream.tsx", "isProgressEvent", True),
    "progress-row": ("src/components/thread/index.tsx", "RunStatus", True),
    "thread-search": ("src/providers/Thread.tsx", "first_message", True),
    "cancel-on-stop": ("src/components/thread/index.tsx", "onDisconnect", True),
}


def _marked(name: str) -> bool:
    """Whether this patch's mark is where it left it.

    A file that is not there counts as applied: upstream moving or renaming one is not drift
    that re-running fixes, and the patch function itself is what says so — reporting it here
    as well would put the same warning on every launch.
    """
    relative, mark, present = PATCH_MARKS[name]
    path = chat_ui_dir() / relative
    if not path.is_file():
        return True
    return (mark in path.read_text(encoding="utf-8", errors="replace")) is present


def unapplied_patches() -> list[str]:
    """Names of the patches whose mark is missing from the clone."""
    return [name for name in PATCH_MARKS if not _marked(name)]


def apply_patches() -> None:
    """Every patch, in order. Safe to call on an already-patched clone."""
    patch_next_config()
    patch_dev_indicator()
    patch_svg_props()
    patch_logo_mark()
    patch_header_logo()
    patch_github_link()
    patch_app_name()
    patch_home_heading()
    patch_deep_helix()
    patch_empty_ai_turns()
    patch_uploads()
    patch_attach_label()
    patch_upload_kinds()
    patch_progress_events()
    patch_progress_row()
    patch_thread_search()
    patch_cancel_on_stop()


REWRITE = """  // setup: artifact components load /ui/* from the page origin, so this proxy is
  // what makes them render at all. See CLAUDE.md.
  async rewrites() {
    return [
      { source: "/ui/:path*", destination: "http://localhost:2024/ui/:path*" },
    ];
  },
"""


def patch_next_config() -> None:
    """Reapplied on every run, and load-bearing: without it the artifact components
    silently render nothing (see CLAUDE.md).
    """
    cfg = chat_ui_dir() / "next.config.mjs"
    if not cfg.is_file():
        say(TAG, f"warning: no next.config.mjs in {chat_ui_dir()}; skipped the /ui/* rewrite.")
        return
    if _marked("rewrite"):
        return
    text = cfg.read_text(encoding="utf-8")

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


DEV_INDICATOR = """\
  // setup: this app is run by end users through `uv run scripts/dev.py`, so Next's dev
  // overlay button in the bottom-left corner is chrome for a toolchain they are not being
  // asked to think about. See scripts/CLAUDE.md.
  devIndicators: false,
"""


def patch_dev_indicator() -> None:
    """Hide the Next dev-tools button. Its own patch rather than a second line inside
    REWRITE, so a clone that already has the rewrite still picks this up.
    """
    cfg = chat_ui_dir() / "next.config.mjs"
    if not cfg.is_file():
        return
    if _marked("dev-indicator"):
        return
    text = cfg.read_text(encoding="utf-8")

    anchor = "const nextConfig = {"
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if len(hits) != 1:
        say(TAG, f"warning: {cfg} is not the shape expected. Add this to its config by hand:")
        print(DEV_INDICATOR)
        return
    lines.insert(hits[0] + 1, DEV_INDICATOR.rstrip())
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(TAG, "hid the Next dev-tools button in next.config.mjs")


# The patches below are cosmetic rather than load-bearing, unlike the rewrite above.
# They are applied anyway because setup's job is a working app, and an app that logs a
# console error, opens on a screenful of whitespace, draws the logo's wordmark next to its
# own name, or offers an upload it then refuses is not one. Each is anchored on an exact
# upstream string and prints the change instead of guessing if that string ever moves, so an
# upstream fix is never clobbered and a stale patch never lands silently.


def patch_svg_props() -> None:
    """`clip-path` is valid SVG and invalid JSX, so upstream's logo makes React log
    `Invalid DOM property` on every render, which parks the dev overlay's error badge in the
    corner of an otherwise healthy app.
    """
    icon = chat_ui_dir() / "src" / "components" / "icons" / "langgraph.tsx"
    if not icon.is_file():
        return
    if _marked("svg"):
        return
    text = icon.read_text(encoding="utf-8")
    icon.write_text(text.replace("clip-path=", "clipPath="), encoding="utf-8")
    say(TAG, "fixed the clip-path JSX warning in langgraph.tsx")


def patch_logo_mark() -> None:
    """Crop upstream's logo to the LangChain mark, leaving the wordmark outside the viewBox.

    Cosmetic like the rest, and see `LOGO_VIEWBOX_OLD` for what it buys: every place the logo
    is drawn sets the product name beside or under it, so the mark on its own is the half that
    belongs there.
    """
    icon = chat_ui_dir() / "src" / "components" / "icons" / "langgraph.tsx"
    if not icon.is_file():
        return
    if _marked("logo-mark"):
        return
    text = icon.read_text(encoding="utf-8")
    if text.count(LOGO_VIEWBOX_OLD) != 1:
        say(TAG, f"warning: {icon} is not the shape expected; left the logo uncropped, so "
                 "it draws the wordmark as well as the mark. Narrow the viewBox by hand to "
                 "crop it. Missing anchor:")
        print(LOGO_VIEWBOX_OLD)
        return
    icon.write_text(text.replace(LOGO_VIEWBOX_OLD, LOGO_VIEWBOX_NEW), encoding="utf-8")
    say(TAG, "cropped the logo to the LangChain mark")


def patch_header_logo() -> None:
    """Shrink the header logo to the size of the icons on either side of it. Only the header:
    the home screen draws the mark on its own line, where 32px is the point of it.
    """
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("header-logo"):
        return
    text = thread.read_text(encoding="utf-8")
    if text.count(HEADER_LOGO_OLD) != 1:
        say(TAG, f"warning: {thread} is not the shape expected; left the header logo at "
                 "32px, over the icons beside it. Missing anchor:")
        print(HEADER_LOGO_OLD)
        return
    thread.write_text(text.replace(HEADER_LOGO_OLD, HEADER_LOGO_NEW), encoding="utf-8")
    say(TAG, "sized the header logo to the icons beside it")


# Both call sites plus the component and its import, since a component left behind with no
# call site is an unused-import lint error rather than dead-but-harmless code.
GITHUB_LINK_EDITS = [
    ('import { GitHubSVG } from "../icons/github";\n', ""),
    ("""import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
""", ""),
    ("""function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/langchain-ai/agent-chat-ui"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG
              width="24"
              height="24"
            />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

""", ""),
    ("""              <div className="absolute top-2 right-4 flex items-center">
                <OpenGitHubRepo />
              </div>
""", ""),
    ("""                <div className="flex items-center">
                  <OpenGitHubRepo />
                </div>
""", ""),
]


def patch_github_link() -> None:
    """Drop the header link to the agent-chat-ui repo, which points at the chat client
    rather than at this agent.
    """
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("github-link"):
        return
    text = thread.read_text(encoding="utf-8")
    for old, _ in GITHUB_LINK_EDITS:
        if text.count(old) != 1:
            say(TAG, f"warning: {thread} is not the shape expected; left the GitHub link "
                     "in the header. Missing anchor:")
            print(old)
            return
    for old, new in GITHUB_LINK_EDITS:
        text = text.replace(old, new)
    thread.write_text(text, encoding="utf-8")
    say(TAG, "removed the GitHub link from the chat UI header")


# Every name this reads is already in scope at the anchor below; it adds only its own.
EMPTY_TURN_GUARD = """
  // setup: an AI turn that is only thinking + tool_use renders no content of its own, but
  // its hover CommandBar is opacity-0 rather than absent and still occupies its row. A run
  // here is dozens of such turns, so unpatched the first visible output sits about a
  // screenful below the question. See scripts/CLAUDE.md.
  const hasCustomComponents = !!thread.values.ui?.some(
    (ui) => ui.metadata?.message_id === message?.id,
  );
  if (
    !isToolResult &&
    !threadInterrupt &&
    contentString.length === 0 &&
    !hasCustomComponents &&
    (hideToolCalls || (!hasToolCalls && !hasAnthropicToolCalls))
  ) {
    return null;
  }
"""

# The tool-result guard, which is the last statement before the component's own `return (`.
EMPTY_TURN_ANCHOR = """  if (isToolResult && hideToolCalls) {
    return null;
  }
"""


def patch_empty_ai_turns() -> None:
    ai = chat_ui_dir() / "src" / "components" / "thread" / "messages" / "ai.tsx"
    if not ai.is_file():
        return
    if _marked("empty-turns"):
        return
    text = ai.read_text(encoding="utf-8")
    if text.count(EMPTY_TURN_ANCHOR) != 1:
        say(TAG, f"warning: {ai} is not the shape expected. Add this to AssistantMessage, "
                 "after its `isToolResult && hideToolCalls` guard, by hand:")
        print(EMPTY_TURN_GUARD)
        return
    text = text.replace(EMPTY_TURN_ANCHOR, EMPTY_TURN_ANCHOR + EMPTY_TURN_GUARD)
    ai.write_text(text, encoding="utf-8")
    say(TAG, "collapsed the empty thinking-only AI turns in ai.tsx")


# Upstream accepts JPEG/PNG/GIF/WEBP and PDF, all of which ride in model context and nothing
# more. This agent accepts CSV/TSV/xlsx instead, because those are the attachments it can
# actually compute over: `middleware/uploads.py` lifts the payload out of the human message
# before the first model call, stores it per thread, and materialises it in the sandbox at
# /workspace/uploads. So the block travelling through the message is transport, not context.
#
# Images and PDFs stay refused — reopening them means giving them somewhere to go first.
#
# Two baselines have to work: upstream, and a clone already carrying the earlier "nothing is
# accepted yet" patch. The header anchor therefore matches either, and the toast/composer
# rewrites are skipped when that earlier patch already made them.
UPLOAD_HEADER_UPSTREAM = """export const SUPPORTED_FILE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "application/pdf",
];"""

UPLOAD_HEADER_REFUSED = """\
// setup: nothing is accepted yet. An attachment reaches the model as context and never
// reaches the sandbox, so a CSV cannot be computed over — which is the only upload worth
// having here. Emptying the list routes every attempt into the toast below, which says so
// rather than listing types the agent has no use for. See scripts/CLAUDE.md.
export const SUPPORTED_FILE_TYPES: string[] = [];

export const UNSUPPORTED_FILE_TITLE = "Attachments aren't supported yet";
export const UNSUPPORTED_FILE_BODY =
  "Spreadsheet and CSV upload is coming soon. Papers, figures and trial records the " +
  "agent fetches for itself — just ask for them.";"""

UPLOAD_HEADER = """\
// setup: CSV/TSV/xlsx only. Those are the attachments the agent can do something with —
// they do not stay in model context, `deep_life_sci/middleware/uploads.py` moves them into
// the sandbox at /workspace/uploads and keeps them there across turns. An image or a PDF
// would be context and nothing else, so both stay refused. See scripts/CLAUDE.md.
export const SUPPORTED_FILE_TYPES: string[] = [...SPREADSHEET_TYPES];

// Every call site below tests these rather than the list, because a MIME-only check rejects
// the file the user came to attach: Windows with Excel installed reports a .csv as
// `application/vnd.ms-excel`, and some browsers report "" or application/octet-stream.
export function isSupportedUpload(file: File): boolean {
  return isSpreadsheetUpload(file);
}

// Which uploads become a `type: "file"` block rather than an image one. Everything accepted
// here is one, so the image branch beside each call site is now unreachable rather than
// wrong; the name is what keeps those call sites legible if an image type ever comes back.
export function isFileBlockUpload(file: File): boolean {
  return isSpreadsheetUpload(file);
}

export const UNSUPPORTED_FILE_TITLE = "That file type isn't supported";
export const UNSUPPORTED_FILE_BODY = "Upload types limited to CSV, TSV or .xlsx";"""

# The helpers live in lib/ rather than in the hook because `fileToContentBlock` needs them
# too and the hook already imports from there — the other direction would be a cycle.
UPLOAD_HELPERS = """\
// setup: a spreadsheet is the one attachment worth having here, and it does not reach the
// model. It rides in as a file block carrying its filename, and the graph takes it back out
// (see deep_life_sci/middleware/uploads.py).
//
// Extension first and MIME second, deliberately — see the note in use-file-upload.tsx.
// `.xls` is in neither list: reading it needs xlrd, which is not in the sandbox snapshot,
// and the sandbox blocks runtime installs, so it is refused at the composer rather than
// failing deep inside a run. `application/vnd.ms-excel` is left out for the same reason,
// even though a Windows .csv arrives claiming it — the extension check has already passed
// that one by the time MIME is consulted.
export const SPREADSHEET_SUFFIXES = [".csv", ".tsv", ".xlsx", ".xlsm"];

export const SPREADSHEET_TYPES = [
  "text/csv",
  "text/tab-separated-values",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel.sheet.macroEnabled.12",
];

export function isSpreadsheetUpload(file: File): boolean {
  const name = file.name.toLowerCase();
  if (SPREADSHEET_SUFFIXES.some((suffix) => name.endsWith(suffix))) return true;
  return SPREADSHEET_TYPES.includes(file.type);
}

// Normalised off the extension, because the browser's value is the unreliable half and the
// server keys on the extension as well.
export function spreadsheetMimeType(file: File): string {
  const name = file.name.toLowerCase();
  if (name.endsWith(".xlsx"))
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  if (name.endsWith(".xlsm")) return "application/vnd.ms-excel.sheet.macroEnabled.12";
  if (name.endsWith(".tsv")) return "text/tab-separated-values";
  return "text/csv";
}

"""

_LIB_ANCHOR = (
    "// Returns a Promise of a typed multimodal block for images or PDFs\n"
    "export async function fileToContentBlock("
)

# Anchors that must be present whichever baseline we start from. Order matters only in that
# the header goes in before anything references the helpers it declares.
UPLOAD_EDITS = [
    # The hook reaches the helpers through the import it already has.
    (
        "hook",
        'import { fileToContentBlock } from "@/lib/multimodal-utils";',
        "import {\n"
        "  fileToContentBlock,\n"
        "  isSpreadsheetUpload,\n"
        "  spreadsheetMimeType,\n"
        "  SPREADSHEET_TYPES,\n"
        '} from "@/lib/multimodal-utils";',
    ),
    # Eight call sites — picker, drop and paste each filter twice, plus the two duplicate
    # checks. One literal covers them all, which is why the helper exists at all.
    ("hook", "SUPPORTED_FILE_TYPES.includes(file.type)", "isSupportedUpload(file)"),
    # The duplicate checks: upstream's file-block branch is PDF-only, and its inner
    # comparison hardcodes the same type. Both appear twice, in `isDuplicate` and again in
    # the copy inlined into the paste handler.
    ("hook", 'file.type === "application/pdf"', "isFileBlockUpload(file)"),
    # `file.type` would be the obvious replacement and is the one thing that cannot go here:
    # the block was built with the normalised type, so on the Windows .csv that arrives as
    # `application/vnd.ms-excel` the two sides never match and the dedupe silently stops.
    (
        "hook",
        'b.mimeType === "application/pdf" &&',
        "b.mimeType === spreadsheetMimeType(file) &&",
    ),
    # lib: the helpers, then the branch that turns a spreadsheet into a file block.
    # Anchored on upstream's comment as well as the signature, so the helpers go in above it
    # rather than between it and the function it describes.
    (
        "lib",
        _LIB_ANCHOR,
        UPLOAD_HELPERS + _LIB_ANCHOR,
    ),
    (
        "lib",
        '  const supportedFileTypes = [...supportedImageTypes, "application/pdf"];\n'
        "\n"
        "  if (!supportedFileTypes.includes(file.type)) {",
        '  const supportedFileTypes = [...supportedImageTypes, "application/pdf"];\n'
        "\n"
        "  if (isSpreadsheetUpload(file)) {\n"
        "    return {\n"
        '      type: "file",\n'
        "      mimeType: spreadsheetMimeType(file),\n"
        "      data: await fileToBase64(file),\n"
        "      metadata: { filename: file.name },\n"
        "    };\n"
        "  }\n"
        "\n"
        "  if (!supportedFileTypes.includes(file.type)) {",
    ),
    # Without this the composer shows no chip for an attached CSV: the preview is filtered
    # through this guard, and a spreadsheet block satisfies neither existing branch.
    (
        "lib",
        "  // file type (legacy)",
        "  // spreadsheet type — transport for the graph rather than model context\n"
        "  if (\n"
        '    (block as { type: unknown }).type === "file" &&\n'
        '    "mimeType" in block &&\n'
        '    typeof (block as { mimeType?: unknown }).mimeType === "string" &&\n'
        "    SPREADSHEET_TYPES.includes((block as { mimeType: string }).mimeType)\n"
        "  ) {\n"
        "    return true;\n"
        "  }\n"
        "  // file type (legacy)",
    ),
    # The chip. Generalising the PDF branch to any file block is a strict widening — a PDF
    # still lands in it — and it is the whole of what a CSV needs to render by name.
    (
        "preview",
        '  // PDF block\n  if (block.type === "file" && block.mimeType === "application/pdf") {',
        "  // Any file block: PDF, or a spreadsheet on its way to the sandbox\n"
        '  if (block.type === "file" && typeof block.mimeType === "string") {',
    ),
    ("preview", '|| "PDF file";', '|| "attached file";'),
    ("preview", 'aria-label="Remove PDF"', 'aria-label="Remove file"'),
]

# Applied only from the upstream baseline; the earlier patch already made all four. `accept`
# has to stay `*/*` in both: with a real list the picker greys out the .xls a user is about
# to be told to re-save, so the attempt never happens and no message is ever shown.
# One per call site: the picker and drop handlers share one message, paste has its own.
_UPSTREAM_TOAST_UPLOAD = (
    '"You have uploaded invalid file type. Please upload a JPEG, PNG, GIF, WEBP image or a PDF."'
)
_UPSTREAM_TOAST_PASTE = (
    '"You have pasted an invalid file type. Please paste a JPEG, PNG, GIF, WEBP image or a PDF."'
)
_TOAST_REPLACEMENT = "UNSUPPORTED_FILE_TITLE, { description: UNSUPPORTED_FILE_BODY }"

UPLOAD_EDITS_FROM_UPSTREAM = [
    ("hook", _UPSTREAM_TOAST_UPLOAD, _TOAST_REPLACEMENT),
    ("hook", _UPSTREAM_TOAST_PASTE, _TOAST_REPLACEMENT),
    (
        "composer",
        'accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"',
        'accept="*/*"',
    ),
    ("composer", "Upload PDF or Image", "Attach a file"),
]


def patch_uploads() -> None:
    """Open the composer to the one attachment the agent can use: a CSV or a spreadsheet."""
    paths = {
        "hook": chat_ui_dir() / "src" / "hooks" / "use-file-upload.tsx",
        "lib": chat_ui_dir() / "src" / "lib" / "multimodal-utils.ts",
        "preview": chat_ui_dir() / "src" / "components" / "thread" / "MultimodalPreview.tsx",
        "composer": chat_ui_dir() / "src" / "components" / "thread" / "index.tsx",
    }
    if any(not path.is_file() for path in paths.values()):
        say(TAG, "warning: the chat UI is not the shape expected; left file uploads alone.")
        return

    contents = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}
    if _marked("uploads"):
        return

    header = next(
        (h for h in (UPLOAD_HEADER_UPSTREAM, UPLOAD_HEADER_REFUSED) if h in contents["hook"]),
        None,
    )
    if header is None:
        say(TAG, f"warning: {paths['hook']} is not the shape expected; left file uploads "
                 "alone. CSV and spreadsheet attachments will bounce. Missing anchor:")
        print(UPLOAD_HEADER_UPSTREAM)
        return

    edits = [("hook", header, UPLOAD_HEADER), *UPLOAD_EDITS]
    if header is UPLOAD_HEADER_UPSTREAM:
        edits += UPLOAD_EDITS_FROM_UPSTREAM

    # All or nothing, and for a sharper reason than the other patches: half of this is the
    # UI accepting a file and the other half is the graph being told about it. A partial
    # apply is an upload that silently goes nowhere.
    for key, old, _ in edits:
        if old not in contents[key]:
            say(TAG, f"warning: {paths[key]} is not the shape expected; left file uploads "
                     "alone. CSV and spreadsheet attachments will bounce. Missing anchor:")
            print(old)
            return
    for key, old, new in edits:
        contents[key] = contents[key].replace(old, new)
    for key, text in contents.items():
        paths[key].write_text(text, encoding="utf-8")
    say(TAG, "opened CSV/TSV/xlsx uploads in the chat UI")


# `patch_uploads` above opened the composer to spreadsheets. This widens it to everything
# `deep_life_sci/middleware/uploads.py` now has a reader for — bibliographies, PDFs,
# compound sets, sequences and images — and is a separate patch rather than an edit inside
# that one so a clone already carrying the spreadsheet allowlist picks the rest up. Its
# anchors are therefore `patch_uploads`'s own output, verbatim.
#
# The other half of the widening is that **every** accepted upload now becomes a
# `type: "file"` block, images included. Upstream sends an image as a `type: "image"` block
# because upstream wants it in model context; here it is transport to the sandbox, where
# `figure-analyst` reads it from a path for a fraction of what the root model pays to look
# at it. One block type is also one contract for the graph to strip.
UPLOAD_HELPERS_V2 = """\
// setup: the attachments this agent can do something with. None of them stay in model
// context — each rides in as a file block carrying its filename, and the graph takes the
// payload back out and materialises it in the sandbox (deep_life_sci/middleware/uploads.py,
// whose UPLOAD_KINDS is the server-side half of this list).
//
// Extension first and MIME second, deliberately — see the note in use-file-upload.tsx.
// `.xls` is in neither list: reading it needs xlrd, which is not in the sandbox snapshot,
// and the sandbox blocks runtime installs, so it is refused at the composer rather than
// failing deep inside a run. `application/vnd.ms-excel` is left out for the same reason,
// even though a Windows .csv arrives claiming it — the extension check has already passed
// that one by the time MIME is consulted.
export const UPLOAD_SUFFIXES = [
  // tables, gzipped or not
  ".csv", ".tsv", ".txt", ".xlsx", ".xlsm",
  ".csv.gz", ".tsv.gz", ".txt.gz",
  // bibliographies
  ".nbib", ".medline", ".ris", ".bib", ".bibtex",
  // a paper the agent cannot fetch for itself
  ".pdf",
  // compound sets
  ".sdf", ".sdf.gz", ".mol", ".smi", ".smiles",
  // sequences
  ".fasta", ".fa", ".fna", ".faa", ".fasta.gz", ".gb", ".gbk", ".genbank",
  // figures, gels, panels
  ".png", ".jpg", ".jpeg", ".gif", ".webp",
];

// MIME is the fallback, so this only needs the types a browser reliably reports for the
// list above. Anything it gets wrong is caught by the extension check first.
export const UPLOAD_TYPES = [
  "text/csv",
  "text/plain",
  "text/tab-separated-values",
  "application/gzip",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel.sheet.macroEnabled.12",
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
];

export function isSandboxUpload(file: File): boolean {
  const name = file.name.toLowerCase();
  if (UPLOAD_SUFFIXES.some((suffix) => name.endsWith(suffix))) return true;
  return UPLOAD_TYPES.includes(file.type);
}

// Normalised off the extension, because the browser's value is the unreliable half and the
// server keys on the extension as well. A type the browser reported is better than nothing
// for anything not listed here — the graph never reads it, but the dedupe check does.
export function uploadMimeType(file: File): string {
  const name = file.name.toLowerCase();
  if (name.endsWith(".xlsx"))
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  if (name.endsWith(".xlsm")) return "application/vnd.ms-excel.sheet.macroEnabled.12";
  if (name.endsWith(".pdf")) return "application/pdf";
  if (name.endsWith(".gz")) return "application/gzip";
  if (name.endsWith(".tsv")) return "text/tab-separated-values";
  if (name.endsWith(".csv")) return "text/csv";
  if (name.endsWith(".png")) return "image/png";
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
  if (name.endsWith(".gif")) return "image/gif";
  if (name.endsWith(".webp")) return "image/webp";
  return file.type || "text/plain";
}

"""

UPLOAD_HEADER_V2 = """\
// setup: everything deep_life_sci/middleware/uploads.py has a reader for. None of it is
// model context — the graph lifts the payload off the human message before the first model
// call and materialises it in the sandbox at /workspace/uploads, so a table gets computed
// over, a bibliography becomes a corpus, and a PDF or an image is read by a cheap subagent
// instead of by the root model. See scripts/CLAUDE.md.
export const SUPPORTED_FILE_TYPES: string[] = [...UPLOAD_TYPES];

// Every call site below tests these rather than the list, because a MIME-only check rejects
// the file the user came to attach: Windows with Excel installed reports a .csv as
// `application/vnd.ms-excel`, and some browsers report "" or application/octet-stream.
export function isSupportedUpload(file: File): boolean {
  return isSandboxUpload(file);
}

// Which uploads become a `type: "file"` block rather than an image one: all of them. An
// image is transport to the sandbox here rather than model context, so it takes the same
// shape as everything else and the graph has one block type to strip.
export function isFileBlockUpload(file: File): boolean {
  return isSandboxUpload(file);
}

export const UNSUPPORTED_FILE_TITLE = "That file type isn't supported";
export const UNSUPPORTED_FILE_BODY =
  "Tables, bibliographies (.nbib/.ris/.bib), PDFs, .sdf/.smi, FASTA/GenBank and images.";"""

# `fileToContentBlock`'s spreadsheet branch, as patch_uploads leaves it, and the branch
# that replaces it. The image branch below it becomes unreachable rather than wrong — an
# accepted image is caught here first, and the name of the guard is what keeps that legible.
_LIB_BRANCH_OLD = (
    "  if (isSpreadsheetUpload(file)) {\n"
    "    return {\n"
    '      type: "file",\n'
    "      mimeType: spreadsheetMimeType(file),\n"
    "      data: await fileToBase64(file),\n"
    "      metadata: { filename: file.name },\n"
    "    };\n"
    "  }\n"
)
_LIB_BRANCH_NEW = (
    "  if (isSandboxUpload(file)) {\n"
    "    return {\n"
    '      type: "file",\n'
    "      mimeType: uploadMimeType(file),\n"
    "      data: await fileToBase64(file),\n"
    "      metadata: { filename: file.name },\n"
    "    };\n"
    "  }\n"
)

# The header block above is also this patch's anchor, so rewording a comment inside it
# strands every clone patched before the reword: patch_uploads' mark is set, so nothing
# rewrites theirs. Accept the earlier wording too, the way ATTACH_LABEL_BASELINES does.
UPLOAD_HEADER_BASELINES = [
    UPLOAD_HEADER,
    UPLOAD_HEADER.replace("See scripts/CLAUDE.md.", "See CLAUDE.md."),
]

UPLOAD_KIND_EDITS = [
    (
        "hook",
        "import {\n"
        "  fileToContentBlock,\n"
        "  isSpreadsheetUpload,\n"
        "  spreadsheetMimeType,\n"
        "  SPREADSHEET_TYPES,\n"
        '} from "@/lib/multimodal-utils";',
        "import {\n"
        "  fileToContentBlock,\n"
        "  isSandboxUpload,\n"
        "  uploadMimeType,\n"
        "  UPLOAD_TYPES,\n"
        '} from "@/lib/multimodal-utils";',
    ),
    ("hook", UPLOAD_HEADER_BASELINES, UPLOAD_HEADER_V2),
    (
        "hook",
        "b.mimeType === spreadsheetMimeType(file) &&",
        "b.mimeType === uploadMimeType(file) &&",
    ),
    ("lib", UPLOAD_HELPERS, UPLOAD_HELPERS_V2),
    ("lib", _LIB_BRANCH_OLD, _LIB_BRANCH_NEW),
    (
        "lib",
        "    SPREADSHEET_TYPES.includes((block as { mimeType: string }).mimeType)\n",
        "    UPLOAD_TYPES.includes((block as { mimeType: string }).mimeType)\n",
    ),
    (
        "lib",
        "  // spreadsheet type — transport for the graph rather than model context\n",
        "  // any accepted upload — transport for the graph rather than model context\n",
    ),
]


def patch_attach_label() -> None:
    """Say `Attach files`, plural: the input is `multiple` and the agent reads a whole set."""
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("attach-label"):
        return
    text = thread.read_text(encoding="utf-8")
    baseline = next((b for b in ATTACH_LABEL_BASELINES if text.count(b) == 1), None)
    if baseline is None:
        say(TAG, f"warning: {thread} is not the shape expected; left the composer's attach "
                 "label alone. Missing anchor:")
        print(ATTACH_LABEL_BASELINES[0])
        return
    thread.write_text(text.replace(baseline, ATTACH_LABEL), encoding="utf-8")
    say(TAG, f"renamed the composer's attach label to {ATTACH_LABEL!r}")


def patch_upload_kinds() -> None:
    """Widen the composer from spreadsheets to every attachment the agent has a reader for."""
    paths = {
        "hook": chat_ui_dir() / "src" / "hooks" / "use-file-upload.tsx",
        "lib": chat_ui_dir() / "src" / "lib" / "multimodal-utils.ts",
    }
    if any(not path.is_file() for path in paths.values()):
        return
    if _marked("upload-kinds"):
        return

    contents = {key: path.read_text(encoding="utf-8") for key, path in paths.items()}

    # All or nothing, for the same reason patch_uploads is: half of this is the UI accepting
    # a file and the other half is the block shape the graph strips. A partial apply is an
    # upload that silently goes nowhere. An `old` may be a list of acceptable baselines.
    edits = []
    for key, old, new in UPLOAD_KIND_EDITS:
        baselines = old if isinstance(old, list) else [old]
        found = next((b for b in baselines if b in contents[key]), None)
        if found is None:
            say(TAG, f"warning: {paths[key]} is not the shape expected; left the upload "
                     "allowlist at spreadsheets only. Missing anchor:")
            print(baselines[0])
            return
        edits.append((key, found, new))
    for key, old, new in edits:
        contents[key] = contents[key].replace(old, new)
    for key, text in contents.items():
        paths[key].write_text(text, encoding="utf-8")
    say(TAG, "widened chat UI uploads to bibliographies, PDFs, chemistry, sequences and images")


# The agent does its work inside one `eval` call, so the transcript shows a single tool call
# that has not returned yet — and with tool calls hidden, nothing at all. The graph narrates
# the calls happening inside it over the custom-event channel instead
# (deep_life_sci/middleware/progress.py); these two patches carry that line to the screen.
# Split in two because they fail differently: without the first there is no event to read,
# without the second the events arrive and nothing renders them.
PROGRESS_TYPES = """\
// setup: the agent orchestrates inside one `eval`, so a multi-minute run produces no visible
// message until it is over. It narrates itself over the same custom-event channel the UI
// components already use — see deep_life_sci/middleware/progress.py — and the latest line
// travels to the thread view through the context below.
export type ProgressEvent = { type: "progress"; text: string };

function isProgressEvent(event: unknown): event is ProgressEvent {
  return (
    typeof event === "object" &&
    event !== null &&
    (event as { type?: unknown }).type === "progress" &&
    typeof (event as { text?: unknown }).text === "string"
  );
}

// A context of its own rather than another key on the stream value: that value is the SDK
// hook's return, and spreading it to add one would fix whatever it computes per render.
const RunProgressContext = createContext<string | null>(null);
export const useRunProgress = (): string | null => useContext(RunProgressContext);

"""

PROGRESS_PROVIDER = """\
  // A finished run's last line is not progress any more. Clearing on `isLoading` also covers
  // the run that failed, where no completion event is coming.
  useEffect(() => {
    if (!streamValue.isLoading) setRunProgress(null);
  }, [streamValue.isLoading]);

  return (
    <StreamContext.Provider value={streamValue}>
      <RunProgressContext.Provider value={runProgress}>
        {children}
      </RunProgressContext.Provider>
    </StreamContext.Provider>
  );
"""

PROGRESS_EDITS = [
    ("export type StateType = { messages: Message[]; ui?: UIMessage[] };\n\n",
     "export type StateType = { messages: Message[]; ui?: UIMessage[] };\n\n" + PROGRESS_TYPES),
    ("    CustomEventType: UIMessage | RemoveUIMessage;",
     "    CustomEventType: UIMessage | RemoveUIMessage | ProgressEvent;"),
    ('  const { getThreads, setThreads } = useThreads();',
     '  const { getThreads, setThreads } = useThreads();\n'
     '  const [runProgress, setRunProgress] = useState<string | null>(null);'),
    ("    onCustomEvent: (event, options) => {\n"
     "      if (isUIMessage(event) || isRemoveUIMessage(event)) {",
     "    onCustomEvent: (event, options) => {\n"
     "      if (isProgressEvent(event)) {\n"
     "        setRunProgress(event.text);\n"
     "        return;\n"
     "      }\n"
     "      if (isUIMessage(event) || isRemoveUIMessage(event)) {"),
    ("  return (\n"
     "    <StreamContext.Provider value={streamValue}>\n"
     "      {children}\n"
     "    </StreamContext.Provider>\n"
     "  );\n",
     PROGRESS_PROVIDER),
]


def patch_progress_events() -> None:
    """Receive the graph's progress events and publish the latest one."""
    stream = chat_ui_dir() / "src" / "providers" / "Stream.tsx"
    if not stream.is_file():
        return
    if _marked("progress-events"):
        return
    text = stream.read_text(encoding="utf-8")
    for old, _ in PROGRESS_EDITS:
        if text.count(old) != 1:
            say(TAG, f"warning: {stream} is not the shape expected; left run progress alone. "
                     "Runs will show no status while they work. Missing anchor:")
            print(old)
            return
    for old, new in PROGRESS_EDITS:
        text = text.replace(old, new)
    stream.write_text(text, encoding="utf-8")
    say(TAG, "wired the graph's progress events into the chat UI")


# Upstream drops the typing dots as soon as any AI message arrives, which here is the first
# `eval` — about two seconds into a run that lasts minutes. The replacement is a component of
# ours rather than JSX spliced in here: see `ensure_overlay` below for why. What is left is
# the smallest anchored change that can mount it.
PROGRESS_ROW_ANCHOR = """\
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
"""

# `firstTokenReceived` existed to hide those dots, and `prevMessageLength` existed to set it;
# with the dots gone every reader of both is gone too, and what is left is state that is
# written on three paths and read on none. Removed rather than left in place — it looks like
# it still governs the loading indicator, and the next person to touch this file would have
# to work out that it does not.
PROGRESS_ROW_EDITS = [
    ('import { useStreamContext } from "@/providers/Stream";',
     'import { useStreamContext } from "@/providers/Stream";\n'
     'import { RunStatus } from "./RunStatus";'),
    ('import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";',
     'import { AssistantMessage } from "./messages/ai";'),
    ("  const [firstTokenReceived, setFirstTokenReceived] = useState(false);\n", ""),
    ("""  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

""", ""),
    ("      return;\n    setFirstTokenReceived(false);\n", "      return;\n"),
    ("""    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
""", ""),
    (PROGRESS_ROW_ANCHOR, "                  <RunStatus />\n"),
]


def patch_progress_row() -> None:
    """Mount the run status component, and clear out what upstream's dots left behind."""
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("progress-row"):
        return
    text = thread.read_text(encoding="utf-8")
    for old, _ in PROGRESS_ROW_EDITS:
        if text.count(old) != 1:
            say(TAG, f"warning: {thread} is not the shape expected; left the status row "
                     "alone. Runs will show no status while they work. Missing anchor:")
            print(old)
            return
    for old, new in PROGRESS_ROW_EDITS:
        text = text.replace(old, new)
    thread.write_text(text, encoding="utf-8")
    say(TAG, "mounted the run status row in the chat UI")


# The thread sidebar renders one line per thread: the first human message. Upstream asks
# for whole threads to get it, and whole threads here means the QuickJS heap snapshot —
# `_quickjs_snapshot_payload`, up to 11 MB of base64 per thread and 118 MB of a measured
# 125 MB response for 40 threads, all of it parsed in the browser before the list paints.
# The server answers that in 0.2s; the wait is entirely the wire and the parse.
#
# `select` drops `values` from the response and `extract` pulls back the one message the
# list needs, which is then reshaped into the `values.messages` shape the list already
# reads — so `components/thread/history/index.tsx` stays untouched. Same 100 threads:
# 125,523,841 bytes -> ~13 KB.
#
# Not a fix for this repo's agent so much as for any agent that keeps bytes in state; it
# is upstream's code unchanged. `select` and `extract` are recent API additions, so the
# call falls back to the plain search if the server rejects them.
THREAD_SEARCH_OLD = """\
    const threads = await client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
      },
      limit: 100,
    });

    return threads;
"""

THREAD_SEARCH_NEW = """\
    // setup: the sidebar needs exactly one field of thread state — the first human
    // message, for the title. Asking for all of `values` drags this agent's QuickJS
    // heap snapshot down with it (125 MB for 40 threads, measured). See CLAUDE.md.
    const query = {
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
      },
      limit: 100,
    };
    let threads: Thread[];
    try {
      threads = await client.threads.search({
        ...query,
        select: ["thread_id", "created_at", "updated_at", "metadata", "status"],
        extract: { first_message: "values.messages[0]" },
      });
    } catch {
      // Older servers have neither `select` nor `extract`. Slow beats empty.
      threads = await client.threads.search(query);
    }

    // Put the extracted message back where the thread list already looks for it.
    return threads.map((t) =>
      t.extracted?.first_message
        ? ({ ...t, values: { messages: [t.extracted.first_message] } } as Thread)
        : t,
    );
"""


def patch_thread_search() -> None:
    """Stop the thread sidebar downloading every thread's full state to render its titles."""
    provider = chat_ui_dir() / "src" / "providers" / "Thread.tsx"
    if not provider.is_file():
        return
    if _marked("thread-search"):
        return
    text = provider.read_text(encoding="utf-8")
    if text.count(THREAD_SEARCH_OLD) != 1:
        say(TAG, f"warning: {provider} is not the shape expected; left the thread search "
                 "alone. The sidebar will download every thread's full state — slow to "
                 "open, and slower the more threads there are. Missing anchor:")
        print(THREAD_SEARCH_OLD)
        return
    provider.write_text(text.replace(THREAD_SEARCH_OLD, THREAD_SEARCH_NEW), encoding="utf-8")
    say(TAG, "narrowed the chat UI's thread search to the fields the sidebar renders")


# `stream.stop()` aborts the client's own fetch and nothing else: it does not call the
# runs cancel endpoint. The server's default for a dropped stream is `on_disconnect:
# "continue"`, so pressing stop leaves the run executing, the thread `busy`, and every
# later turn on that thread queued behind a run the user believes they killed — which
# also makes any server-side stall look like a stuck UI.
#
# `onDisconnect` is a per-submit option, not a hook-level one, so it goes on each
# `stream.submit` call rather than on `useTypedStream`.
#
# The cost is paid against `streamResumable: true`, which is here so a reload or a
# backgrounded tab can rejoin a run in progress. "cancel" cannot tell a deliberate stop
# from an accidental drop, so a refresh mid-run now ends that run instead of rejoining it.
# For this agent that is the right trade — a run holds a sandbox and bills for it, and a
# turn is minutes long, so an abandoned one is expensive in a way a lost rejoin is not.
# The version that keeps both passes an explicit `runId` on submit and has the stop button
# call `client.runs.cancel` with it, leaving disconnects on "continue"; do that instead if
# rejoining turns out to matter more than reclaiming the container.
CANCEL_ON_STOP_OLD = """      {
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({"""

CANCEL_ON_STOP_NEW = """      {
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        // setup: stop() only aborts the client stream; without this the run keeps
        // going server-side and the thread stays busy. See setup.py.
        onDisconnect: "cancel",
        optimisticValues: (prev) => ({"""

CANCEL_ON_STOP_REGEN_OLD = """    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });"""

CANCEL_ON_STOP_REGEN_NEW = """    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
      onDisconnect: "cancel",
    });"""


def patch_cancel_on_stop() -> None:
    """Make the stop button actually cancel the run, not just the client's stream."""
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("cancel-on-stop"):
        return
    text = thread.read_text(encoding="utf-8")
    missing = [
        old
        for old in (CANCEL_ON_STOP_OLD, CANCEL_ON_STOP_REGEN_OLD)
        if text.count(old) != 1
    ]
    if missing:
        say(TAG, f"warning: {thread} is not the shape expected; left the stop button "
                 "aborting only the client stream. A stopped run will keep executing and "
                 "hold its thread busy. Missing anchor:")
        for old in missing:
            print(old)
        return
    text = text.replace(CANCEL_ON_STOP_OLD, CANCEL_ON_STOP_NEW)
    text = text.replace(CANCEL_ON_STOP_REGEN_OLD, CANCEL_ON_STOP_REGEN_NEW)
    thread.write_text(text, encoding="utf-8")
    say(TAG, "wired the chat UI's stop button to cancel the run server-side")


def patch_app_name() -> None:
    """Put this agent's name on the app in place of upstream's.

    Three files rather than one: the header and empty-state heading are what a user reads,
    but leaving the browser tab and the setup form saying `Agent Chat` is how a rename looks
    half-done. Unanchored on purpose — a bare rename of a string upstream may move around.
    """
    tagline = ("Agent Chat UX by LangChain", f"{APP_NAME}, a Deep Agents demo")
    files = (
        ("src/app/layout.tsx", (tagline,)),
        ("src/providers/Stream.tsx", ()),
        ("src/components/thread/index.tsx", ()),
    )
    if _marked("app-name"):
        return
    touched = False
    for relative, extra in files:
        path = chat_ui_dir() / relative
        if not path.is_file():
            continue
        text = original = path.read_text(encoding="utf-8")
        for old, new in extra:
            text = text.replace(old, new)
        for stale in ("Agent Chat", *PRIOR_APP_NAMES):
            text = text.replace(stale, APP_NAME)
        if text != original:
            path.write_text(text, encoding="utf-8")
            touched = True
    if touched:
        say(TAG, f"renamed the chat UI to {APP_NAME}")


def patch_home_heading() -> None:
    """Stack the empty state's name under the logo instead of beside it."""
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file():
        return
    if _marked("home-heading"):
        return
    text = thread.read_text(encoding="utf-8")
    if text.count(HOME_HEADING_OLD) != 1:
        say(TAG, f"warning: {thread} is not the shape expected; left the home screen "
                 "heading beside the logo. Missing anchor:")
        print(HOME_HEADING_OLD)
        return
    thread.write_text(text.replace(HOME_HEADING_OLD, HOME_HEADING_NEW), encoding="utf-8")
    say(TAG, "stacked the home screen heading under the logo")


def patch_deep_helix() -> None:
    """Use Deep helix in the header and empty state, removing the old extra flask.

    Accept both a fresh upstream clone and the previous flask-decorated headings.
    Keeping the local LangGraphLogoSVG alias preserves the independent size patch.
    The component uses a transparent cutout and theme colors on every surface.
    """
    thread = chat_ui_dir() / "src" / "components" / "thread" / "index.tsx"
    if not thread.is_file() or _marked("deep-helix"):
        return
    text = thread.read_text(encoding="utf-8")
    if text.count(DEEP_HELIX_IMPORT_OLD) != 1:
        say(TAG, f"warning: {thread} is not the shape expected; left the logo unchanged. "
                 "Missing anchor:")
        print(DEEP_HELIX_IMPORT_OLD)
        return
    text = text.replace(DEEP_HELIX_IMPORT_OLD, DEEP_HELIX_IMPORT_NEW)
    text = text.replace('import { FlaskSVG } from "../icons/flask";\n', "")
    text = re.sub(
        r'^ *<FlaskSVG className="h-\[1\.1em\] w-\[1\.1em\] shrink-0" />\n',
        "", text, flags=re.M,
    )
    text = text.replace("\N{TEST TUBE} " + APP_NAME, APP_NAME)
    thread.write_text(text, encoding="utf-8")
    say(TAG, "set the header and home screen logo to Deep helix")


# Components this repo owns, copied into the clone rather than patched into it.
#
# The fixes above are upstream-shaped: small, anchored, and things upstream might
# plausibly want. Product surface is the opposite — it has no upstream counterpart, will never
# converge, and is exactly what a search-and-replace inside a Python string is worst at. So it
# lives here as ordinary .tsx files: in git, reviewable in a diff, linted and edited like code,
# with the anchored patch reduced to the one line that mounts them.
#
# The directory mirrors the clone's `src/`, so a file's path here is where it lands. The copy
# is one-way and unconditional: the clone is gitignored and this is the original, so an edit
# made over there is a lost edit either way — better lost on the next launch than silently
# kept and diverging.
OVERLAY_DIR = REPO_ROOT / "chat-ui-overlay"


def ensure_overlay() -> list[str]:
    """Copy the overlay into the clone. Returns the files it actually had to write."""
    if not OVERLAY_DIR.is_dir():
        return []
    src = chat_ui_dir() / "src"
    if not src.is_dir():
        say(TAG, f"warning: no src/ in {chat_ui_dir()}; left the overlay components out.")
        return []

    written: list[str] = []
    for path in sorted(OVERLAY_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in (".ts", ".tsx"):
            continue
        target = src / path.relative_to(OVERLAY_DIR)
        content = path.read_text(encoding="utf-8")
        # Compared rather than always written, so `next dev` is not handed a changed mtime and
        # a rebuild on every launch of an unchanged app.
        if target.is_file() and target.read_text(encoding="utf-8") == content:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(str(target.relative_to(chat_ui_dir())))
    return written


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


def install_node() -> None:
    """Offer the install, or die naming one. Only ever called with node missing or too old.

    Installing is offered only through a package manager the machine already has, and only
    with a yes. That is where the line falls for a practical reason: `brew install node`
    puts node on the PATH of *future* processes, which is what `dev.py` and the user's next
    shell need, while a tarball this script downloaded and unpacked itself would be visible
    to this process alone — the UI would install and then vanish. Windows has no equivalent
    one-liner: the nodejs.org .msi wants administrator rights, which a managed machine may
    not grant, and fnm needs a shell hook after installing.
    So the per-user managers below are named rather than run, which is the difference
    between "add the UI later" and "cannot".
    """
    brew = None if WINDOWS else tool("brew")
    if brew and confirm("install Node with `brew install node` ?"):
        run(["brew", "install", "node"])
        if (node_major() or 0) >= NODE_MIN_MAJOR:
            return
        die(TAG, "node installed but not on PATH — open a new terminal and re-run this script.")
    if WINDOWS:
        say(TAG, "  with admin rights:  winget install OpenJS.NodeJS.LTS")
        say(TAG, "  without:            winget install Schniz.fnm  &&  fnm install 22")
        say(TAG, "  or unzip the Windows binary from https://nodejs.org onto your PATH")
    else:
        say(TAG, "  https://nodejs.org, or `brew install node` on a Mac")
    die(TAG, "install it and re-run this script to add the UI.")


def ensure_node() -> None:
    """Node is a real prerequisite, not a nicety: it builds the frontend and installs the
    artifact components. Offered, but only through `install_node`'s narrower path. Reached
    only after the steps above, so the message can truthfully say the headless path already
    works.

    Says nothing about pnpm. That used to live here and could not: the pin it has to match
    is inside the clone, which `ensure_chat_ui` has not made yet.
    """
    major = node_major()
    if major is None or major < NODE_MIN_MAJOR:
        say(TAG, 'everything else is ready — ask questions now with:  uv run agent "your question"')
        found = "none is installed" if major is None else f"yours is {major}"
        say(TAG, f"the chat UI needs Node {NODE_MIN_MAJOR}+ — {found}.")
        install_node()


def clone_pinned(ui_dir: Path) -> None:
    """Shallow-fetch exactly `UI_REPO_REF`.

    The long form rather than `git clone --depth 1`, which takes a branch or a tag and
    never a commit: an empty repo, then a one-commit fetch of the SHA. GitHub serves a
    bare SHA to `fetch`, and the result is the same ~560K and ~1.5s as the shallow clone
    of `main` this replaces. Left on a detached FETCH_HEAD, which is what the clone is —
    a read-only checkout that setup patches in place.
    """
    ui_dir.mkdir(parents=True, exist_ok=True)
    at = ["git", "-C", str(ui_dir)]
    run(["git", "init", "--quiet", str(ui_dir)])
    run([*at, "remote", "add", "origin", UI_REPO])
    run([*at, "fetch", "--quiet", "--depth", "1", "origin", UI_REPO_REF])
    run([*at, "checkout", "--quiet", "FETCH_HEAD"])


def warn_if_off_pin(ui_dir: Path) -> None:
    """Say so when an existing clone is not at `UI_REPO_REF`, and move on.

    A clone made before the pin sits at whatever `main` was that day, and the patches are
    only *mostly* version-independent — `unapplied_patches()` catches an anchor that has
    moved, but not one that still matches in a file whose meaning changed around it.
    Reported rather than fixed: the checkout carries setup's own edits as uncommitted
    changes, so moving it would either clobber them or refuse, and re-cloning is both the
    honest remedy and cheap.
    """
    head = subprocess.run(
        ["git", "-C", str(ui_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    # Not a git checkout at all: AGENT_CHAT_UI can point at a working copy someone
    # manages themselves, and that is their business.
    if not head or head == UI_REPO_REF:
        return
    say(TAG, f"note: {ui_dir} is at {head[:12]}, not the pinned {UI_REPO_REF[:12]}.")
    say(TAG, "  patches still apply; to move to the pin:  rm -rf .chat-ui && "
             "uv run scripts/setup.py")


def ensure_chat_ui() -> None:
    if tool("git") is None:
        die(TAG, "the chat UI needs git.")
    ui_dir = chat_ui_dir()
    if not ui_dir.is_dir():
        say(TAG, f"cloning agent-chat-ui into {ui_dir}…")
        clone_pinned(ui_dir)
    else:
        warn_if_off_pin(ui_dir)

    copied = ensure_overlay()
    if copied:
        say(TAG, f"copied {len(copied)} overlay component(s) into the chat UI")
    apply_patches()

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
    # Resolved here rather than in `ensure_node` because `pnpm_command` matches against the
    # clone's `packageManager` pin, which does not exist until the clone above.
    pnpm = pnpm_or_die(TAG)
    say(TAG, "installing frontend dependencies (~1 min)…")
    run([*pnpm.argv, "install", "--silent"], cwd=ui_dir)


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
    # `--silent` below prints nothing at all until it finishes, so without the duration
    # this is a dead terminal for minutes at the very last step of setup.
    say(TAG, "installing artifact component dependencies (a few minutes)…")
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
