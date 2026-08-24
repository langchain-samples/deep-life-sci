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

# What each patch leaves behind, and where. Every patch is idempotent because it looks for
# its own mark before doing anything, and `unapplied_patches()` asks the same question from
# outside — which is what lets `dev.py` verify the patches are *present* rather than that
# setup once *ran*. The clone is gitignored, so nothing else records what the frontend is:
# a hand-edited or half-upgraded .chat-ui otherwise sits un-patched while the repo believes
# the feature shipped. One table rather than a literal in each function, so the two answers
# cannot drift apart.
#
# Mapped to False where the patch works by *removing* something.
PATCH_MARKS = {
    "rewrite": ("next.config.mjs", "/ui/:path", True),
    "svg": ("src/components/icons/langgraph.tsx", "clip-path=", False),
    "empty-turns": ("src/components/thread/messages/ai.tsx", "hasCustomComponents", True),
    "uploads": ("src/hooks/use-file-upload.tsx", "isSpreadsheetUpload", True),
    "progress-events": ("src/providers/Stream.tsx", "isProgressEvent", True),
    "progress-row": ("src/components/thread/index.tsx", "useRunProgress", True),
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
    patch_svg_props()
    patch_empty_ai_turns()
    patch_uploads()
    patch_progress_events()
    patch_progress_row()


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


# The three patches below are cosmetic rather than load-bearing, unlike the rewrite above.
# They are applied anyway because setup's job is a working app, and an app that logs a
# console error, opens on a screenful of whitespace, or offers an upload it then refuses is
# not one. Each is anchored on an exact upstream string and prints the change instead of
# guessing if that string ever moves, so an upstream fix is never clobbered and a stale
# patch never lands silently.


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


# Every name this reads is already in scope at the anchor below; it adds only its own.
EMPTY_TURN_GUARD = """
  // setup: an AI turn that is only thinking + tool_use renders no content of its own, but
  // its hover CommandBar is opacity-0 rather than absent and still occupies its row. A run
  // here is dozens of such turns, so unpatched the first visible output sits about a
  // screenful below the question. See CLAUDE.md.
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
// rather than listing types the agent has no use for. See CLAUDE.md.
export const SUPPORTED_FILE_TYPES: string[] = [];

export const UNSUPPORTED_FILE_TITLE = "Attachments aren't supported yet";
export const UNSUPPORTED_FILE_BODY =
  "Spreadsheet and CSV upload is coming soon. Papers, figures and trial records the " +
  "agent fetches for itself — just ask for them.";"""

UPLOAD_HEADER = """\
// setup: CSV/TSV/xlsx only. Those are the attachments the agent can do something with —
// they do not stay in model context, `research_agent/middleware/uploads.py` moves them into
// the sandbox at /workspace/uploads and keeps them there across turns. An image or a PDF
// would be context and nothing else, so both stay refused. See CLAUDE.md.
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
// (see research_agent/middleware/uploads.py).
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


# The agent does its work inside one `eval` call, so the transcript shows a single tool call
# that has not returned yet — and with tool calls hidden, nothing at all. The graph narrates
# the calls happening inside it over the custom-event channel instead
# (research_agent/middleware/progress.py); these two patches carry that line to the screen.
# Split in two because they fail differently: without the first there is no event to read,
# without the second the events arrive and nothing renders them.
PROGRESS_TYPES = """\
// setup: the agent orchestrates inside one `eval`, so a multi-minute run produces no visible
// message until it is over. It narrates itself over the same custom-event channel the UI
// components already use — see research_agent/middleware/progress.py — and the latest line
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
# `eval` — about two seconds into a run that lasts minutes. Both halves of the row are
# deliberate: the dots stay for the whole run because the run is still going, and the text is
# what makes them mean something.
PROGRESS_ROW = """\
                  {isLoading && (
                    <div className="mr-auto flex items-center gap-3">
                      <AssistantMessageLoading />
                      {runProgress && (
                        <span className="animate-in fade-in-0 text-muted-foreground text-sm">
                          {runProgress}
                        </span>
                      )}
                    </div>
                  )}
"""

PROGRESS_ROW_ANCHOR = """\
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
"""

PROGRESS_ROW_EDITS = [
    ('import { useStreamContext } from "@/providers/Stream";',
     'import { useRunProgress, useStreamContext } from "@/providers/Stream";'),
    ("  const isLoading = stream.isLoading;",
     "  const isLoading = stream.isLoading;\n"
     "  // setup: what the agent is doing inside the current `eval`. See providers/Stream.tsx.\n"
     "  const runProgress = useRunProgress();"),
    (PROGRESS_ROW_ANCHOR, PROGRESS_ROW),
]


def patch_progress_row() -> None:
    """Render the latest progress line beside the typing dots."""
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
    say(TAG, "added the run status row to the chat UI")


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
