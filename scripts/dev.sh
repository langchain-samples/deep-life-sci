#!/usr/bin/env bash
#
# Start the local stack: the agent server and the chat UI that renders its artifacts.
#
# Both halves are required. :2024 serves graph.py; :3000 serves the frontend, and is the
# side carrying the `/ui/*` rewrite that lets artifact components load at all (see the
# same-origin invariant in CLAUDE.md). Running only one gets you an API with no window,
# or a window pointed at nothing.
#
# A server already listening on its port is left alone and reused, so this is safe to run
# alongside a `langgraph dev` you started yourself. Only what this script starts is what
# it stops.
#
#   ./scripts/dev.sh                            # both, logs interleaved and prefixed
#   AGENT_CHAT_UI=~/src/acu ./scripts/dev.sh    # a chat UI checkout of your own
#   NO_BROWSER=1 ./scripts/dev.sh               # don't open a browser tab
#
set -euo pipefail

# ../ because this script lives in scripts/ but both servers must start from the repo
# root — `langgraph dev` resolves langgraph.json relative to its working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# .chat-ui is where `./setup_sci_agent` puts the frontend: inside the repo,
# gitignored and dockerignored. AGENT_CHAT_UI points at a checkout of your own instead.
UI_DIR="${AGENT_CHAT_UI:-$ROOT/.chat-ui}"

PIDS=()

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

UI_URL="http://localhost:3000"

# The window is the assumed way in, so opening it is part of starting the stack rather
# than something to remember afterwards. NO_BROWSER=1 opts out — a headless box, or a tab
# you already have open. An unknown platform simply prints the URL, as before.
url_opener() {  # echoes the command that opens a URL; empty if there is none, or if opted out
  if [[ -n ${NO_BROWSER:-} ]]; then return 0; fi
  local cmd
  for cmd in open xdg-open wslview; do
    if command -v "$cmd" >/dev/null 2>&1; then printf '%s' "$cmd"; return 0; fi
  done
  return 0
}
OPENER="$(url_opener)"

# Each service runs in its own subshell so $! is a handle to the whole pipeline; the
# cleanup below kills that subshell and the children it spawned (pnpm -> next dev).
start() {
  local name=$1 dir=$2
  shift 2
  ( cd "$dir" && "$@" 2>&1 | sed -l "s/^/[$name] /" ) &
  PIDS+=("$!")
}

cleanup() {
  trap - INT TERM EXIT
  # Nothing started means nothing to stop. Returning quietly matters: announcing a
  # shutdown here would make the "everything was already up" exit below read as though
  # this script had just torn down servers it never owned.
  #
  # The length test also guards macOS's bash 3.2, where expanding an empty array under
  # `set -u` is an error rather than nothing.
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    return
  fi
  echo
  echo "[dev] shutting down"
  for pid in "${PIDS[@]}"; do
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
# Ctrl-C is how you're meant to stop this, so it exits 0. A bare EXIT means a server
# died on its own and the status is worth propagating.
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

# Polled rather than opened straight away: `next dev` needs a few seconds to bind :3000,
# and a browser that arrives first shows a connection error the user has to reload past.
# Backgrounded, and tracked in PIDS so a Ctrl-C during those seconds takes it down too —
# otherwise the tab opens onto servers this script has already stopped.
open_ui_when_ready() {
  if [[ -z $OPENER ]]; then return 0; fi
  (
    for _ in $(seq 1 120); do
      if listening 3000; then "$OPENER" "$UI_URL" >/dev/null 2>&1 || true; exit 0; fi
      sleep 0.5
    done
  ) &
  PIDS+=("$!")
}

if [[ ! -d "$UI_DIR" ]]; then
  echo "[dev] no chat UI at $UI_DIR" >&2
  echo "[dev] install it (clone, /ui/* rewrite, pnpm install) with:" >&2
  echo "[dev]   ./setup_sci_agent" >&2
  exit 1
fi

if listening 2024; then
  echo "[dev] :2024 already serving — reusing it"
else
  # --n-jobs-per-worker: `langgraph dev` defaults this to 1, so a single run occupies the
  # only worker and every later run queues behind it. That bites hardest across restarts:
  # the server persists its queue to .langgraph_api/, so a run abandoned by a Ctrl-C is
  # resumed on the next boot, takes the worker, and starves the query you just typed —
  # which sits `pending` forever, produces no LangSmith trace, and looks like a hang while
  # the console streams the *old* run's progress. Raising it lets a stale run finish
  # alongside a new one instead of blocking it.
  #
  # This makes the asyncio.to_thread rule in sources/cache_io.py load-bearing rather
  # than theoretical: runs share one event loop, so a blocking call stalls its neighbours.
  start agent "$ROOT" uv run --group dev langgraph dev --no-browser --n-jobs-per-worker 5
fi

if listening 3000; then
  echo "[dev] :3000 already serving — reusing it"
  # Already listening, so there is nothing to wait for: open it here rather than through
  # the backgrounded waiter, which the both-already-up exit below would race.
  if [[ -n $OPENER ]]; then "$OPENER" "$UI_URL" >/dev/null 2>&1 || true; fi
else
  start ui "$UI_DIR" pnpm dev
  open_ui_when_ready
fi

# Both already up. There is nothing to supervise and nothing this script may stop — the
# servers belong to whoever started them — so say so and get out of the way, rather than
# blocking on a `wait` that would never return.
if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "[dev] both halves already running — nothing to start."
  echo "[dev] chat UI -> $UI_URL"
  exit 0
fi

echo "[dev] chat UI -> $UI_URL   (Ctrl-C stops what this script started)"
wait
