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
#   ./dev.sh                            # both, logs interleaved and prefixed
#   AGENT_CHAT_UI=~/src/acu ./dev.sh    # UI checkout somewhere other than ../agent-chat-ui
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="${AGENT_CHAT_UI:-$(dirname "$ROOT")/agent-chat-ui}"

PIDS=()

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

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

if [[ ! -d "$UI_DIR" ]]; then
  echo "[dev] no chat UI at $UI_DIR" >&2
  echo "[dev] clone it, then apply the /ui/* rewrite to next.config.mjs:" >&2
  echo "[dev]   git clone https://github.com/langchain-ai/agent-chat-ui.git $UI_DIR" >&2
  exit 1
fi

if listening 2024; then
  echo "[dev] :2024 already serving — reusing it"
else
  start agent "$ROOT" uv run --group dev langgraph dev --no-browser
fi

if listening 3000; then
  echo "[dev] :3000 already serving — reusing it"
else
  start ui "$UI_DIR" pnpm dev
fi

# Both already up. There is nothing to supervise and nothing this script may stop — the
# servers belong to whoever started them — so say so and get out of the way, rather than
# blocking on a `wait` that would never return.
if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "[dev] both halves already running — nothing to start."
  echo "[dev] chat UI -> http://localhost:3000"
  exit 0
fi

echo "[dev] chat UI -> http://localhost:3000   (Ctrl-C stops what this script started)"
wait
