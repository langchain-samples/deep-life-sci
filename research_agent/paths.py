"""Filesystem locations and the lifetime that governs them, resolved in one place.

Two different roots live here and they must not be confused:

* **Host paths** (`DATA_DIR` and below) are the durable cache the modules under
  `sources/` own. The agent never sees them — the sandbox starts empty and the agent
  writes what it needs into it.
* **Sandbox paths** (`WORKSPACE`, `OUT_DIR`) are strings, not `Path`s, because they name
  locations inside a Linux container that this process cannot stat. They are mirrored in
  the system prompt and baked into the snapshot by `scripts/build_snapshot.py`.

`DATA_DIR` is anchored to the repository root rather than to this file's directory.
That distinction used to be invisible, because every module sat at the root and
`Path(__file__).parent` *was* the repo root. Under a package it is not, and getting it
wrong doesn't fail — it silently starts a second, empty cache and re-fetches a corpus
that was already on disk. Evals make that worse: they run from whatever cwd the harness
picked, and a cache miss there costs real NCBI rate limit.

Set `RESEARCH_AGENT_DATA_DIR` to point the cache somewhere else — a scratch disk, or a
per-eval-run directory when you deliberately want cold-cache timings.

`IDLE_TTL_SECONDS` is here rather than in `sandbox.py` because two layers now share it
and neither may import the other: the sandbox's server-side reaper and the host cache's
expiry (`sources/cache_io.py`). `sources/` must not depend on the sandbox layer — it is
host-side data with no container in it — and `sandbox.py` reads env at import time, which
`cli.py` orders around deliberately. A duplicated literal in both places would drift, and
the whole point is that a thread's container and a thread's corpus go stale together.
"""

from __future__ import annotations

import os
from pathlib import Path

# research_agent/paths.py -> research_agent/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("RESEARCH_AGENT_DATA_DIR") or REPO_ROOT / "data")

ABSTRACT_CACHE = DATA_DIR / "abstracts"
PMC_CACHE = DATA_DIR / "pmc"
CTGOV_CACHE = DATA_DIR / "trials"

# Everything `cache_io.sweep` is allowed to delete from. Named explicitly rather than
# walking DATA_DIR, because RESEARCH_AGENT_DATA_DIR can point anywhere and a sweep that
# recurses into whatever else lives there is a footgun, not a cleanup.
CACHE_ROOTS = (ABSTRACT_CACHE, PMC_CACHE, CTGOV_CACHE)

# How long an idle thing stays alive — both a sandbox container and a host cache entry.
#
# For the sandbox: it is deleted on context exit, but a `finally` doesn't survive SIGKILL
# and billing doesn't care why the process died. This is the server-side backstop.
#
# For the cache: this is an *idle* TTL, refreshed on every hit (`cache_io`), so a thread
# that keeps working keeps its corpus and a thread resumed tomorrow refetches. Tying the
# two together is deliberate — past this window a returning thread finds neither its
# container nor its cache, instead of a warm cache pointing into a container that is gone.
IDLE_TTL_SECONDS = 600

# Where the agent works inside the sandbox. Mirrored in the system prompt.
WORKSPACE = "/workspace"

# User deliverables. `middleware/artifacts.py` sweeps this after every writing tool call
# and publishes what it finds, so anything landing here reaches the user.
OUT_DIR = f"{WORKSPACE}/out"

# Where files the user attached are materialised, by `middleware/uploads.py`. Deliberately
# not under `OUT_DIR`: that directory is swept and published, so a file the user gave us
# would be handed straight back to them as a deliverable of their own question.
#
# Nothing here is durable. The durable copy of an upload lives in the LangGraph store and
# this directory is rebuilt from it on every turn, because a container reaped by
# IDLE_TTL_SECONDS is replaced by an empty one. See `middleware/uploads.py`.
UPLOAD_DIR = f"{WORKSPACE}/uploads"

__all__ = [
    "ABSTRACT_CACHE",
    "CACHE_ROOTS",
    "CTGOV_CACHE",
    "DATA_DIR",
    "IDLE_TTL_SECONDS",
    "OUT_DIR",
    "PMC_CACHE",
    "REPO_ROOT",
    "UPLOAD_DIR",
    "WORKSPACE",
]
