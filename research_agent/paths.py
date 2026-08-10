"""Filesystem locations, resolved in one place.

Two different roots live here and they must not be confused:

* **Host paths** (`DATA_DIR` and below) are the durable cache `sources/pubmed.py` and
  `sources/pmc.py` own. The agent never sees them — the sandbox starts empty and the
  agent writes what it needs into it.
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
"""

from __future__ import annotations

import os
from pathlib import Path

# research_agent/paths.py -> research_agent/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("RESEARCH_AGENT_DATA_DIR") or REPO_ROOT / "data")

ABSTRACT_CACHE = DATA_DIR / "abstracts"
SEARCH_DUMPS = DATA_DIR / "searches"
PMC_CACHE = DATA_DIR / "pmc"

# Where the agent works inside the sandbox. Mirrored in the system prompt.
WORKSPACE = "/workspace"

# User deliverables. `middleware/artifacts.py` sweeps this after every writing tool call
# and publishes what it finds, so anything landing here reaches the user.
OUT_DIR = f"{WORKSPACE}/out"

__all__ = [
    "ABSTRACT_CACHE",
    "DATA_DIR",
    "OUT_DIR",
    "PMC_CACHE",
    "REPO_ROOT",
    "SEARCH_DUMPS",
    "WORKSPACE",
]
