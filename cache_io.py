"""Host-side cache reads and writes, kept off the event loop.

The PubMed and PMC caches were written for `agent.py`, which runs one question and
exits. Blocking the loop there costs nothing, because nothing else is on it.

Under `langgraph dev` the same coroutines run inside an ASGI server, and that changes
the rules twice over. In development, blockbuster raises `BlockingError` on any
blocking syscall made from a coroutine, so a cached `read_text()` doesn't slow the run
down — it kills it. In production nothing raises, which is worse: a synchronous read
stalls every other run in the process, and the health check with them.

So every blocking call goes through `asyncio.to_thread`. The functions here are
deliberately forgiving — a cache is an optimisation, so an unreadable or corrupt entry
reports a miss and the caller refetches, rather than taking down a run over a bad file.

Note the distinction between `MISSING` and `None`: a resolved-to-nothing PMCID is
cached as literal `null`, and that is a real answer meaning "this article has no
objects". Collapsing the two would re-resolve every absent paper on every turn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

# Sentinel for "no usable cache entry", distinct from a cached JSON `null`.
MISSING: Any = object()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return MISSING


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def aread_json(path: Path) -> Any:
    """Cached JSON, or `MISSING` if absent/corrupt. A cached `null` returns None."""
    return await asyncio.to_thread(_read_json, path)


async def awrite_json(path: Path, obj: Any) -> None:
    """Write JSON, creating parent directories."""
    await asyncio.to_thread(_write_json, path, obj)


async def aread_bytes(path: Path) -> bytes | None:
    """Cached bytes, or None if absent/unreadable."""
    return await asyncio.to_thread(_read_bytes, path)


async def awrite_bytes(path: Path, data: bytes) -> None:
    """Write bytes, creating parent directories."""
    await asyncio.to_thread(_write_bytes, path, data)
