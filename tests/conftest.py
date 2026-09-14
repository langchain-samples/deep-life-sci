"""Shared fixtures. Three concerns, and they are the three ways this suite could lie.

**Environment.** Nearly every module here reads its configuration from `os.environ` at
*call* time rather than at import — a deliberate choice the repo documents in `paths.py`
and `cache_io.py`, because `cli.py` and `evals/run.py` both `load_dotenv(override=True)`
after the package is importable. That makes the environment shared mutable state between
tests, and a developer's own `.env` is loaded into the process by anything that imports
`evals.sync`. `isolated_env` is therefore autouse: every test starts from an environment
with this project's variables stripped.

**The cache.** `sources/pubmed.py` and `sources/ctgov.py` bind their cache directory from
`paths` at import, so a test that let them write would put files in the developer's real
`data/`. The cache fixtures rebind the module attribute to a tmp_path. Rebinding the
module rather than the env var is the honest thing to do: the env var is only read at
import, so setting it inside a test would do nothing and the test would pass while writing
to the wrong place.

**HTTP.** No test in this suite makes a network call. `mock_ncbi`/`mock_ctgov` put an
`httpx.MockTransport` behind `httpx.AsyncClient`, which is the narrowest seam that still
exercises the real `_request` — its retry ladder, its POST threshold and its status
handling are the parts worth testing, and they live inside that function. Those two
fixtures also stub each module's `backoff_delay` and `Throttle.wait` to zero: the delays
themselves belong to `_http` and are tested there directly, so leaving them live would only
add ~30s of real sleeping to a suite that asserts nothing about it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

# Tracing, forced off for the whole session rather than per test.
#
# This has to happen at import — before any test module is collected — because
# `evals/sync.py` calls `load_dotenv(override=True)` at *its* import, which pulls the
# developer's real `.env` into `os.environ` during collection. With tracing left on,
# every `@tool` invocation in this suite opens a run against LangSmith and ships it: a
# unit suite making network calls, on someone's real workspace, with test fixtures as the
# payload. Verified by watching 401s scroll past on a first run of this file.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_OTEL_ENABLED"] = "false"


def pytest_configure(config) -> None:
    """Re-assert the above after plugins and `.env` loads have had their turn."""
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_OTEL_ENABLED"] = "false"


# Everything this project reads out of the environment. Stripped before each test so a
# developer's `.env` cannot decide whether an assertion holds.
_PROJECT_ENV = (
    "DEEP_LIFE_SCI_DATA_DIR",
    "DEEP_LIFE_SCI_CACHE_TTL",
    "LANGSMITH_API_KEY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_GATEWAY_BASE_URL",
    "LANGSMITH_GATEWAY_ANTHROPIC_URL",
    "NCBI_API_KEY",
    "NCBI_EMAIL",
    "NCBI_TOOL",
    "EVALS_DATASET_PREFIX",
    "SANDBOX_SNAPSHOT_NAME",
    "DEEP_LIFE_SCI_PERF",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every variable this project reads, plus the four role x axis model vars."""
    from deep_life_sci.models import ENV_VARS

    for name in (*_PROJECT_ENV, *ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    # Belt and braces: a test that imports `evals.sync` re-runs `load_dotenv(override=True)`
    # and can put tracing back on mid-session.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def abstract_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `pubmed.ABSTRACT_CACHE` at a tmp_path, and hand it back for seeding."""
    from deep_life_sci.sources import pubmed

    cache = tmp_path / "abstracts"
    cache.mkdir()
    monkeypatch.setattr(pubmed, "ABSTRACT_CACHE", cache)
    return cache


@pytest.fixture
def cache_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ...]:
    """Rebind `cache_io`'s sweep scope to tmp_path directories.

    `sweep()` reads `CACHE_ROOTS` from the `cache_io` module namespace, and the real value
    is the developer's `data/`. Deleting from there because a test ran would be
    unforgivable, so this is rebound rather than mocked at the filesystem layer.
    """
    from deep_life_sci.sources import cache_io

    roots = tuple((tmp_path / name) for name in ("abstracts", "pmc", "trials"))
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(cache_io, "CACHE_ROOTS", roots)
    monkeypatch.setattr(cache_io, "_last_sweep", None)
    return roots


class RecordingTransport(httpx.MockTransport):
    """A `MockTransport` that keeps every request it served.

    The recorded requests are half the point: `_request`'s POST branch and its retry
    ladder are only observable as *how many* requests went out and by which method.
    """

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []

        def recording(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return handler(request)

        super().__init__(recording)


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> RecordingTransport:
    """Make every `httpx.AsyncClient` built from here on serve `handler`.

    Patched on `httpx` itself rather than on the source module, because both clients
    construct their client inline (`httpx.AsyncClient(timeout=120.0)`) and there is no
    injection point short of the attribute. monkeypatch restores it at teardown, and no
    test in this suite wants a real client, so the breadth costs nothing.
    """
    transport = RecordingTransport(handler)
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return transport


@pytest.fixture
def mock_ncbi(monkeypatch: pytest.MonkeyPatch):
    """Serve E-utilities responses from a handler, and neutralise the throttle.

    The throttle is patched out rather than left in place because its interval is 0.34s
    without an API key, which would add a third of a second to every request a test makes
    for no assertion's benefit. `test_http.py` tests the throttle itself.
    """
    from deep_life_sci.sources import pubmed

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(pubmed._throttle, "wait", no_wait)
    monkeypatch.setattr(pubmed, "backoff_delay", lambda *a, **k: 0.0)

    def install(handler):
        return _install_transport(monkeypatch, handler)

    return install


@pytest.fixture
def mock_ctgov(monkeypatch: pytest.MonkeyPatch):
    """The `mock_ncbi` equivalent for the ClinicalTrials.gov client."""
    from deep_life_sci.sources import ctgov

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(ctgov._throttle, "wait", no_wait)
    monkeypatch.setattr(ctgov, "backoff_delay", lambda *a, **k: 0.0)

    def install(handler):
        return _install_transport(monkeypatch, handler)

    return install


def json_response(payload: Any, status: int = 200, **headers: str) -> httpx.Response:
    """A JSON `httpx.Response`, for a mock handler to return."""
    return httpx.Response(status, json=payload, headers=headers)


def text_response(body: str, status: int = 200, **headers: str) -> httpx.Response:
    """A text/XML `httpx.Response`, for a mock handler to return."""
    return httpx.Response(status, text=body, headers=headers)


def write_cached_abstract(cache: Path, pmid: str, **fields: Any) -> Path:
    """Seed one abstract-cache entry. Defaults to a complete, current-schema record."""
    record = {
        "pmid": pmid,
        "title": f"Paper {pmid}",
        "abstract": "An abstract.",
        "sections": [],
        "journal": "Journal",
        "year": "2025",
        "retracted": False,
        "publication_types": ["Journal Article"],
        "doi": None,
        "pmcid": None,
    }
    record.update(fields)
    path = cache / f"{pmid}.json"
    path.write_text(json.dumps(record))
    return path
