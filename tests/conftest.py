"""Offline test fixtures.

Disable dotenv before test collection, strip project settings before each test, and
redirect every source cache to temporary directories. Tests that need seeded cache
entries can override those module attributes with the narrower fixtures below.

Socket connections are blocked per test, including attempts that application code
catches. HTTP tests inject MockTransport so URL construction, status handling, retries,
parsing, and caching still execute. Source pacing is disabled there; test_http.py
checks the throttle itself with a controlled clock.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

# Apply these before entry points can be imported by test collection. Per-test
# fixtures repeat them because tests can temporarily change process configuration.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
# Prevent entry-point imports from loading a developer's configuration during collection.
os.environ["PYTHON_DOTENV_DISABLED"] = "true"


def pytest_configure(config) -> None:
    """Keep tracing disabled before collection starts."""
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
    "PERF_PROBE",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every variable this project reads, plus the four role x axis model vars."""
    from deep_life_sci.models import ENV_VARS

    for name in (*_PROJECT_ENV, *ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_OTEL_ENABLED", "false")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail even if production catches the socket exception and returns an error value."""
    attempts = []
    original = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(True)
            raise AssertionError("Unit tests must use a mock transport")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    yield
    assert not attempts, "A test attempted a real network connection"


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    """No test may read, refresh, or sweep the developer's corpus."""
    from deep_life_sci.sources import cache_io, ctgov, pmc, pubmed
    from evals.evaluators import citations

    roots = tuple(tmp_path / name for name in ("abstracts", "pmc", "trials"))
    for module, name, root in (
        (pubmed, "ABSTRACT_CACHE", roots[0]),
        (citations, "ABSTRACT_CACHE", roots[0]),
        (pmc, "PMC_CACHE", roots[1]),
        (pmc, "RESOLVED_CACHE", roots[1] / "_resolved"),
        (ctgov, "CTGOV_CACHE", roots[2]),
    ):
        monkeypatch.setattr(module, name, root)
    monkeypatch.setattr(cache_io, "CACHE_ROOTS", roots)
    monkeypatch.setattr(cache_io, "_last_sweep", None)
    monkeypatch.setattr(pmc, "_sem", None)


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
