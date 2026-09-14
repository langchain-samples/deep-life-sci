"""`sources/cache_io.py`: the idle TTL, and the sweep's scope.

Two behaviours here are load-bearing and easy to break silently:

* **`MISSING` is not `None`.** A resolved-to-nothing PMCID is cached as literal `null`,
  and that is a real answer meaning "this article has no objects". Collapsing the two
  re-resolves every absent paper on every turn.
* **`sweep()` is scoped to `CACHE_ROOTS`, never to `DATA_DIR`.** That directory is
  operator-configurable via `DEEP_LIFE_SCI_DATA_DIR`, so a sweep that recursed into
  whatever else lives there would delete files this module does not own.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pytest

from deep_life_sci.paths import IDLE_TTL_SECONDS
from deep_life_sci.sources import cache_io
from deep_life_sci.sources.cache_io import (
    MISSING,
    aread_bytes,
    aread_json,
    awrite_bytes,
    awrite_json,
    is_fresh,
    sweep,
    sweep_if_due,
    touch,
    ttl_seconds,
)


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


class TestTtlSeconds:
    def test_defaults_to_the_window_that_also_reaps_the_sandbox(self):
        """Tied together deliberately: a thread's container and its corpus go stale together."""
        assert ttl_seconds() == float(IDLE_TTL_SECONDS)

    @pytest.mark.parametrize("value", ["off", "never", "none", "0", "-1", "", "OFF", " off "])
    def test_every_spelling_of_disabled_turns_expiry_off(self, value: str, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", value)
        assert ttl_seconds() is None

    def test_a_number_is_taken_as_seconds(self, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "45")
        assert ttl_seconds() == 45.0

    def test_a_negative_number_disables_expiry_rather_than_expiring_everything(
        self, monkeypatch
    ):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "-30")
        assert ttl_seconds() is None

    def test_an_unparseable_value_warns_loudly_and_falls_back(self, monkeypatch, caplog):
        """Otherwise the cache runs at a TTL nobody asked for and nothing says which."""
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "ten minutes")
        with caplog.at_level(logging.WARNING, logger=cache_io.__name__):
            assert ttl_seconds() == float(IDLE_TTL_SECONDS)
        assert "ten minutes" in caplog.text

    def test_is_read_per_call_rather_than_captured_at_import(self, monkeypatch):
        """`cli.py` and `evals/run.py` load `.env` after this module is importable."""
        assert ttl_seconds() == float(IDLE_TTL_SECONDS)
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "5")
        assert ttl_seconds() == 5.0


class TestIsFresh:
    def test_a_recent_file_is_fresh(self, tmp_path):
        path = tmp_path / "f"
        path.write_text("x")
        assert is_fresh(path, 600) is True

    def test_a_file_past_the_ttl_is_not(self, tmp_path):
        path = tmp_path / "f"
        path.write_text("x")
        _age(path, 700)
        assert is_fresh(path, 600) is False

    def test_a_missing_file_is_a_miss_rather_than_an_error(self, tmp_path):
        assert is_fresh(tmp_path / "absent", 600) is False

    def test_with_expiry_off_anything_is_fresh(self, tmp_path):
        path = tmp_path / "f"
        path.write_text("x")
        _age(path, 10_000_000)
        assert is_fresh(path, None) is True



class TestTouch:
    def test_refreshes_the_mtime(self, tmp_path):
        path = tmp_path / "f"
        path.write_text("x")
        _age(path, 500)
        before = path.stat().st_mtime
        touch(path)
        assert path.stat().st_mtime > before

    def test_a_file_that_cannot_be_touched_does_not_raise(self, tmp_path):
        """Losing the refresh costs a refetch; raising would cost the whole run."""
        touch(tmp_path / "absent")


class TestJsonRoundTrip:
    async def test_writes_and_reads_back(self, tmp_path):
        path = tmp_path / "nested" / "f.json"
        await awrite_json(path, {"a": 1})
        assert await aread_json(path) == {"a": 1}

    async def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.json"
        await awrite_json(path, [])
        assert path.is_file()

    async def test_an_absent_entry_is_missing(self, tmp_path):
        assert await aread_json(tmp_path / "absent.json") is MISSING

    async def test_a_cached_null_is_none_and_not_missing(self, tmp_path):
        """A resolved-to-nothing PMCID is a real answer: this article has no objects."""
        path = tmp_path / "f.json"
        await awrite_json(path, None)
        result = await aread_json(path)
        assert result is None
        assert result is not MISSING

    async def test_a_corrupt_entry_reports_a_miss_rather_than_raising(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("{not json")
        assert await aread_json(path) is MISSING

    async def test_a_stale_entry_reports_a_miss(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        path = tmp_path / "f.json"
        await awrite_json(path, {"a": 1})
        _age(path, 120)
        assert await aread_json(path) is MISSING

    async def test_a_successful_read_refreshes_the_entry(self, tmp_path):
        path = tmp_path / "f.json"
        await awrite_json(path, {"a": 1})
        _age(path, 60)
        before = path.stat().st_mtime
        await aread_json(path)
        assert path.stat().st_mtime > before

    async def test_a_corrupt_entry_keeps_its_mtime_so_the_sweep_collects_it(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("{not json")
        _age(path, 60)
        before = path.stat().st_mtime
        await aread_json(path)
        assert path.stat().st_mtime == before

    async def test_freshness_is_checked_before_the_read(self, tmp_path, monkeypatch):
        """A PMC figure runs to megabytes; one stat beats reading bytes we will discard."""
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        path = tmp_path / "f.json"
        path.write_text("{not json")
        _age(path, 120)
        reads = []
        real_read = Path.read_text
        monkeypatch.setattr(
            Path, "read_text", lambda self, *a, **k: (reads.append(self), real_read(self))[1]
        )
        assert await aread_json(path) is MISSING
        assert reads == []


class TestBytesRoundTrip:
    async def test_writes_and_reads_back(self, tmp_path):
        path = tmp_path / "img.png"
        await awrite_bytes(path, b"\x89PNG")
        assert await aread_bytes(path) == b"\x89PNG"

    async def test_an_absent_entry_is_none(self, tmp_path):
        assert await aread_bytes(tmp_path / "absent.png") is None

    async def test_a_stale_entry_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        path = tmp_path / "img.png"
        await awrite_bytes(path, b"x")
        _age(path, 120)
        assert await aread_bytes(path) is None


class TestBlockingCallsAreOffTheEventLoop:
    """Under `langgraph dev`, blockbuster turns a blocking read in a coroutine into a
    `BlockingError` that kills the run; in production it stalls every other run.
    """

    async def test_reads_and_writes_are_handed_to_a_thread(self, tmp_path, monkeypatch):
        import threading

        loop_thread = threading.current_thread()
        seen: list[threading.Thread] = []
        real_read = Path.read_text

        def recording(self, *args, **kwargs):
            seen.append(threading.current_thread())
            return real_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", recording)
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"a": 1}))
        await aread_json(path)

        assert seen and all(t is not loop_thread for t in seen)


class TestSweep:
    def test_removes_entries_past_the_ttl(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        stale = cache_roots[0] / "old.json"
        stale.write_text("{}")
        _age(stale, 120)
        assert sweep() == 1
        assert not stale.exists()

    def test_leaves_fresh_entries_alone(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "600")
        fresh = cache_roots[0] / "new.json"
        fresh.write_text("{}")
        assert sweep() == 0
        assert fresh.exists()

    def test_does_nothing_when_expiry_is_off(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "off")
        stale = cache_roots[0] / "old.json"
        stale.write_text("{}")
        _age(stale, 10_000_000)
        assert sweep() == 0
        assert stale.exists()

    def test_removes_the_empty_package_directories_pmc_leaves_behind(
        self, cache_roots, monkeypatch
    ):
        """The PMC cache mirrors the S3 layout; 125 empty husks make it illegible."""
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        package = cache_roots[1] / "PMC5904197.1"
        package.mkdir()
        entry = package / "fig1.jpg"
        entry.write_bytes(b"x")
        _age(entry, 120)
        sweep()
        assert not package.exists()

    def test_a_directory_still_holding_a_fresh_entry_survives(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "600")
        package = cache_roots[1] / "PMC5904197.1"
        package.mkdir()
        (package / "fig1.jpg").write_bytes(b"x")
        sweep()
        assert package.exists()

    def test_never_touches_anything_outside_the_named_roots(self, cache_roots, monkeypatch):
        """DEEP_LIFE_SCI_DATA_DIR can point anywhere; a sibling is not ours to delete."""
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        outsider = cache_roots[0].parent / "someones_notes.txt"
        outsider.write_text("important")
        _age(outsider, 10_000_000)
        sweep()
        assert outsider.exists()

    def test_a_root_that_does_not_exist_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_io, "CACHE_ROOTS", (tmp_path / "never_created",))
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        assert sweep() == 0


class TestSweepIfDue:
    async def test_sweeps_the_first_time_it_is_called(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "60")
        stale = cache_roots[0] / "old.json"
        stale.write_text("{}")
        _age(stale, 120)
        assert await sweep_if_due() == 1

    async def test_self_gates_to_once_per_ttl_window(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "600")
        await sweep_if_due()
        second = cache_roots[0] / "old.json"
        second.write_text("{}")
        _age(second, 10_000)
        assert await sweep_if_due() == 0
        assert second.exists(), "the second call must not have swept"

    async def test_does_nothing_when_expiry_is_off(self, cache_roots, monkeypatch):
        """`evals/run.py` sets this, so refetching does not depend on wall clock."""
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "off")
        assert await sweep_if_due() == 0

    async def test_concurrent_callers_sweep_once_between_them(self, cache_roots, monkeypatch):
        monkeypatch.setenv("DEEP_LIFE_SCI_CACHE_TTL", "600")
        calls = []
        real = cache_io.sweep
        monkeypatch.setattr(
            cache_io, "sweep", lambda ttl=None: (calls.append(ttl), real(ttl))[1]
        )
        await asyncio.gather(*(sweep_if_due() for _ in range(4)))
        assert len(calls) == 1
