"""Retry the real wrapper against a fake SDK; never create a container."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from deepagents.backends import LangSmithSandbox
from langsmith.sandbox import SandboxConnectionError

from deep_life_sci import sandbox as sb


@pytest.mark.parametrize("attempts", [0, 1, 4])
async def test_retry_exhaustion_is_bounded_and_preserves_exception(monkeypatch, attempts):
    error = SandboxConnectionError("unreachable")
    execute = AsyncMock(side_effect=error)
    sleep = AsyncMock()
    monkeypatch.setattr(LangSmithSandbox, "aexecute", execute)
    monkeypatch.setattr(sb.asyncio, "sleep", sleep)
    backend = sb.ResilientSandbox(SimpleNamespace(), attempts=attempts)
    with pytest.raises(SandboxConnectionError) as caught:
        await backend.aexecute("echo ok", timeout=12)
    assert caught.value is error
    assert execute.await_count == max(1, attempts)
    assert sleep.await_count == max(1, attempts) - 1
    execute.assert_awaited_with("echo ok", timeout=12)


async def test_retry_delays_double_to_the_cap(monkeypatch):
    execute = AsyncMock(side_effect=[SandboxConnectionError("retry")] * 4 + ["done"])
    sleep = AsyncMock()
    monkeypatch.setattr(LangSmithSandbox, "aexecute", execute)
    monkeypatch.setattr(sb.asyncio, "sleep", sleep)
    backend = sb.ResilientSandbox(SimpleNamespace(), attempts=5, max_delay=1)
    assert await backend.aexecute("echo ok") == "done"
    assert sleep.await_args_list == [call(0.5), call(1), call(1), call(1)]


@pytest.mark.parametrize("error", [ValueError("bug"), asyncio.CancelledError()])
async def test_non_connection_errors_and_cancellation_are_never_retried(monkeypatch, error):
    execute = AsyncMock(side_effect=error)
    monkeypatch.setattr(LangSmithSandbox, "aexecute", execute)
    backend = sb.ResilientSandbox(SimpleNamespace())
    with pytest.raises(type(error)):
        await backend.aexecute("echo ok")
    assert execute.await_count == 1


async def test_second_failure_reacquires_and_closes_stale_client(monkeypatch):
    fresh = SimpleNamespace(name="fresh")
    acquire = Mock(return_value=fresh)
    execute = AsyncMock(side_effect=[SandboxConnectionError("gone")] * 2 + ["done"])
    monkeypatch.setattr(LangSmithSandbox, "aexecute", execute)
    backend = sb.ResilientSandbox(SimpleNamespace(), reacquire=acquire, base_delay=0)
    stale = SimpleNamespace(aclose=AsyncMock())
    backend._async_client = stale
    backend._async_sandbox = object()
    assert await backend.aexecute("echo ok") == "done"
    acquire.assert_called_once_with()
    stale.aclose.assert_awaited_once_with()
    assert backend._sandbox is fresh
    assert backend._async_client is None
    assert backend._async_sandbox is None


async def test_concurrent_rebinds_acquire_only_once():
    acquire = Mock(return_value=SimpleNamespace(name="fresh"))
    backend = sb.ResilientSandbox(SimpleNamespace(), reacquire=acquire)
    assert all(await asyncio.gather(*(backend._arebind(0) for _ in range(20))))
    acquire.assert_called_once_with()
    assert backend._rebind_generation == 1


async def test_failed_reacquisition_keeps_old_handle():
    old = SimpleNamespace()
    backend = sb.ResilientSandbox(old, reacquire=Mock(side_effect=OSError("offline")))
    assert await backend._arebind(0) is False
    assert backend._sandbox is old
    assert backend._rebind_generation == 0


@pytest.mark.parametrize(
    "command", ["pip install x", "python3 -m pip install x", "apt-get install x"]
)
async def test_install_guard_covers_both_public_execute_methods(monkeypatch, command):
    sync, asynchronous = Mock(), AsyncMock()
    monkeypatch.setattr(LangSmithSandbox, "execute", sync)
    monkeypatch.setattr(LangSmithSandbox, "aexecute", asynchronous)
    backend = sb.ResilientSandbox(SimpleNamespace())
    assert backend.execute(command).exit_code == 1
    assert (await backend.aexecute(command)).exit_code == 1
    sync.assert_not_called()
    asynchronous.assert_not_awaited()


def test_sync_retry_preserves_timeout_and_backoff(monkeypatch):
    execute = Mock(side_effect=[SandboxConnectionError("blip"), "done"])
    sleep = Mock()
    monkeypatch.setattr(LangSmithSandbox, "execute", execute)
    monkeypatch.setattr(sb.time, "sleep", sleep)
    assert sb.ResilientSandbox(SimpleNamespace()).execute("echo ok", timeout=3) == "done"
    assert execute.call_args_list == [call("echo ok", timeout=3)] * 2
    sleep.assert_called_once_with(0.5)


def test_snapshot_lookup_requires_an_exact_name():
    client = Mock()
    client.list_snapshots.return_value = [SimpleNamespace(name=sb.SNAPSHOT_NAME + "-old")]
    assert sb.find_snapshot(client) is None
    client.list_snapshots.return_value.append(SimpleNamespace(name=sb.SNAPSHOT_NAME))
    assert sb.find_snapshot(client) == sb.SNAPSHOT_NAME
    client.list_snapshots.side_effect = OSError("offline")
    assert sb.find_snapshot(client) is None


def test_failed_boot_deletes_the_named_container():
    client = Mock()
    client.wait_for_sandbox.side_effect = TimeoutError("boot timeout")
    with pytest.raises(TimeoutError, match="boot timeout"):
        sb.boot(client, "snapshot", "owned")
    client.create_sandbox.assert_called_once_with(
        snapshot_name="snapshot",
        name="owned",
        wait_for_ready=False,
        idle_ttl_seconds=sb.IDLE_TTL_SECONDS,
        delete_after_stop_seconds=sb.DELETE_AFTER_STOP_SECONDS,
    )
    client.wait_for_sandbox.assert_called_once_with("owned", timeout=sb.BOOT_TIMEOUT_SECONDS)
    client.delete_sandbox.assert_called_once_with("owned")


@pytest.mark.parametrize("failure", [None, "run", "provision", "close"])
async def test_session_cleanup_on_success_and_failure(monkeypatch, failure):
    client = Mock()
    raw = SimpleNamespace()
    backend = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(sb, "SandboxClient", Mock(return_value=client))
    monkeypatch.setattr(sb, "find_snapshot", Mock(return_value=None))
    boot = Mock(return_value=raw)
    monkeypatch.setattr(sb, "boot", boot)
    monkeypatch.setattr(sb, "ResilientSandbox", Mock(return_value=backend))
    provision = Mock(side_effect=RuntimeError("provision") if failure == "provision" else None)
    monkeypatch.setattr(sb, "provision", provision)
    if failure == "close":
        backend.aclose.side_effect = RuntimeError("close")

    async def run():
        async with sb.sandbox_session() as owned:
            assert owned is backend
            if failure == "run":
                raise RuntimeError("run")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            await run()
    else:
        await run()
    client.delete_sandbox.assert_called_once_with(boot.call_args.args[2])
    assert backend.aclose.await_count == (0 if failure == "provision" else 1)
