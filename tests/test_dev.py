"""Launcher regressions: port prompts stay visible and startup reaches the browser."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from deep_life_sci import paths

sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))

import dev


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    ui = tmp_path / "frontend"
    (ui / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(dev, "_started", [])
    monkeypatch.setattr(dev, "require_setup", lambda tag: None)
    monkeypatch.setattr(dev, "frontend_dir", lambda: ui)
    monkeypatch.setattr(dev, "deps_current", lambda *args: True)
    monkeypatch.setattr(dev, "pnpm_or_die", lambda tag: SimpleNamespace(argv=["pnpm"]))
    monkeypatch.setattr(dev, "answering", lambda *args, **kwargs: True)
    monkeypatch.setattr(sys, "argv", ["dev.py"])
    monkeypatch.delenv("NO_BROWSER", raising=False)
    events = []

    def spawn(name, cwd, argv, env=None):
        events.append((name, argv, env))
        dev._started.append((name, SimpleNamespace(poll=lambda: None)))

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(dev, "spawn", spawn)
    monkeypatch.setattr(dev.time, "sleep", interrupt)
    monkeypatch.setattr(dev, "open_ui_when_ready", lambda: events.append("wait for UI"))
    monkeypatch.setattr(dev.webbrowser, "open", lambda url: events.append(("browser", url)))
    return events


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("replace_ui", [False, True])
def test_ui_port_prompt_finishes_before_agent_logs_start(
    launcher, monkeypatch, windows, replace_ui,
):
    monkeypatch.setattr(dev, "WINDOWS", windows)
    monkeypatch.setattr(dev, "listening", lambda port: port == 3000)

    def take_over(port, *args):
        assert port == 3000
        assert not dev._started, "server output must not bury the pending prompt"
        launcher.append("port decision")
        return replace_ui

    monkeypatch.setattr(dev, "_take_over", take_over)
    assert dev._launch() == 0
    assert launcher[0] == "port decision"
    name, argv, env = launcher[1]
    assert name == "agent"
    assert "--no-reload" in argv  # no .venv-triggered reload loop on any platform
    assert "--no-sync" in argv  # launching must not mutate installed dependencies
    assert env == ({"PYTHONUTF8": "1"} if windows else None)
    if replace_ui:
        assert launcher[2][0] == "ui"
        assert launcher[3] == "wait for UI"
    else:
        assert launcher[2] == ("browser", dev.UI_URL)


def test_refusing_an_unresponsive_ui_starts_no_server(launcher, monkeypatch):
    monkeypatch.setattr(dev, "listening", lambda port: port == 3000)
    monkeypatch.setattr(dev, "answering", lambda *args, **kwargs: False)
    monkeypatch.setattr(dev, "_port_holder", lambda port: (123, "/other/checkout"))
    monkeypatch.setattr(dev, "_ask", lambda *args, **kwargs: False)
    with pytest.raises(SystemExit):
        dev._launch()
    assert launcher == []
    assert dev._started == []


def test_reusing_both_servers_opens_browser_and_exits(launcher, monkeypatch):
    monkeypatch.setattr(dev, "listening", lambda port: True)
    monkeypatch.setattr(dev, "_take_over", lambda *args: False)
    assert dev._launch() == 0
    assert launcher == [("browser", dev.UI_URL)]
    assert dev._started == []


def test_remote_ui_conflict_starts_no_server(launcher, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dev.py", "--remote", "https://example.com"])
    monkeypatch.setattr(dev, "listening", lambda port: port == 3000)
    monkeypatch.setattr(dev, "_take_over", lambda *args: False)
    with pytest.raises(SystemExit):
        dev._launch()
    assert launcher == []


def test_browser_waits_until_ui_listens(monkeypatch):
    monkeypatch.delenv("NO_BROWSER", raising=False)
    states = iter([False, False, True])
    monkeypatch.setattr(dev, "listening", lambda port: next(states))
    pauses, opened = [], []
    monkeypatch.setattr(dev.time, "sleep", pauses.append)
    monkeypatch.setattr(dev.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        dev.threading, "Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )
    dev.open_ui_when_ready()
    assert pauses == [0.5, 0.5]
    assert opened == [dev.UI_URL]
