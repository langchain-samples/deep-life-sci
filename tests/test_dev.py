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
    monkeypatch.setattr(
        dev, "open_ui_when_ready",
        lambda *, wait_for_agent: events.append(("wait for UI", wait_for_agent)),
    )
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
        assert launcher[3] == ("wait for UI", True)
    else:
        # A reused UI still waits: the agent server this run started may not answer yet.
        assert launcher[2] == ("wait for UI", True)


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


@pytest.fixture
def opener(monkeypatch):
    """`open_ui_when_ready` run inline, with no real sleeping, recording what it opens."""
    monkeypatch.delenv("NO_BROWSER", raising=False)
    pauses, opened = [], []
    monkeypatch.setattr(dev.time, "sleep", pauses.append)
    monkeypatch.setattr(dev.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        dev.threading, "Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )
    return pauses, opened


def test_browser_waits_until_ui_listens(opener, monkeypatch):
    pauses, opened = opener
    states = iter([False, False, True])
    monkeypatch.setattr(dev, "listening", lambda port: next(states))
    dev.open_ui_when_ready(wait_for_agent=False)
    assert pauses == [0.5, 0.5]
    assert opened == [dev.UI_URL]


def test_browser_also_waits_for_a_local_agent_server(opener, monkeypatch):
    """The page checks the server as it loads; arriving first showed a connection error."""
    pauses, opened = opener
    monkeypatch.setattr(dev, "listening", lambda port: True)
    answers = iter([False, False, True])
    monkeypatch.setattr(dev, "answering", lambda port, **kwargs: port == 2024 and next(answers))
    dev.open_ui_when_ready(wait_for_agent=True)
    assert pauses == [0.5, 0.5]
    assert opened == [dev.UI_URL]


def test_a_server_that_never_answers_is_named_not_opened(opener, monkeypatch, capsys):
    _, opened = opener
    clock = iter(range(0, 10_000, 60))
    monkeypatch.setattr(dev.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(dev, "listening", lambda port: True)
    monkeypatch.setattr(dev, "answering", lambda *args, **kwargs: False)
    dev.open_ui_when_ready(wait_for_agent=True)
    assert opened == []
    assert "agent server is still not answering" in capsys.readouterr().out


def test_no_browser_opens_nothing(opener, monkeypatch):
    _, opened = opener
    monkeypatch.setenv("NO_BROWSER", "1")
    monkeypatch.setattr(dev, "listening", lambda port: True)
    dev.open_ui_when_ready(wait_for_agent=False)
    assert opened == []
