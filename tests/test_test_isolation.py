"""The unit suite must fail even when application code catches a network attempt."""

import os
import subprocess
import sys

from deep_life_sci.paths import REPO_ROOT


def test_caught_network_attempt_still_fails_the_test(tmp_path):
    child = tmp_path / "test_accidental_network.py"
    child.write_text("""
import socket

def test_caught_connect():
    with socket.socket() as sock:
        try:
            sock.connect(("127.0.0.1", 9))
        except Exception:
            pass
""")
    program = (
        "import sys, pytest; from tests import conftest; "
        "sys.exit(pytest.main([sys.argv[1], '-q'], plugins=[conftest]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(child)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "A test attempted a real network connection" in result.stdout


def test_dotenv_loading_is_disabled_before_entry_point_imports(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    monkeypatch.delenv("TEST_DOTENV_SENTINEL", raising=False)
    config = tmp_path / "test-config.txt"
    config.write_text("TEST_DOTENV_SENTINEL=should-not-load\n")
    assert load_dotenv(config, override=True) is False
    assert "TEST_DOTENV_SENTINEL" not in os.environ
