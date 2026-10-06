"""The launch scripts' shared helpers (`scripts/_common.py`): reading .env, and knowing
when an installed node_modules no longer matches its lockfile."""

from __future__ import annotations

import sys

import pytest
from dotenv import dotenv_values

from deep_life_sci import paths

sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))

import _common


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(_common, "ENV_FILE", env)
    return env


# The server, the CLI and `langgraph deploy` all read .env with python-dotenv; the scripts
# read it with `env_value`, before dependencies are necessarily installed. A line the two read
# differently is a setting a deploy silently leaves behind.
LINES = [
    "OIDC_ALLOWED_EMAIL_DOMAINS=example.edu",
    "   OIDC_ALLOWED_EMAIL_DOMAINS=example.edu",  # uncommented by deleting only its '#'
    "\tOIDC_ISSUER=https://idp.example.edu",
    "export NCBI_EMAIL=me@example.org",
    "NCBI_API_KEY = k123",
    'OIDC_CLIENT_ID="abc-123"',
    "OIDC_CLIENT_ID='abc-123'",
    'OIDC_SCOPE="openid profile email offline_access"',
    "OIDC_SCOPE=openid profile email",
    "LANGSMITH_API_KEY=lsv2_x  # personal",
    'LANGSMITH_API_KEY="lsv2_x" # quoted, then a comment',
    "NCBI_TOOL=a#b",
    'NCBI_TOOL="with \\"escaped\\" quotes"',
    "OIDC_AUDIENCE=",
]


@pytest.mark.parametrize("line", LINES)
def test_env_value_reads_a_line_as_python_dotenv_does(env_file, line):
    env_file.write_text(f"# a comment\n{line}\n")
    ((key, expected),) = dotenv_values(env_file).items()
    assert _common.env_value(key) == expected


def test_env_value_takes_the_last_assignment_and_skips_comments(env_file):
    env_file.write_text("#   OIDC_ISSUER=https://commented.example\n"
                        "OIDC_ISSUER=https://first.example\n"
                        "OIDC_ISSUER=https://second.example\n")
    assert _common.env_value("OIDC_ISSUER") == dotenv_values(env_file)["OIDC_ISSUER"]
    _common.set_env("OIDC_ISSUER", "https://third.example")
    assert _common.env_value("OIDC_ISSUER") == "https://third.example"
    assert env_file.read_text().startswith("#   OIDC_ISSUER=https://commented.example\n")


def test_a_placeholder_still_counts_as_unset(env_file):
    env_file.write_text("  LANGSMITH_SANDBOX_API_KEY=lsv2_...\n")
    assert _common.env_value("LANGSMITH_SANDBOX_API_KEY") == ""


def test_an_install_is_current_only_for_the_lockfile_it_came_from(tmp_path):
    modules, lockfile = tmp_path / "node_modules", tmp_path / "pnpm-lock.yaml"
    modules.mkdir()
    lockfile.write_text("lockfileVersion: '9.0'\n")
    # node_modules that merely exists is not enough: setup never stamped it.
    assert not _common.deps_current(modules, lockfile)
    _common.stamp_deps(modules, lockfile)
    assert _common.deps_current(modules, lockfile)
    # A pull that changes the lockfile — a new dependency — makes it stale again.
    lockfile.write_text("lockfileVersion: '9.0'\npackages: {oidc-client-ts@3.5.0: {}}\n")
    assert not _common.deps_current(modules, lockfile)
    assert not _common.deps_current(tmp_path / "missing", lockfile)


def test_quiet_node_adds_no_deprecation_and_keeps_existing_options(monkeypatch):
    """The pinned pnpm calls url.parse(); Node 24's DEP0169 for it reads like a failed install."""
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    assert _common.quiet_node() == {"NODE_OPTIONS": "--no-deprecation"}
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=4096")
    assert _common.quiet_node() == {"NODE_OPTIONS": "--max-old-space-size=4096 --no-deprecation"}
