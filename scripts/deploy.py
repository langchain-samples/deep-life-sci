"""Deploy the agent server to LangSmith Deployment.

    uv run scripts/deploy.py                  # create, or update, the `deep-life-sci` deployment
    uv run scripts/deploy.py --name my-lab    # a deployment of another name
    uv run scripts/deploy.py --type prod      # the type only applies when one is created
    uv run scripts/deploy.py -- --remote      # anything after `--` goes to `langgraph deploy`

Then point the local chat UI at it with `uv run scripts/dev.py --remote <deployment URL>`.

A wrapper rather than `langgraph deploy` alone, for three reasons:

1. **Secrets.** `langgraph deploy` uploads every entry of the env file that the config names
   as a deployment secret. `langgraph.json` names `.env`, which also holds settings that
   belong to one developer's machine: model overrides for an eval sweep, a cache path, this
   install's NCBI `tool`. So a deploy reads `langgraph.deploy.json` instead — the same config
   with `env` pointed at `.env.deploy` — and this script writes `.env.deploy` from the
   allowlist in `deploy_secrets`, and nothing else.
2. **Keys.** The platform injects a `LANGSMITH_API_KEY` of its own under a reserved name,
   and nothing documents whether it may call the LLM gateway or create sandboxes. Both get
   an explicit key instead, so a deployment never depends on it.
3. **Workspace-global names.** Sandbox and snapshot names are shared by everything using a
   LangSmith workspace. The deployment gets its own sandbox name prefix and its own snapshot,
   so its sandboxes never answer to a local server's thread ids, and a local
   `build_snapshot.py` never deletes the snapshot a deployment is booting from.

Python dependencies resolve from pyproject.toml's bounds, not from uv.lock: the image
installs this package against the server's own constraints file. A locked export would ship
the `dev` group (langgraph-api, the CLI, ruff) and pin versions that constraints file may
reject, which is why the beta dependencies are capped in pyproject.toml instead.

Models are whatever models.yaml says when this runs. Editing it changes a deployment only on
the next deploy; the hot reload is a local-server convenience.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import ENV_FILE, REPO_ROOT, die, env_value, require_setup, say

TAG = "deploy"
CONFIG = REPO_ROOT / "langgraph.json"
DEPLOY_CONFIG = REPO_ROOT / "langgraph.deploy.json"
DEPLOY_ENV = REPO_ROOT / ".env.deploy"
DEFAULT_NAME = "deep-life-sci"
EU_HOST_URL = "https://eu.api.host.langchain.com"


def deploy_config() -> dict:
    """`langgraph.json`, with `env` naming the file this script writes."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["env"] = DEPLOY_ENV.name
    return config


def sync_deploy_config() -> None:
    """Keep the tracked twin in step with `langgraph.json`, which is the one people edit.

    Tracked rather than generated into an ignored file because a remote build ships only
    what `.gitignore` lets through, and the config it is told to build from has to be in it.
    """
    text = json.dumps(deploy_config(), indent=2) + "\n"
    if DEPLOY_CONFIG.exists() and DEPLOY_CONFIG.read_text(encoding="utf-8") == text:
        return
    DEPLOY_CONFIG.write_text(text, encoding="utf-8")
    say(TAG, f"updated {DEPLOY_CONFIG.name} to match {CONFIG.name}; commit it with that change")


def normalize(name: str) -> str:
    """The form the platform accepts: lowercase letters, digits and hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def snapshot_name(name: str) -> str:
    return f"pubmed-py-bio-{name}"


def deploy_secrets(name: str) -> dict[str, str]:
    """Everything the deployed server reads from its environment, and nothing else.

    Empty values are dropped, as `langgraph deploy` would drop them. Model settings are
    absent on purpose: the deployment runs what models.yaml says, not this machine's
    overrides.
    """
    langsmith_key = env_value("LANGSMITH_API_KEY")
    # One `tool` per deployment, derived from this install's own so that two people who
    # both keep the default deployment name are still two callers to NCBI.
    local_tool = env_value("NCBI_TOOL") or "deep_life_sci"
    secrets = {
        "LANGSMITH_GATEWAY_API_KEY": env_value("LANGSMITH_GATEWAY_API_KEY") or langsmith_key,
        "LANGSMITH_GATEWAY_ANTHROPIC_URL": env_value("LANGSMITH_GATEWAY_ANTHROPIC_URL"),
        "LANGSMITH_GATEWAY_BASE_URL": env_value("LANGSMITH_GATEWAY_BASE_URL"),
        "LANGSMITH_SANDBOX_API_KEY": env_value("LANGSMITH_SANDBOX_API_KEY") or langsmith_key,
        "SANDBOX_SNAPSHOT_NAME": snapshot_name(name),
        # Short, because it leads every sandbox name ahead of a 36-character thread id.
        "SANDBOX_NAME_PREFIX": name[:20].rstrip("-"),
        "NCBI_TOOL": f"{local_tool}_{name.replace('-', '_')}",
        "NCBI_EMAIL": env_value("NCBI_EMAIL"),
        "NCBI_API_KEY": env_value("NCBI_API_KEY"),
    }
    return {key: value for key, value in secrets.items() if value}


def write_deploy_env(secrets: dict[str, str]) -> None:
    lines = [
        "# Written by scripts/deploy.py on every deploy; edits here are overwritten.",
        "# Every entry becomes a secret on the LangSmith deployment.",
        *(f"{key}={value}" for key, value in secrets.items()),
    ]
    DEPLOY_ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"[{TAG}] {question} [Y/n] ").strip().lower() in ("", "y", "yes")


def ensure_snapshot(name: str, key: str, assume_yes: bool) -> None:
    """Build the deployment's own snapshot the first time. A missing one is slow, not
    broken: every new thread then pays the ~95s runtime install."""
    from langsmith.sandbox import SandboxClient

    try:
        snapshots = SandboxClient(api_key=key).list_snapshots(name_contains=name)
    except Exception as exc:  # noqa: BLE001 - any failure here is the same user-facing problem
        die(TAG, f"could not list sandbox snapshots ({exc}). Check LANGSMITH_API_KEY in .env.")
    if any(s.name == name for s in snapshots):
        say(TAG, f"sandbox snapshot {name} ready.")
        return
    if not _confirm(f"build the deployment's sandbox snapshot {name} now (~2 min)?", assume_yes):
        say(TAG, f"warning: no snapshot {name}; each new thread will install its packages "
                 f"at runtime (~95s). Build it later with:  "
                 f"uv run scripts/build_snapshot.py --name {name}")
        return
    env = {**os.environ, "LANGSMITH_SANDBOX_API_KEY": key}
    if subprocess.call(["uv", "run", "scripts/build_snapshot.py", "--name", name],
                       cwd=REPO_ROOT, env=env) != 0:
        die(TAG, "the snapshot build failed; see above.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="uv run scripts/deploy.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=DEFAULT_NAME, help="deployment name")
    parser.add_argument("--type", default="dev",
                        help="deployment type, used only when the deployment is created")
    parser.add_argument("--yes", "-y", action="store_true", help="build a missing snapshot "
                        "without asking")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="after `--`: arguments for `langgraph deploy`")
    args = parser.parse_args()
    passthrough = args.passthrough[1:] if args.passthrough[:1] == ["--"] else args.passthrough

    require_setup(TAG)
    name = normalize(args.name)
    if not name:
        die(TAG, f"{args.name!r} has no letters or digits to name a deployment with.")

    # The snapshot check below reaches LangSmith, and the endpoint comes from .env.
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)

    sync_deploy_config()
    secrets = deploy_secrets(name)
    write_deploy_env(secrets)
    say(TAG, f"wrote {DEPLOY_ENV.name}: {', '.join(secrets)}")
    if "NCBI_API_KEY" not in secrets:
        say(TAG, "warning: no NCBI_API_KEY in .env. A deployment is one caller to NCBI for "
                 "every user, and without a key that caller is held to 3 requests/sec.")

    ensure_snapshot(secrets["SANDBOX_SNAPSHOT_NAME"], secrets["LANGSMITH_SANDBOX_API_KEY"],
                    args.yes)

    env = {**os.environ, "LANGGRAPH_HOST_API_KEY": env_value("LANGSMITH_API_KEY")}
    if "eu." in os.environ.get("LANGSMITH_ENDPOINT", ""):
        env.setdefault("LANGGRAPH_HOST_URL", EU_HOST_URL)
    argv = ["uv", "run", "--group", "dev", "langgraph", "deploy",
            "--config", str(DEPLOY_CONFIG), "--name", name, "--deployment-type", args.type,
            *passthrough]
    code = subprocess.call(argv, cwd=REPO_ROOT, env=env)
    if code == 0:
        say(TAG, "try it from the local chat UI:  uv run scripts/dev.py --remote <deployment URL>")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
