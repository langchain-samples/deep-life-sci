"""Deploy the agent server to LangSmith Deployment.

    uv run scripts/deploy.py                  # create, or update, the `deep-life-sci` deployment
    uv run scripts/deploy.py --name my-lab    # a deployment of another name
    uv run scripts/deploy.py --type dedicated # the type only applies when one is created
    uv run scripts/deploy.py --size m         # Small, Medium or Large; changeable later
    uv run scripts/deploy.py -- --verbose     # anything after `--` goes to `langgraph deploy`

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

**Type and size** follow the usage-based pricing: Serverless or Dedicated, each Small, Medium
or Large (https://docs.langchain.com/langsmith/cloud-platform-features#sizes). The type is
fixed at creation; the size is not. The released `langgraph deploy` can only create the
previous pricing's `dev`/`prod` deployments, and a `dev` one can never become Serverless. So
this creates a new deployment itself, with the Deployments API's own name for the type
(`dev_zero` is what an existing Serverless deployment reports), sets its size, and hands the
build to `langgraph deploy --deployment-id`. Builds are remote: a deployment created for
uploaded source cannot take a locally built image.

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
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import ENV_FILE, REPO_ROOT, die, env_value, require_setup, say

TAG = "deploy"
CONFIG = REPO_ROOT / "langgraph.json"
DEPLOY_CONFIG = REPO_ROOT / "langgraph.deploy.json"
DEPLOY_ENV = REPO_ROOT / ".env.deploy"
DEFAULT_NAME = "deep-life-sci"
HOST_URL = "https://api.host.langchain.com"
EU_HOST_URL = "https://eu.api.host.langchain.com"

SIZES = ("s", "m", "l")
# The Deployments API's `deployment_type` for each type. Serverless is what this workspace's
# own Serverless deployments report; Dedicated is `prod`, the always-on type.
API_TYPES = {"serverless": "dev_zero", "dedicated": "prod"}


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


def tier(deployment_type: str, size: str) -> str:
    """The Deployments API's name for a type and size, e.g. `SERVERLESS_S`."""
    return f"{deployment_type}_{size}".upper()


def type_of(deployment: dict[str, Any]) -> str | None:
    """`serverless` or `dedicated`, read off an existing deployment's tier."""
    current = (deployment.get("deployment_tier") or "").split("_")[0].lower()
    return current if current in API_TYPES else None


class Host:
    """The Deployments API calls `langgraph deploy` makes with names it cannot send."""

    def __init__(self, url: str, api_key: str, transport: Any = None) -> None:
        import httpx

        headers = {"X-Api-Key": api_key}
        # The CLI sends the same header, for keys that span more than one workspace.
        if tenant := os.environ.get("LANGSMITH_TENANT_ID"):
            headers["X-Tenant-ID"] = tenant
        self._http = httpx.Client(base_url=url, headers=headers, timeout=30.0,
                                  transport=transport)

    def _check(self, response: Any) -> Any:
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json() if response.content else None

    def find(self, name: str) -> dict[str, Any] | None:
        found = self._check(self._http.get("/v2/deployments", params={"name_contains": name}))
        return next((d for d in found.get("resources", []) if d.get("name") == name), None)

    def create(self, name: str, deployment_type: str, secrets: dict[str, str]) -> str:
        """The request `langgraph deploy` makes for a remote build, with the new type name."""
        created = self._check(self._http.post("/v2/deployments", json={
            "name": name,
            "source": "internal_source",
            "source_config": {"deployment_type": API_TYPES[deployment_type]},
            "source_revision_config": {"langgraph_config_path": DEPLOY_CONFIG.name},
            "secrets": [{"name": k, "value": v} for k, v in secrets.items()],
        }))
        return str(created["id"])

    def set_tier(self, deployment_id: str, deployment_tier: str) -> None:
        self._check(self._http.patch(f"/v2/deployments/{deployment_id}/deployment-tier",
                                     json={"deployment_tier": deployment_tier}))


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
    parser.add_argument("--type", choices=tuple(API_TYPES),
                        help="serverless (default) or dedicated; fixed once the deployment "
                        "is created")
    parser.add_argument("--size", choices=SIZES, type=str.lower,
                        help="Small, Medium or Large (default: s for a new deployment; an "
                        "existing one keeps its size unless this is given)")
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

    api_key = env_value("LANGSMITH_API_KEY")
    host_url = os.environ.get("LANGGRAPH_HOST_URL") or (
        EU_HOST_URL if "eu." in os.environ.get("LANGSMITH_ENDPOINT", "") else HOST_URL
    )
    host = Host(host_url, api_key)
    try:
        deployment = host.find(name)
        if deployment is None:
            deployment_type = args.type or "serverless"
            say(TAG, f"creating {name} ({deployment_type})")
            deployment_id = host.create(name, deployment_type, secrets)
            current = None
        else:
            deployment_id = str(deployment["id"])
            current = deployment.get("deployment_tier")
            deployment_type = type_of(deployment)
            if args.type and deployment_type and args.type != deployment_type:
                say(TAG, f"warning: {name} is {deployment_type}, and a type cannot change; "
                         f"deploy under another --name for {args.type}.")
    except Exception as exc:  # noqa: BLE001 - refused or unreachable, the same dead end
        die(TAG, f"the Deployments API refused: {exc}")

    # A new deployment gets the size asked for, Small by default; an existing one keeps
    # whatever it was resized to unless --size says otherwise. A size the platform refuses
    # is reported rather than fatal: the deployment can still be built, and resized later.
    size = args.size or ("s" if deployment is None else None)
    if size and deployment_type and current != tier(deployment_type, size):
        try:
            host.set_tier(deployment_id, tier(deployment_type, size))
            current = tier(deployment_type, size)
        except Exception as exc:  # noqa: BLE001 - reported; the deploy can go ahead
            say(TAG, f"warning: could not size {name} as {tier(deployment_type, size)} ({exc}); "
                     "change it on its LangSmith page.")
    say(TAG, f"deploying {name} ({current or 'its default size'})")

    env = {**os.environ, "LANGGRAPH_HOST_API_KEY": api_key, "LANGGRAPH_HOST_URL": host_url}
    argv = ["uv", "run", "--group", "dev", "langgraph", "deploy",
            "--config", str(DEPLOY_CONFIG), "--deployment-id", deployment_id, "--remote",
            *passthrough]
    code = subprocess.call(argv, cwd=REPO_ROOT, env=env)
    if code == 0:
        say(TAG, "try it from the local chat UI:  uv run scripts/dev.py --remote <deployment URL>")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
