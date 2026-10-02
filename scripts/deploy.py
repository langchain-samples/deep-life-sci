"""Deploy the agent server to LangSmith Deployment.

    uv run scripts/deploy.py                  # create, or update, `deep-life-sci-cloud`
    uv run scripts/deploy.py --name my-lab    # a deployment of another name
    uv run scripts/deploy.py --type serverless  # cheaper; the type is fixed at creation
    uv run scripts/deploy.py --size m         # Small, Medium or Large; changeable later
    uv run scripts/deploy.py --auth langsmith # LangSmith API keys only, no sign-in
    uv run scripts/deploy.py -- --verbose     # anything after `--` goes to `langgraph deploy`

Then point the local chat UI at it with `uv run scripts/dev.py --remote <deployment URL>`.

A wrapper rather than `langgraph deploy` alone, for three reasons:

1. **Secrets.** `langgraph deploy` uploads every entry of the env file that the config names
   as a deployment secret. `langgraph.json` names `.env`, which also holds settings that
   belong to one developer's machine: model overrides for an eval sweep, a cache path, this
   install's NCBI `tool`. So a deploy reads `langgraph.deploy.json` instead — the same config
   with `env` pointed at `.env.deploy` — and this script writes `.env.deploy` from the
   allowlist in `deploy_settings`, and nothing else.
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
(`dev_zero` is what an existing Serverless deployment reports) and hands the build to
`langgraph deploy --deployment-id`. It starts at its type's default size (Small); a
`--size` other than that is set once the first build has made a revision to resize. Builds
are remote: a deployment created for uploaded source cannot take a locally built image.

Dedicated is the default because the deployment is user-facing: a Serverless one that has
scaled to zero makes the first person back wait for it to start. Serverless suits trying the
agent out.

**Auth** is `oidc` — people sign in with the institution's identity provider, configured by
the OIDC_* settings in .env (see .env.example and deep_life_sci/auth.py) — when OIDC_ISSUER
is set, and `langsmith` otherwise: only holders of a LangSmith API key for the workspace
(Studio, scripts, `dev.py --remote`). LangSmith keys work in both. The deploy config always
carries the auth handler; `DEEP_LIFE_SCI_AUTH` tells it which mode it is in.

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
# Not `deep-life-sci`: a deployment creates a tracing project of its own name, and that is
# the project every local run already traces to (.env.example's LANGSMITH_PROJECT).
DEFAULT_NAME = "deep-life-sci-cloud"
HOST_URL = "https://api.host.langchain.com"
EU_HOST_URL = "https://eu.api.host.langchain.com"

SIZES = ("s", "m", "l")
# The Deployments API's `deployment_type` for each type. `dev_zero` is what this workspace's
# own Serverless deployments report. `prod` for Dedicated is the always-on type's name, not
# yet seen on a Dedicated deployment: `resize` warns if the platform disagrees.
API_TYPES = {"serverless": "dev_zero", "dedicated": "prod"}
DEFAULT_TYPE = "dedicated"


# What a deployment adds to `langgraph.json`. None of it applies to a local server, which
# stays without auth: see deep_life_sci/auth.py for the handler and its two modes.
DEPLOY_AUTH = {"path": "./deep_life_sci/auth.py:auth", "allow_langsmith_api_keys": True}
# The store API would hand any signed-in user every thread's uploads; the graph's own store
# calls do not go through it and are unaffected.
DEPLOY_HTTP = {"disable_store": True}
# The chat UI's static build, which webapp.py serves under /app when sign-in is on. Built in
# every image, so the deploy config is the same in both auth modes; in `langsmith` mode it is
# just not served. These lines run before the image installs Node or copies the repo in, so
# they do both for themselves, and drop everything but the built files. Node 22 because the
# UI's Next wants 20.9 or later. The pnpm pin is frontend/package.json's `packageManager`.
UI_DIR_IN_IMAGE = "/opt/deep-life-sci-ui"
DEPLOY_DOCKERFILE_LINES = [
    "ENV NODE_VERSION=22",
    "RUN /storage/install-node.sh",
    "ADD frontend /opt/deep-life-sci-frontend",
    "RUN cd /opt/deep-life-sci-frontend"
    " && npx --yes pnpm@10.5.1 install --frozen-lockfile"
    " && DEEP_LIFE_SCI_STATIC=1 npx --yes pnpm@10.5.1 build"
    f" && mv out {UI_DIR_IN_IMAGE}"
    " && cd / && rm -rf /opt/deep-life-sci-frontend"
    " /root/.npm /root/.cache /root/.local/share/pnpm",
    f"ENV DEEP_LIFE_SCI_UI_DIR={UI_DIR_IN_IMAGE}",
]
AUTH_MODES = ("oidc", "langsmith")


def deploy_config() -> dict:
    """`langgraph.json`, with `env` naming the file this script writes, plus auth."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["env"] = DEPLOY_ENV.name
    config["auth"] = DEPLOY_AUTH
    config["http"] = {**config.get("http", {}), **DEPLOY_HTTP}
    config["dockerfile_lines"] = DEPLOY_DOCKERFILE_LINES
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


# Read by the Agent Server, not this package. A redeploy stops a revision's queue workers,
# and a run still going when the grace period ends is resumed on the new revision from its
# last checkpoint, re-running the step it was in. The default is 180s; research runs often
# take longer, so most would be cut off and partly repeated. Longer means a rollout can wait
# up to this long for in-flight runs. Resuming is safe (see tests/test_upload_lifecycle.py
# and tests/test_artifact_lifecycle.py), just slower and costlier than finishing.
PLATFORM_SETTINGS = {"BG_JOB_SHUTDOWN_GRACE_PERIOD_SECS": "600"}

OIDC_SETTINGS = ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_AUDIENCE", "OIDC_SCOPE", "OIDC_TOKEN",
                 "OIDC_ALLOWED_EMAIL_DOMAINS")


def deploy_settings(name: str, auth_mode: str = "langsmith") -> dict[str, str]:
    """Everything the deployed server reads from its environment, and nothing else.

    Empty values are dropped, as `langgraph deploy` would drop them. Model settings are
    absent on purpose: the deployment runs what models.yaml says, not this machine's
    overrides.
    """
    langsmith_key = env_value("LANGSMITH_API_KEY")
    # One `tool` per deployment, derived from this install's own so that two people who
    # both keep the default deployment name are still two callers to NCBI.
    local_tool = env_value("NCBI_TOOL") or "deep_life_sci"
    settings = {
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
        "DEEP_LIFE_SCI_AUTH": auth_mode,
        **PLATFORM_SETTINGS,
    }
    if auth_mode == "oidc":
        settings |= {key: env_value(key) for key in OIDC_SETTINGS}
    return {key: value for key, value in settings.items() if value}



def auth_mode(asked: str | None) -> str:
    """`--auth`, or `oidc` when .env configures a provider and `langsmith` when it doesn't."""
    mode = asked or ("oidc" if env_value("OIDC_ISSUER") else "langsmith")
    if mode == "oidc":
        missing = [key for key in ("OIDC_ISSUER", "OIDC_CLIENT_ID") if not env_value(key)]
        if missing:
            die(TAG, f"--auth oidc needs {' and '.join(missing)} in .env; see .env.example.")
    return mode


def write_deploy_env(settings: dict[str, str]) -> None:
    lines = [
        "# Written by scripts/deploy.py on every deploy; edits here are overwritten.",
        "# Every entry becomes a secret on the LangSmith deployment.",
        *(f"{key}={value}" for key, value in settings.items()),
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


def resize(host: Host, name: str, deployment_id: str, wanted: str) -> str | None:
    """Set the size tier, and return it; a refusal is reported, never fatal, because the
    deployment works at any size and can be resized from its LangSmith page."""
    try:
        host.set_tier(deployment_id, wanted)
    except Exception as exc:  # noqa: BLE001 - reported; the deploy can go ahead
        say(TAG, f"warning: could not size {name} as {wanted} ({exc}); change it on its "
                 "LangSmith page.")
        return None
    say(TAG, f"sized {name} as {wanted}")
    return wanted


def deployment_url(host: Host, name: str) -> str:
    """The deployment's URL, or a placeholder if the lookup fails after a good deploy."""
    try:
        deployment = host.find(name)
    except Exception:  # noqa: BLE001 - the deploy already succeeded; this is a courtesy
        deployment = None
    return (deployment or {}).get("url") or "<deployment URL from its LangSmith page>"


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
                        help=f"dedicated or serverless (default {DEFAULT_TYPE}); fixed once the "
                        "deployment is created")
    parser.add_argument("--size", choices=SIZES, type=str.lower,
                        help="Small, Medium or Large; without it a new deployment starts at "
                        "its type's default size and an existing one keeps its own")
    parser.add_argument("--auth", choices=AUTH_MODES,
                        help="oidc (sign-in; the default when .env sets OIDC_ISSUER) or "
                        "langsmith (API keys only)")
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

    if name == normalize(os.environ.get("LANGSMITH_PROJECT", "")):
        die(TAG, f"{name!r} is the tracing project your local runs use (LANGSMITH_PROJECT), and "
                 "a deployment creates a tracing project of its own name. Pick another --name.")

    sync_deploy_config()
    mode = auth_mode(args.auth)
    say(TAG, "auth: " + ("sign-in with " + env_value("OIDC_ISSUER") if mode == "oidc"
                         else "LangSmith API keys only"))
    # Named `settings` rather than `secrets`: most are not secret, and every one of them
    # becomes a deployment secret. Only their names are ever printed.
    settings = deploy_settings(name, mode)
    write_deploy_env(settings)
    say(TAG, f"wrote {DEPLOY_ENV.name}: {', '.join(sorted(settings))}")
    if "NCBI_API_KEY" not in settings:
        say(TAG, "warning: no NCBI_API_KEY in .env. A deployment is one caller to NCBI for "
                 "every user, and without a key that caller is held to 3 requests/sec.")

    api_key = env_value("LANGSMITH_API_KEY")
    host_url = os.environ.get("LANGGRAPH_HOST_URL") or (
        EU_HOST_URL if "eu." in os.environ.get("LANGSMITH_ENDPOINT", "") else HOST_URL
    )
    host = Host(host_url, api_key)
    try:
        deployment = host.find(name)
        if deployment is None:
            deployment_type = args.type or DEFAULT_TYPE
            say(TAG, f"creating {name} ({deployment_type})")
            deployment_id = host.create(name, deployment_type, settings)
            # Read back rather than assumed: a new deployment starts at its type's default
            # size, and that is the size the platform reports, not one we pick.
            deployment = host.find(name) or {"id": deployment_id}
            created = True
        else:
            deployment_id = str(deployment["id"])
            deployment_type = type_of(deployment)
            created = False
            asked = args.type or DEFAULT_TYPE
            if deployment_type and asked != deployment_type:
                say(TAG, f"warning: {name} is {deployment_type}, and a type cannot change, so "
                         f"it stays {deployment_type}. Deploy under another --name for {asked}.")
    except Exception as exc:  # noqa: BLE001 - refused or unreachable, the same dead end
        die(TAG, f"the Deployments API refused: {exc}")

    current = deployment.get("deployment_tier")
    wanted = tier(deployment_type, args.size) if args.size and deployment_type else None
    # An existing deployment is resized before the build, so the new revision has the size.
    # A new one only after it: resizing a deployment with no revision yet answered HTTP 500.
    if wanted and wanted != current and not created:
        current = resize(host, name, deployment_id, wanted) or current
    # After the Deployments API has accepted the name, so that a refusal costs seconds rather
    # than the snapshot build that would otherwise come first.
    ensure_snapshot(snapshot_name(name), settings["LANGSMITH_SANDBOX_API_KEY"],
                    args.yes)
    say(TAG, f"deploying {name} ({current or 'its default size'})")

    env = {**os.environ, "LANGGRAPH_HOST_API_KEY": api_key, "LANGGRAPH_HOST_URL": host_url}
    argv = ["uv", "run", "--group", "dev", "langgraph", "deploy",
            "--config", str(DEPLOY_CONFIG), "--deployment-id", deployment_id, "--remote",
            *passthrough]
    code = subprocess.call(argv, cwd=REPO_ROOT, env=env)
    if code == 0 and wanted and wanted != current and created:
        resize(host, name, deployment_id, wanted)
    if code == 0:
        url = deployment_url(host, name)
        if mode == "oidc":
            say(TAG, f"open the chat UI:  {url}/app/")
        else:
            say(TAG, f"try it from the local chat UI:  uv run scripts/dev.py --remote {url}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
