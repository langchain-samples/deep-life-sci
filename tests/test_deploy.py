"""What a LangSmith deployment is built from, and what `scripts/deploy.py` hands it.

Nothing here reaches LangSmith. These pin the parts of a deploy that fail silently or only
at build time on the platform: a Python version the package cannot install on, artifact
components whose dependencies never get installed, a secret the server never reads, or a
developer's own `.env` following the code into a shared server.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from unittest.mock import Mock

import pytest

from deep_life_sci import paths

REPO = paths.REPO_ROOT
sys.path.insert(0, str(REPO / "scripts"))

import _common  # noqa: E402
import deploy  # noqa: E402

CONFIG = json.loads((REPO / "langgraph.json").read_text(encoding="utf-8"))


def _read(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


class TestTheImageCanInstallThePackage:
    def test_python_version_satisfies_requires_python(self):
        """The server image defaults to 3.11, which `pip install -e .` rejects outright."""
        requires = tomllib.loads(_read("pyproject.toml"))["project"]["requires-python"]
        floor = re.fullmatch(r">=(\d+)\.(\d+)", requires)
        assert floor, f"unexpected requires-python {requires!r}"
        version = tuple(int(part) for part in CONFIG["python_version"].split("."))
        assert version >= (int(floor[1]), int(floor[2]))

    def test_artifact_dependencies_install_beside_langgraph_json(self):
        """The build runs its npm install in langgraph.json's directory and nowhere else;
        a manifest only in ui/ leaves `xlsx` unresolved while /ui/<graph> still answers 200."""
        manifest = json.loads(_read("package.json"))
        lock = json.loads(_read("package-lock.json"))
        assert not (REPO / "ui" / "package-lock.json").exists(), "one lockfile, at the root"
        for component in CONFIG["ui"].values():
            workspace = component.removeprefix("./").split("/")[0]
            assert workspace in manifest["workspaces"]
            ui_manifest = json.loads(_read(f"{workspace}/package.json"))
            locked = lock["packages"][workspace]["dependencies"]
            assert locked == ui_manifest["dependencies"], "package-lock.json is stale"

    def test_deploy_config_is_langgraph_json_plus_the_deploy_only_keys(self):
        """A local server stays without auth; only the deployment gets the handler."""
        assert json.loads(_read("langgraph.deploy.json")) == deploy.deploy_config()
        assert "auth" not in CONFIG
        assert deploy.deploy_config() == {
            **CONFIG,
            "env": ".env.deploy",
            "auth": {"path": "./deep_life_sci/auth.py:auth", "allow_langsmith_api_keys": True},
            "http": {**CONFIG["http"], "disable_store": True},
            "dockerfile_lines": deploy.DEPLOY_DOCKERFILE_LINES,
        }

    def test_the_image_builds_the_ui_where_the_server_looks_for_it(self):
        lines = deploy.DEPLOY_DOCKERFILE_LINES
        assert f"ENV DEEP_LIFE_SCI_UI_DIR={deploy.UI_DIR_IN_IMAGE}" in lines
        build = next(line for line in lines if "pnpm" in line)
        assert "DEEP_LIFE_SCI_STATIC=1" in build and f"mv out {deploy.UI_DIR_IN_IMAGE}" in build
        pin = json.loads(_read("frontend/package.json"))["packageManager"].split("+")[0]
        assert pin in build, "the image's pnpm must match frontend's packageManager pin"

    @pytest.mark.parametrize("config", ["langgraph.json", "langgraph.deploy.json"])
    def test_the_thread_ttl_is_the_one_each_run_restarts(self, config):
        """graph.py sets a thread's TTL to `paths.THREAD_TTL_MINUTES` at every run; the
        config's TTL is what a new thread starts with, so the two must agree. A thread's
        uploads, in the store, are kept as long as the thread is (README.md)."""
        settings = json.loads(_read(config))
        ttl = settings["checkpointer"]["ttl"]
        assert ttl["strategy"] == "delete"
        assert ttl["default_ttl"] == paths.THREAD_TTL_MINUTES
        store = settings["store"]["ttl"]
        assert store["refresh_on_read"] is True
        assert store["default_ttl"] == paths.THREAD_TTL_MINUTES

    def test_the_auth_handler_the_deploy_config_names_exists(self):
        module, _, name = deploy.DEPLOY_AUTH["path"].partition(":")
        assert (REPO / module).is_file()
        from deep_life_sci import auth

        assert getattr(auth, name) is auth.auth

    def test_secrets_files_stay_out_of_the_image(self):
        ignored = set(_read(".dockerignore").split("\n"))
        assert {".env", ".env.*"} <= ignored
        # A remote build uploads what .gitignore lets through, too.
        assert ".env.*" in _read(".gitignore").split("\n")

    @pytest.mark.parametrize(
        "needed",
        ["langgraph.json", "langgraph.deploy.json", "package.json", "package-lock.json",
         "models.yaml", "pyproject.toml", "README.md", "deep_life_sci/", "ui/", "frontend/"],
    )
    def test_build_inputs_are_not_ignored(self, needed):
        entries = [line.strip() for line in _read(".dockerignore").splitlines()]
        assert needed not in entries and needed.rstrip("/") not in entries


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    """A `.env` for `scripts/_common.env_value` to read, instead of the developer's."""
    env = tmp_path / ".env"
    monkeypatch.setattr(_common, "ENV_FILE", env)

    def write(**values: str) -> None:
        env.write_text("".join(f"{key}={value}\n" for key, value in values.items()))

    return write


class TestDeploySecrets:
    def test_only_the_allowlist_ships_and_both_keys_are_explicit(self, dotenv):
        dotenv(
            LANGSMITH_API_KEY="lsv2_personal",
            NCBI_TOOL="deep_life_sci_ab12cd34",
            NCBI_EMAIL="me@example.org",
            ROOT_MODEL="claude-sonnet-5",
            DEEP_LIFE_SCI_DATA_DIR="/tmp/cache",
            NCBI_API_KEY="",
        )
        secrets = deploy.deploy_settings("my-lab")
        assert secrets == {
            "LANGSMITH_GATEWAY_API_KEY": "lsv2_personal",
            "LANGSMITH_SANDBOX_API_KEY": "lsv2_personal",
            "SANDBOX_SNAPSHOT_NAME": "pubmed-py-bio-my-lab",
            "SANDBOX_NAME_PREFIX": deploy.sandbox_prefix("my-lab"),
            "NCBI_TOOL": "deep_life_sci_ab12cd34_my_lab",
            "NCBI_EMAIL": "me@example.org",
            "DEEP_LIFE_SCI_AUTH": "langsmith",
            "BG_JOB_SHUTDOWN_GRACE_PERIOD_SECS": "600",
        }

    def test_oidc_settings_ship_only_in_oidc_mode(self, dotenv):
        dotenv(LANGSMITH_API_KEY="k", OIDC_ISSUER="https://idp.example", OIDC_CLIENT_ID="app")
        assert "OIDC_ISSUER" not in deploy.deploy_settings("x", "langsmith")
        secrets = deploy.deploy_settings("x", "oidc")
        assert secrets["DEEP_LIFE_SCI_AUTH"] == "oidc"
        assert secrets["OIDC_ISSUER"] == "https://idp.example"
        assert secrets["OIDC_CLIENT_ID"] == "app"

    def test_auth_mode_follows_env_and_refuses_half_a_provider(self, dotenv):
        dotenv(LANGSMITH_API_KEY="k")
        assert deploy.auth_mode(None) == "langsmith"
        dotenv(LANGSMITH_API_KEY="k", OIDC_ISSUER="https://idp.example")
        with pytest.raises(SystemExit):
            deploy.auth_mode(None)
        dotenv(LANGSMITH_API_KEY="k", OIDC_ISSUER="https://idp.example", OIDC_CLIENT_ID="a")
        assert deploy.auth_mode(None) == "oidc"
        assert deploy.auth_mode("langsmith") == "langsmith"

    def test_overrides_in_env_win_over_the_main_key(self, dotenv):
        dotenv(
            LANGSMITH_API_KEY="lsv2_personal",
            LANGSMITH_GATEWAY_API_KEY="lsv2_gateway",
            LANGSMITH_SANDBOX_API_KEY="lsv2_sandbox",
        )
        secrets = deploy.deploy_settings("x")
        assert secrets["LANGSMITH_GATEWAY_API_KEY"] == "lsv2_gateway"
        assert secrets["LANGSMITH_SANDBOX_API_KEY"] == "lsv2_sandbox"

    def test_sandbox_prefix_stays_short(self, dotenv):
        dotenv(LANGSMITH_API_KEY="lsv2_personal")
        name = deploy.normalize("A Very Long Deployment Name For Our Lab")
        prefix = deploy.deploy_settings(name)["SANDBOX_NAME_PREFIX"]
        assert len(prefix) <= 16 and not prefix.startswith("-")

    def test_sandbox_prefixes_differ_wherever_names_do(self):
        """Cut short, `deep-life-sci-cloud-staging` was `deep-life-sci-cloud`, and the two
        deployments' sandboxes answered to each other's thread ids."""
        names = ["deep-life-sci-cloud", "deep-life-sci-cloud-staging", "deep-life-sci-cloud-2",
                 "pubmed", "pubmed-agent-production", "pubmed-agent-production-2"]
        prefixes = {deploy.sandbox_prefix(name) for name in names}
        assert len(prefixes) == len(names)
        # Nor ever the prefix a local server uses.
        assert "pubmed" not in prefixes

    def test_no_secret_is_one_the_platform_reserves(self, dotenv):
        reserved = pytest.importorskip("langgraph_cli.deploy").RESERVED_ENV_VARS
        dotenv(LANGSMITH_API_KEY="k", NCBI_API_KEY="k", NCBI_EMAIL="e",
               LANGSMITH_GATEWAY_ANTHROPIC_URL="u", LANGSMITH_GATEWAY_BASE_URL="u",
               OIDC_ISSUER="i", OIDC_CLIENT_ID="c", OIDC_AUDIENCE="a", OIDC_SCOPE="s",
               OIDC_TOKEN="id", OIDC_ALLOWED_EMAIL_DOMAINS="d")
        assert not set(deploy.deploy_settings("x", "oidc")) & reserved

    def test_every_secret_is_read_by_the_package(self, dotenv):
        """A secret nothing reads is a setting that silently does nothing."""
        dotenv(LANGSMITH_API_KEY="k", NCBI_API_KEY="k", NCBI_EMAIL="e",
               LANGSMITH_GATEWAY_ANTHROPIC_URL="u", LANGSMITH_GATEWAY_BASE_URL="u",
               OIDC_ISSUER="i", OIDC_CLIENT_ID="c", OIDC_AUDIENCE="a", OIDC_SCOPE="s",
               OIDC_TOKEN="id", OIDC_ALLOWED_EMAIL_DOMAINS="d")
        source = "".join(
            path.read_text(encoding="utf-8") for path in (REPO / "deep_life_sci").rglob("*.py")
        )
        for name in set(deploy.deploy_settings("x", "oidc")) - set(deploy.PLATFORM_SETTINGS):
            assert f'"{name}"' in source, name


def test_the_default_name_is_not_the_local_tracing_project():
    """The platform refuses a deployment named after an existing tracing project."""
    example = dict(
        line.split("=", 1) for line in _read(".env.example").splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    assert deploy.normalize(example["LANGSMITH_PROJECT"]) != deploy.DEFAULT_NAME


class TestTypeAndSize:
    @pytest.fixture
    def requests(self):
        import httpx

        seen = []

        def handle(request):
            seen.append(request)
            if request.method == "POST":
                return httpx.Response(201, json={"id": "dep-1"})
            if request.method == "PATCH":
                return httpx.Response(422, json={"detail": "tier not allowed"})
            return httpx.Response(200, json={"resources": [
                {"id": "dep-2", "name": "lab-x", "deployment_tier": "SERVERLESS_S"},
                {"id": "dep-3", "name": "lab", "deployment_tier": "DEDICATED_M",
                 "url": "https://lab-123.us.langgraph.app"},
            ]})

        transport = httpx.MockTransport(handle)
        host = deploy.Host("https://host.example", "lsv2_key", transport=transport)
        return host, seen

    def test_a_new_deployment_is_created_under_the_api_type_for_a_remote_build(self, requests):
        host, seen = requests
        assert host.create("lab", "serverless", {"NCBI_TOOL": "t"}) == "dep-1"
        body = json.loads(seen[-1].content)
        assert body == {
            "name": "lab",
            "source": "internal_source",
            "source_config": {"deployment_type": "dev_zero"},
            "source_revision_config": {"langgraph_config_path": "langgraph.deploy.json"},
            "secrets": [{"name": "NCBI_TOOL", "value": "t"}],
        }
        assert seen[-1].headers["x-api-key"] == "lsv2_key"

    def test_lookup_is_by_exact_name_and_reads_the_type_off_the_tier(self, requests):
        host, _ = requests
        found = host.find("lab")
        assert found["id"] == "dep-3" and deploy.type_of(found) == "dedicated"
        assert host.find("la") is None
        assert deploy.tier("serverless", "m") == "SERVERLESS_M"

    def test_the_closing_hint_names_the_deployments_url(self, requests):
        host, _ = requests
        assert deploy.deployment_url(host, "lab") == "https://lab-123.us.langgraph.app"
        assert deploy.deployment_url(host, "lab-x").startswith("<")

    def test_a_refused_tier_raises_with_the_platforms_reason(self, requests):
        host, _ = requests
        with pytest.raises(RuntimeError, match=r"422.*tier not allowed"):
            host.set_tier("dep-1", "SERVERLESS_L")


def test_sandbox_client_prefers_the_sandbox_key(monkeypatch):
    from deep_life_sci import sandbox

    client = Mock()
    monkeypatch.setattr(sandbox, "SandboxClient", client)
    sandbox.make_client(timeout=5)
    client.assert_called_with(api_key=None, timeout=5)
    monkeypatch.setenv("LANGSMITH_SANDBOX_API_KEY", " lsv2_sandbox ")
    sandbox.make_client()
    client.assert_called_with(api_key="lsv2_sandbox")


def test_thread_sandboxes_carry_the_name_prefix(monkeypatch):
    import langsmith.sandbox

    from deep_life_sci import sandbox

    monkeypatch.setattr(langsmith.sandbox, "SandboxClient", Mock())
    monkeypatch.setattr(sandbox, "SandboxClient", Mock())
    from deep_life_sci import graph

    monkeypatch.setattr(graph, "NAME_PREFIX", "my-lab")
    assert graph._sandbox_name("01a0eec7-8e09") == "my-lab-01a0eec7-8e09"


def test_a_signed_in_owners_sandbox_names_fit_and_differ_by_owner(monkeypatch):
    """Thread ids are client-chosen, and a deleted thread's id can be created again by
    someone else; that must not reach the sandbox the first owner left behind."""
    import langsmith.sandbox

    from deep_life_sci import ownership, sandbox

    monkeypatch.setattr(langsmith.sandbox, "SandboxClient", Mock())
    monkeypatch.setattr(sandbox, "SandboxClient", Mock())
    from deep_life_sci import graph

    monkeypatch.setattr(graph, "NAME_PREFIX", deploy.sandbox_prefix("deep-life-sci-cloud"))
    thread = "0f6b2b0e-3c1a-4c2e-9e57-6a8e5d1b2c3d"

    def scope(caller: str | None, *, signed_in: bool = True, **metadata: str) -> str:
        configurable = {"thread_id": thread}
        if caller:
            configurable["langgraph_auth_user_id"] = caller
            configurable["langgraph_auth_permissions"] = [ownership.USER] if signed_in else []
        return ownership.thread_scope({"configurable": configurable, "metadata": metadata})

    alice, bob = graph._sandbox_name(scope("alice")), graph._sandbox_name(scope("bob"))
    assert len({alice, bob, graph._sandbox_name(scope(None))}) == 3
    assert len(alice) <= 63 and "alice" not in alice
    # A workspace member running alice's thread keeps to alice's sandbox; a signed-in user
    # cannot claim it by naming her as owner in the run's metadata.
    member = scope("studio-user", signed_in=False, owner="alice")
    assert graph._sandbox_name(member) == alice
    assert graph._sandbox_name(scope("bob", owner="alice")) == bob
