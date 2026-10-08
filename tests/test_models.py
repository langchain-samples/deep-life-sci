"""`models.py`: the four roles x three axes, and the rules that keep a swap honest.

Every assertion here is about configuration resolution, not about calling a model. The
two paths are not cosmetically different — an Anthropic id sent down the OpenAI-compatible
path silently loses prompt caching, and a prefixed id sent down the native path returns a
501 that reads like an outage — so `_provider_for` raising on a contradiction is a feature
and is tested as one.

The subtlest behaviour in the module, and the one most likely to be broken by a
well-intentioned edit: a **default** provider describes the default model it sits beside,
so it must not survive that model being replaced. Without that, `ROOT_MODEL=claude-sonnet-5`
alone would contradict a models file's `provider: openai` and refuse to run.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from langchain_core.messages import HumanMessage

from deep_life_sci import models
from deep_life_sci.models import (
    _BEDROCK_WEB_SEARCH_SPEC,
    ANTHROPIC_BASE_URL,
    DEFAULTS,
    ENV_VARS,
    LABELS,
    PROVIDERS,
    ROLES,
    ROOT_TIMEOUT,
    WEB_SEARCH_SPECS,
    WebSearchUnavailable,
    _bedrock_parts,
    _effort,
    _infer_provider,
    _load,
    _messages_base_url,
    _provider_for,
    _resolve,
    _setting,
    _web_search_spec,
    check_model_access,
    describe,
    file_needs,
    gateway_key,
    refresh,
    rejection_message,
    report_web_search_problem,
    root_model,
    slug,
    summary,
    validate,
    web_search_model,
    web_search_problem,
)


class TestInferProvider:
    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-haiku-4-5-20251001"])
    def test_a_bare_claude_id_is_the_native_path(self, model: str):
        assert _infer_provider(model) == "anthropic"

    @pytest.mark.parametrize("model", ["openai/gpt-5.6-terra", "anthropic/claude-sonnet-5"])
    def test_a_prefixed_id_is_the_openai_compatible_path(self, model: str):
        assert _infer_provider(model) == "openai"

    def test_an_id_in_neither_form_says_nothing(self):
        assert _infer_provider("some-new-model") == ""


class TestWebSearchSpec:
    def test_openai_search_uses_only_responses_api_tool_keys(self, monkeypatch):
        monkeypatch.setenv("SEARCH_MODEL", "openai/search-model")
        monkeypatch.setenv("SEARCH_EFFORT", "")
        assert "name" not in WEB_SEARCH_SPECS["openai"]
        assert "max_uses" not in WEB_SEARCH_SPECS["openai"]

        class CapturingModel:
            def bind_tools(self, tools):
                self.tools = tools
                return self

        model = CapturingModel()
        monkeypatch.setattr(models, "_build", lambda *args, **kwargs: model)
        web_search_model()

        assert set(model.tools[0]) <= {
            "type",
            "search_context_size",
            "user_location",
            "filters",
        }


class TestProviderFor:
    def test_a_declared_path_that_agrees_with_the_form_is_kept(self):
        assert _provider_for("root", "claude-sonnet-5", "anthropic") == "anthropic"

    def test_the_form_decides_when_nothing_is_declared(self):
        assert _provider_for("root", "openai/gpt-5.6-terra", "") == "openai"

    def test_a_declared_path_is_the_escape_hatch_for_an_unknown_form(self):
        """A model in neither known form is usable without editing this module."""
        assert _provider_for("root", "some-new-model", "openai") == "openai"

    def test_a_contradiction_is_an_error_rather_than_a_preference(self):
        """Sending an id down the wrong path 501s, or silently drops prompt caching."""
        with pytest.raises(SystemExit, match="is a 'anthropic' id but"):
            _provider_for("root", "claude-sonnet-5", "openai")

    def test_a_path_that_is_not_a_gateway_path_is_refused(self):
        with pytest.raises(SystemExit, match="is not a gateway path"):
            _provider_for("root", "claude-sonnet-5", "bedrock")

    def test_an_unknown_form_with_no_declared_path_is_refused_legibly(self):
        with pytest.raises(SystemExit, match="Cannot tell which gateway path"):
            _provider_for("root", "some-new-model", "")


class TestSetting:
    def test_falls_back_to_the_module_default(self):
        assert _setting("root", "model") == DEFAULTS["root"]["model"]

    def test_an_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        assert _setting("root", "model") == "claude-sonnet-5"

    def test_whitespace_reads_as_unset_rather_than_as_an_empty_id(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "   ")
        assert _setting("root", "model") == DEFAULTS["root"]["model"]


class TestEffort:
    def test_defaults_to_the_roles_constant(self):
        assert _effort("root") == DEFAULTS["root"]["effort"]

    @pytest.mark.parametrize("level", ["none", "minimal", "xhigh", "max"])
    def test_reads_the_level_as_set(self, level: str, monkeypatch):
        """Checking it against the model is `_resolve`'s job (`TestEffortCheck`)."""
        monkeypatch.setenv("SUBAGENT_EFFORT", level)
        assert _effort("subagent") == level

    def test_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("ROOT_EFFORT", "HIGH")
        assert _effort("root") == "high"

    def test_an_explicitly_empty_effort_is_unset(self, monkeypatch):
        """`SUBAGENT_EFFORT=` is how Haiku 4.5, which has no effort scale, is used."""
        monkeypatch.setenv("SUBAGENT_EFFORT", "")
        assert _effort("subagent") == ""


class TestResolve:
    def test_the_defaults_resolve_without_any_environment(self):
        model, provider, effort = _resolve("root")
        default = DEFAULTS["root"]
        assert (model, provider, effort) == (
            default["model"],
            # The models file may leave the path to the id's form.
            default["provider"] or _infer_provider(default["model"]),
            default["effort"],
        )

    def test_a_model_swap_alone_moves_the_path_with_it(self, monkeypatch):
        """The default provider describes the default model, so it must not survive it."""
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        assert _resolve("root") == ("claude-sonnet-5", "anthropic", DEFAULTS["root"]["effort"])

    def test_an_explicit_provider_still_wins_over_inference(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MODEL", "some-new-model")
        monkeypatch.setenv("SUBAGENT_PROVIDER", "anthropic")
        assert _resolve("subagent")[1] == "anthropic"

    def test_a_provider_that_contradicts_an_explicit_model_still_raises(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("ROOT_PROVIDER", "openai")
        with pytest.raises(SystemExit):
            _resolve("root")

    def test_a_provider_is_read_case_insensitively(self, monkeypatch):
        monkeypatch.setenv("ROOT_PROVIDER", "OpenAI")
        assert _resolve("root")[1] == "openai"

    @pytest.mark.parametrize("role", sorted(DEFAULTS))
    def test_every_role_resolves_to_a_real_gateway_path(self, role: str):
        assert _resolve(role)[1] in PROVIDERS

    def test_the_four_roles_are_independent(self, monkeypatch):
        """One run mixing providers across roles is the point of three axes."""
        monkeypatch.setenv("SUBAGENT_MODEL", "claude-haiku-4-5-20251001")
        monkeypatch.setenv("SUBAGENT_EFFORT", "")
        assert _resolve("subagent") == ("claude-haiku-4-5-20251001", "anthropic", "")
        assert _resolve("root")[1] == (
            DEFAULTS["root"]["provider"] or _infer_provider(DEFAULTS["root"]["model"])
        )


class TestEnvVars:
    def test_names_every_role_crossed_with_every_axis(self):
        expected = {
            f"{role.upper()}_{axis.upper()}"
            for role in DEFAULTS
            for axis in ("model", "provider", "effort")
        }
        assert set(ENV_VARS) == expected

    @pytest.mark.parametrize("name", sorted(ENV_VARS))
    def test_each_name_is_one_resolve_actually_reads(self, name: str, monkeypatch):
        """A drifting hand-copied list is how a CLI override silently loses to .env."""
        role, axis = name.rsplit("_", 1)
        role, axis = role.lower(), axis.lower()
        if axis == "model":
            monkeypatch.setenv(name, "openai/probe-model")
            assert _resolve(role)[0] == "openai/probe-model"
        elif axis == "provider":
            monkeypatch.setenv(f"{role.upper()}_MODEL", "some-new-model")
            monkeypatch.setenv(name, "anthropic")
            assert _resolve(role)[1] == "anthropic"
        else:
            monkeypatch.setenv(name, "high")
            assert _resolve(role)[2] == "high"


class TestGatewayKey:
    def test_falls_back_to_the_main_langsmith_key(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_main")
        assert gateway_key() == "lsv2_main"

    def test_the_override_wins_when_it_is_set(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_main")
        monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "lsv2_gateway")
        assert gateway_key() == "lsv2_gateway"

    def test_an_empty_override_falls_through_rather_than_authenticating_as_empty(
        self, monkeypatch
    ):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_main")
        monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "   ")
        assert gateway_key() == "lsv2_main"

    def test_no_key_at_all_is_the_empty_string(self):
        assert gateway_key() == ""


class TestCheckGatewayConfig:
    def test_fails_legibly_rather_than_deep_inside_the_sdk(self):
        with pytest.raises(SystemExit, match="LANGSMITH_API_KEY is not set"):
            check_model_access()

    def test_says_a_provider_key_is_not_what_it_wants(self):
        """The commonest mistake: pasting an sk-... here gets a 403 at the gateway."""
        with pytest.raises(SystemExit, match="provider credential"):
            check_model_access()

    def test_passes_once_a_key_is_present(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_x")
        check_model_access()


class TestDirectAccess:
    """A models file with `access: direct`: the user's own provider keys, not the gateway."""

    def test_gateway_is_the_default_and_ignores_provider_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        with pytest.raises(SystemExit, match="LANGSMITH_API_KEY is not set"):
            check_model_access()

    def test_an_unknown_access_is_refused(self, access):
        access("byok")
        with pytest.raises(SystemExit, match="access is 'byok'"):
            check_model_access()

    def test_needs_no_langsmith_key_for_models(self, monkeypatch, direct):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        check_model_access()

    def test_a_role_whose_provider_has_no_key_is_named(self, monkeypatch, direct):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        with pytest.raises(SystemExit, match=r"root model.*OPENAI_API_KEY is not set"):
            check_model_access()
        for role in ("ROOT", "SUBAGENT", "SEARCH"):
            monkeypatch.setenv(f"{role}_MODEL", "claude-sonnet-5")
            monkeypatch.setenv(f"{role}_EFFORT", "")
        check_model_access()
        # The judge only when asked for, as evals do.
        with pytest.raises(SystemExit, match="judge model"):
            check_model_access(*ROLES)

    @pytest.mark.parametrize("model", ["xai/grok-5", "anthropic/claude-sonnet-5"])
    def test_ids_only_the_gateway_reaches_are_refused(self, monkeypatch, direct, model):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setenv("ROOT_MODEL", model)
        monkeypatch.setenv("ROOT_EFFORT", "")
        with pytest.raises(SystemExit, match="cannot run with your own provider keys"):
            _resolve("root")

    def test_a_file_names_the_credentials_it_needs(self, access, monkeypatch):
        """For setup and deploy, from the file alone: a shell override is not shipped."""
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        assert file_needs(access("gateway")) == ("gateway", ())
        assert file_needs(access("direct")) == ("direct", ("openai",))

    def test_rejections_name_the_provider_rather_than_the_gateway(self, access):
        error = Exception("bad request")
        error.status_code = 400
        assert rejection_message("root", error).startswith("The LLM Gateway returned")
        access("direct")
        assert rejection_message("root", error).startswith("The model provider returned")

    @pytest.mark.parametrize("destination", ["anthropic", "openai", "bedrock"])
    def test_each_providers_file_runs_on_its_own_credentials_alone(
        self, monkeypatch, destination
    ):
        """The files setup points MODELS_FILE at, read from the repository as shipped."""
        from deep_life_sci.models import DIRECT_KEYS, MODEL_FILES, _destination
        from deep_life_sci.paths import REPO_ROOT

        path = REPO_ROOT / MODEL_FILES[destination]
        _use_models_file(monkeypatch, path)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv(DIRECT_KEYS[destination], "k")
        assert file_needs(path) == ("direct", (destination,))
        for role in ROLES:
            model, provider, _ = _resolve(role)
            assert _destination(model, provider) == destination
            assert models.LABELS.get(model), f"{model} has no label for the badge"
        # And the search role can search there.
        assert web_search_problem() is None

    def test_the_openai_file_runs_the_gateways_models(self):
        """The same models as models.gateway.yaml, called with the user's own key: one file of
        defaults to change, and this test to say when the other was forgotten."""
        from deep_life_sci.models import MODEL_FILES
        from deep_life_sci.paths import REPO_ROOT

        gateway, direct = (_load(REPO_ROOT / MODEL_FILES[name]) for name in ("gateway", "openai"))
        assert (gateway.access, direct.access) == ("gateway", "direct")
        assert gateway.defaults == direct.defaults

    @pytest.mark.parametrize("model", ["bedrock/us.anthropic.claude-sonnet-5",
                                       "bedrock/openai.gpt-5.6-luna"])
    def test_bedrock_needs_credentials_and_a_region(self, monkeypatch, direct, model):
        monkeypatch.setenv("ROOT_MODEL", model)
        monkeypatch.setenv("ROOT_EFFORT", "")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
        with pytest.raises(SystemExit, match="AWS_REGION is not set"):
            _resolve("root")
        # Any region: which models it serves is the user's to check (models.bedrock.yaml's notes).
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        _resolve("root")
        # AWS access keys stand in for the key, both halves of them.
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        with pytest.raises(SystemExit, match="none of AWS_BEARER_TOKEN_BEDROCK, AWS_PROFILE"):
            _resolve("root")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        _resolve("root")

    def test_aws_default_region_is_a_region(self, monkeypatch, direct):
        monkeypatch.setenv("ROOT_MODEL", "bedrock/openai.gpt-5.6-luna")
        monkeypatch.setenv("ROOT_EFFORT", "")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        _resolve("root")

    def test_a_signed_in_profile_serves_bedrocks_gpt_but_not_its_claude(self, monkeypatch, direct):
        """Claude's SDK would resolve the profile on the server's event loop, which the dev
        server refuses; the GPT models' SDK mints its key off the loop."""
        monkeypatch.setenv("AWS_PROFILE", "research")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("ROOT_EFFORT", "")
        monkeypatch.setenv("ROOT_MODEL", "bedrock/openai.gpt-5.6-terra")
        _resolve("root")
        monkeypatch.setenv("ROOT_MODEL", "bedrock/us.anthropic.claude-sonnet-5")
        with pytest.raises(SystemExit, match="cannot sign in with AWS_PROFILE"):
            _resolve("root")
        # The key, or AWS access keys, beside the profile are what it signs in with.
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        _resolve("root")

    def test_claude_on_bedrock_refuses_an_empty_key(self, monkeypatch, direct):
        """Its SDK sends an empty AWS_BEARER_TOKEN_BEDROCK as `Bearer `, never the profile."""
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("ROOT_EFFORT", "")
        monkeypatch.setenv("ROOT_MODEL", "bedrock/us.anthropic.claude-sonnet-5")
        with pytest.raises(SystemExit, match="empty AWS_BEARER_TOKEN_BEDROCK"):
            _resolve("root")
        # The GPT models' SDK skips an empty one.
        monkeypatch.setenv("ROOT_MODEL", "bedrock/openai.gpt-5.6-terra")
        _resolve("root")

    def test_credentials_read_through_a_reader_of_setups_choosing(self):
        """`scripts/setup.py` checks .env with the server's own rule."""
        from deep_life_sci.models import aws_region, has_direct_credentials

        env = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_DEFAULT_REGION": " us-west-2 "}.get
        assert not has_direct_credentials("bedrock", env)  # half a pair
        env = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s"}.get
        assert has_direct_credentials("bedrock", env)
        assert not has_direct_credentials("openai", env)
        assert aws_region({"AWS_DEFAULT_REGION": " us-west-2 "}.get) == "us-west-2"
        assert aws_region({"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "x"}.get) == (
            "eu-west-1"
        )

    def test_aws_credential_failures_name_the_fix(self, monkeypatch):
        """An expired SSO session raises from botocore, with no status code to report."""
        from botocore.exceptions import ProfileNotFound, UnauthorizedSSOTokenError

        monkeypatch.setenv("AWS_PROFILE", "research")
        message = rejection_message("root", UnauthorizedSSOTokenError())
        assert "AWS could not supply credentials for the root model's request" in message
        assert "aws sso login --profile research" in message
        assert "AWS_BEARER_TOKEN_BEDROCK" in rejection_message(
            "subagent", ProfileNotFound(profile="research")
        )

    def test_whether_a_model_takes_a_system_message_midway(self):
        """UpdateCadence's reminder: Claude on Bedrock refuses one in place, though the bare
        id of the same model takes it."""
        from langchain_anthropic import ChatAnthropic
        from langchain_aws import ChatAnthropicBedrock
        from langchain_openai import ChatOpenAI

        from deep_life_sci.models import takes_system_midway

        assert takes_system_midway(ChatOpenAI(model="gpt-5.6-terra", api_key="k"))
        assert takes_system_midway(ChatAnthropic(model="claude-sonnet-5-5", api_key="k"))
        assert not takes_system_midway(ChatAnthropic(model="claude-sonnet-5", api_key="k"))
        assert not takes_system_midway(
            ChatAnthropicBedrock(model="us.anthropic.claude-sonnet-5-5-v1:0",
                                 region_name="us-east-1")
        )


class TestWebSearchSpecs:
    def test_there_is_a_spec_for_every_gateway_path(self):
        """A missing one is a KeyError inside `web_search_model`, at run time."""
        assert set(WEB_SEARCH_SPECS) == set(PROVIDERS)

    def test_bedrock_search_stays_inside_the_aws_boundary(self):
        """Left at its default, every fetch fails without an extra IAM permission."""
        assert _BEDROCK_WEB_SEARCH_SPEC == {"type": "web_search", "external_web_access": False}

    def test_the_anthropic_spec_caps_searches_per_request(self):
        assert WEB_SEARCH_SPECS["anthropic"]["max_uses"] == 5

    def test_each_spec_is_a_dict_the_binder_can_use(self):
        for spec in WEB_SEARCH_SPECS.values():
            assert isinstance(spec, dict) and spec.get("type")


class TestDescribe:
    def test_defaults_to_the_pair_that_does_the_work(self):
        line = describe()
        assert line.startswith("root=")
        assert "subagent=" in line

    def test_prints_the_path_because_it_decides_whether_caching_works(self):
        assert f"({_resolve('root')[1]}" in describe("root")

    def test_an_unset_effort_is_omitted_rather_than_printed_empty(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_EFFORT", "")
        assert describe("subagent").endswith(")")
        assert ", )" not in describe("subagent")


class TestSlug:
    def test_is_the_root_model_and_its_effort(self):
        model = DEFAULTS["root"]["model"].split("/")[-1]
        assert slug() == f"{model}-{DEFAULTS['root']['effort']}"

    def test_strips_the_provider_prefix(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "openai/gpt-5.6-luna")
        monkeypatch.setenv("ROOT_EFFORT", "low")
        assert slug() == "gpt-5.6-luna-low"

    def test_an_effortless_root_slugs_to_the_model_alone(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "claude-haiku-4-5-20251001")
        monkeypatch.setenv("ROOT_EFFORT", "")
        assert slug() == "claude-haiku-4-5-20251001"


VALID_YAML = """
root: {model: openai/gpt-5.6-terra, effort: high}
subagent: {model: claude-haiku-4-5-20251001, effort:}
search: {model: openai/gpt-5.6-luna, provider: OpenAI, effort: low}
judge: {model: openai/gpt-5.6-terra}
labels: {openai/gpt-5.6-terra: GPT-5.6 Terra}
"""


class TestLoadModelsYaml:
    """A models file is what a user edits, so a mistake in it must fail loudly at startup."""

    def _load(self, tmp_path, text: str):
        path = tmp_path / "models.yaml"
        path.write_text(text, encoding="utf-8")
        return _load(path)

    def test_the_shipped_file_is_well_formed_and_every_role_can_run(self, monkeypatch):
        """The one test that reads the repository's own gateway file, whatever it now says."""
        from deep_life_sci import models
        from deep_life_sci.paths import REPO_ROOT

        _use_models_file(monkeypatch, REPO_ROOT / "models.gateway.yaml")
        assert set(models.DEFAULTS) == set(ROLES)
        validate()

    def test_the_tests_run_against_the_pinned_copy(self):
        from deep_life_sci import paths

        assert paths.MODELS_FILE.parent.name == "fixtures"
        assert set(DEFAULTS) == set(ROLES)

    def test_reads_each_axis_and_the_labels(self, tmp_path):
        defaults, labels, access = self._load(tmp_path, VALID_YAML)
        assert defaults["root"] == {
            "model": "openai/gpt-5.6-terra", "provider": "", "effort": "high"
        }
        assert defaults["search"]["provider"] == "openai"
        assert labels == {"openai/gpt-5.6-terra": "GPT-5.6 Terra"}
        assert access == "gateway"  # left out, as an older file would

    def test_access_says_how_calls_are_made(self, tmp_path):
        assert self._load(tmp_path, "access: direct\n" + VALID_YAML).access == "direct"
        with pytest.raises(SystemExit, match=r"access is 'byok'; write `gateway`"):
            self._load(tmp_path, "access: byok\n" + VALID_YAML)

    def test_an_empty_or_absent_effort_is_none_rather_than_a_default(self, tmp_path):
        """Empty is a value on this axis: Haiku 4.5 400s on any effort at all."""
        defaults = self._load(tmp_path, VALID_YAML).defaults
        assert defaults["subagent"]["effort"] == ""
        assert defaults["judge"]["effort"] == ""

    def test_an_unquoted_yaml_boolean_is_refused_rather_than_read_as_none(self, tmp_path):
        """`effort: off` parses as False; reading that as "none" would hide a typo."""
        with pytest.raises(SystemExit, match=r"subagent\.effort is False"):
            self._load(tmp_path, VALID_YAML.replace("effort:}", "effort: off}"))

    def test_a_missing_role_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="`judge` needs a `model`"):
            self._load(tmp_path, VALID_YAML.replace("judge:", "# judge:"))

    def test_a_misspelt_role_is_refused_rather_than_ignored(self, tmp_path):
        with pytest.raises(SystemExit, match=r"unknown key.*subagents"):
            self._load(tmp_path, VALID_YAML + "subagents: {model: x}\n")

    def test_a_misspelt_axis_is_refused_rather_than_ignored(self, tmp_path):
        with pytest.raises(SystemExit, match=r"root has unknown key.*efort"):
            self._load(tmp_path, VALID_YAML.replace("effort: high", "efort: high"))

    def test_a_missing_file_says_what_it_is_for(self, tmp_path):
        with pytest.raises(SystemExit, match="names the model each role runs"):
            _load(tmp_path / "models.yaml")

    def test_a_repeated_role_is_refused_rather_than_last_one_wins(self, tmp_path):
        with pytest.raises(SystemExit, match="found `root` twice"):
            self._load(tmp_path, VALID_YAML + "root: {model: claude-sonnet-5}\n")

    def test_a_repeated_axis_is_refused_rather_than_last_one_wins(self, tmp_path):
        with pytest.raises(SystemExit, match="found `effort` twice"):
            self._load(tmp_path, VALID_YAML.replace("effort: high", "effort: high, effort: low"))

    def test_invalid_yaml_is_refused_legibly(self, tmp_path):
        with pytest.raises(SystemExit, match="is not valid YAML"):
            self._load(tmp_path, "root: [unclosed")


class TestSummary:
    """What the chat UI's model badge shows: the resolved roles, not the file."""

    def test_reports_what_this_process_runs_including_env_overrides(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("ROOT_EFFORT", "medium")
        (root,) = summary("root")
        assert root == {
            "role": "root",
            "model": "claude-sonnet-5",
            "label": LABELS.get("claude-sonnet-5", "claude-sonnet-5"),
            "provider": "anthropic",
            "effort": "medium",
        }

    def test_a_model_without_a_label_shows_its_id(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MODEL", "openai/some-unlabelled-model")
        assert summary("subagent")[0]["label"] == "openai/some-unlabelled-model"

    def test_defaults_to_every_role_in_order(self):
        assert [row["role"] for row in summary()] == list(ROLES)

    def test_the_route_serves_the_three_chat_roles_and_not_the_judge(self):
        pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        from deep_life_sci.webapp import app

        response = TestClient(app).get("/models")
        assert response.status_code == 200
        assert [row["role"] for row in response.json()["roles"]] == [
            "root", "subagent", "search"
        ]


class _Refused(Exception):
    """Stands in for either SDK's APIStatusError: all `rejection_message` reads."""

    def __init__(self, status_code: int, message: str = "Unsupported value: 'xhigh'"):
        super().__init__(message)
        self.status_code = status_code


class TestRejectionMessage:
    """A provider's error is shown as the provider worded it, with no guess at the cause."""

    @pytest.mark.parametrize("status", [400, 404, 422, 429, 500])
    def test_any_status_is_surfaced_with_the_providers_words_and_the_settings(
        self, status: int, monkeypatch
    ):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("ROOT_EFFORT", "high")
        message = rejection_message("root", _Refused(status))
        assert f"({status})" in message
        assert "Unsupported value: 'xhigh'" in message
        assert "'claude-sonnet-5'" in message and "'high'" in message
        assert "valid together" not in message  # no diagnosis: a 400 is often a filter

    def test_an_empty_effort_is_shown_as_none_rather_than_blank(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_EFFORT", "")
        assert "effort '(none)'" in rejection_message("subagent", _Refused(400))

    def test_a_setting_that_fails_its_checks_cannot_raise_in_place_of_the_error(
        self, monkeypatch
    ):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("ROOT_EFFORT", "xhigh")  # refused by `_resolve`
        assert "'xhigh'" in rejection_message("root", _Refused(400))

    @pytest.mark.parametrize("exc", [TimeoutError("slow"), ValueError("our bug")])
    def test_anything_without_a_status_is_left_alone(self, exc: Exception):
        assert rejection_message("root", exc) is None

    def test_a_context_overflow_is_left_for_summarization_to_catch(self):
        """deepagents compacts and retries on `ContextOverflowError`; rewrapping it ends the run."""
        from langchain_core.exceptions import ContextOverflowError

        class _Overflow(_Refused, ContextOverflowError):
            pass

        assert rejection_message("root", _Overflow(400, "context_length_exceeded")) is None


class TestEffortCheck:
    """A level the model cannot take fails in `_resolve`, before anything is built or booted."""

    def test_a_typo_is_caught_before_a_sweep_boots_a_container(self, monkeypatch):
        monkeypatch.setenv("ROOT_EFFORT", "extreme")
        with pytest.raises(SystemExit, match="ROOT_EFFORT='extreme' is not an effort level"):
            describe()

    def test_the_judge_is_checked_too(self, monkeypatch):
        """`evals/run.py` describes the judge before the first example runs."""
        monkeypatch.setenv("JUDGE_EFFORT", "hgih")
        with pytest.raises(SystemExit, match="JUDGE_EFFORT"):
            describe("judge")

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            ("openai/gpt-5.6-terra", "none"),  # the level an allowlist used to refuse
            ("claude-sonnet-5", "xhigh"),
            ("openai/o3", "high"),  # its profile lists no levels, which is not "none"
            ("acme/some-model", "minimal"),  # no profile: only typos are refused
        ],
    )
    def test_a_level_the_model_takes_is_accepted(self, model, effort, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", model)
        monkeypatch.setenv("ROOT_EFFORT", effort)
        assert _resolve("root")[2] == effort

    @pytest.mark.parametrize(
        ("model", "effort", "scope"),
        [
            ("claude-sonnet-4-6", "xhigh", "for 'claude-sonnet-4-6'"),  # from its profile
            ("claude-haiku-4-5-20251001", "none", "on the Anthropic path"),  # ChatAnthropic
            ("claude-some-future-model", "minimal", "on the Anthropic path"),
        ],
    )
    def test_a_level_the_model_cannot_take_is_refused(self, model, effort, scope, monkeypatch):
        monkeypatch.setenv("SUBAGENT_MODEL", model)
        monkeypatch.setenv("SUBAGENT_EFFORT", effort)
        with pytest.raises(SystemExit, match=scope):
            _resolve("subagent")


def _use_models_file(monkeypatch, path):
    """Point models.py at another models file, and forget what it had read."""
    from deep_life_sci import models, paths

    monkeypatch.setattr(paths, "MODELS_FILE", path)
    monkeypatch.setattr(models, "_config_state", None)


class TestModelsFileLifecycle:
    def test_importing_models_does_not_import_paths(self):
        """`cli.py` imports models before `.env`; `paths` fixes DATA_DIR when imported."""
        code = "import sys, deep_life_sci.models; print('deep_life_sci.paths' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.stdout.strip() == "False", out.stderr

    def test_models_file_names_another_file_at_the_root(self):
        """How setup fits models to the user's own keys without editing a models file."""
        code = "from deep_life_sci import paths; print(paths.MODELS_FILE)"
        env = {**os.environ, "MODELS_FILE": " models.bedrock.yaml "}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        from deep_life_sci.paths import REPO_ROOT

        assert out.stdout.strip() == str(REPO_ROOT / "models.bedrock.yaml"), out.stderr
        env["MODELS_FILE"] = ""
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert out.stdout.strip() == str(REPO_ROOT / "models.gateway.yaml"), out.stderr

    def test_errors_name_the_file_in_use(self, tmp_path, monkeypatch):
        path = tmp_path / "models.anthropic.yaml"
        path.write_text(VALID_YAML.replace("openai/gpt-5.6-luna, provider: OpenAI", "gpt-5.6-luna"))
        _use_models_file(monkeypatch, path)
        with pytest.raises(SystemExit, match=r"models\.anthropic\.yaml search\.model="):
            validate("search")

    def test_an_edit_applies_on_the_next_refresh(self, tmp_path, monkeypatch):
        path = tmp_path / "models.yaml"
        path.write_text(VALID_YAML, encoding="utf-8")
        _use_models_file(monkeypatch, path)
        assert _resolve("root")[2] == "high"
        path.write_text(VALID_YAML.replace("effort: high", "effort: medium"), encoding="utf-8")
        os.utime(path, ns=(1, 1))  # a new stamp even on a coarse-mtime filesystem
        refresh()
        assert _resolve("root")[2] == "medium"

    def test_a_yaml_provider_does_not_survive_its_model_being_replaced(
        self, tmp_path, monkeypatch
    ):
        """The shipped file names no provider, so only a file that does exercises this."""
        path = tmp_path / "models.yaml"
        terra = "root: {model: openai/gpt-5.6-terra,"
        path.write_text(VALID_YAML.replace(terra, terra + " provider: openai,"))
        _use_models_file(monkeypatch, path)
        assert _resolve("root")[1] == "openai"
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        assert _resolve("root")[1] == "anthropic"

    def test_a_yaml_value_that_cannot_work_names_the_line_in_the_models_file(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "models.yaml"
        path.write_text(VALID_YAML.replace("openai/gpt-5.6-luna, provider: OpenAI", "gpt-5.6-luna"))
        _use_models_file(monkeypatch, path)
        with pytest.raises(SystemExit, match=r"models\.yaml search\.model='gpt-5\.6-luna'"):
            validate("search")

    def test_the_route_reports_a_setting_that_cannot_work_rather_than_raising(
        self, tmp_path, monkeypatch
    ):
        """A SystemExit inside a request stops `langgraph dev`'s event loop."""
        pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        from deep_life_sci.webapp import app

        path = tmp_path / "models.yaml"
        path.write_text(VALID_YAML.replace("openai/gpt-5.6-luna, provider: OpenAI", "gpt-5.6-luna"))
        _use_models_file(monkeypatch, path)
        response = TestClient(app).get("/models")
        assert response.status_code == 500
        assert "search.model" in response.json()["error"]


class TestAnthropicRoot:
    def test_an_anthropic_root_builds_with_the_read_watchdog(self, monkeypatch):
        """ChatAnthropic refuses an httpx.Timeout; unmocked, because the mocks hid that."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_offline")
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-5")
        model = root_model()
        assert model.default_request_timeout == ROOT_TIMEOUT.read
        assert model.streaming


BEDROCK_CLAUDE = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class TestBedrock:
    """Bedrock through the same gateway and key: Claude on Messages, the rest OpenAI-style."""

    @pytest.mark.parametrize(
        ("model", "parts"),
        [
            (BEDROCK_CLAUDE, ("anthropic", "claude-sonnet-4-5-20250929")),
            ("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
             ("anthropic", "claude-haiku-4-5-20251001")),
            ("bedrock/openai.gpt-5.6-terra", ("openai", "gpt-5.6-terra")),
            ("bedrock/amazon.nova-pro-v1:0", ("", "amazon.nova-pro-v1:0")),
            # Any region prefix, not a list of them, and an ARN by the id it ends in.
            ("bedrock/sa.anthropic.claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
            ("bedrock/amer.openai.gpt-5.6-luna", ("openai", "gpt-5.6-luna")),
            ("bedrock/arn:aws:bedrock:us-east-1:111122223333:inference-profile/"
             "us.anthropic.claude-haiku-4-5-20251001-v1:0",
             ("anthropic", "claude-haiku-4-5-20251001")),
        ],
    )
    def test_an_id_is_read_as_its_maker_and_the_bare_id_its_profile_uses(self, model, parts):
        assert _bedrock_parts(model) == parts

    def test_other_ids_are_not_bedrock(self):
        assert _bedrock_parts("openai/gpt-5.6-terra") is None
        assert _bedrock_parts("claude-sonnet-5") is None

    def test_claude_takes_the_messages_path_and_everything_else_the_openai_one(self):
        assert _infer_provider(BEDROCK_CLAUDE) == "anthropic"
        assert _infer_provider("bedrock/openai.gpt-5.6-terra") == "openai"
        assert _infer_provider("bedrock/amazon.nova-pro-v1:0") == "openai"

    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    def test_an_id_that_names_no_model_takes_the_named_path(self, provider, monkeypatch):
        """An application inference profile ARN: only the provider setting can say."""
        arn = "bedrock/arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/a1b2"
        assert _infer_provider(arn) == ""
        monkeypatch.setenv("ROOT_MODEL", arn)
        monkeypatch.setenv("ROOT_EFFORT", "")
        with pytest.raises(SystemExit, match="Cannot tell which gateway path"):
            _resolve("root")
        monkeypatch.setenv("ROOT_PROVIDER", provider)
        assert _resolve("root")[1] == provider

    def test_claude_is_refused_on_the_openai_path(self, monkeypatch):
        """It would run there, but silently without prompt caching, e.g. under a stale
        ROOT_PROVIDER left from an OpenAI sweep."""
        monkeypatch.setenv("ROOT_MODEL", BEDROCK_CLAUDE)
        monkeypatch.setenv("ROOT_PROVIDER", "openai")
        with pytest.raises(SystemExit, match="Claude on Bedrock, which takes the anthropic"):
            _resolve("root")
        assert _provider_for("root", BEDROCK_CLAUDE, "anthropic") == "anthropic"

    def test_the_search_warning_names_where_the_model_came_from(self, monkeypatch):
        monkeypatch.setitem(models.DEFAULTS["search"], "model", BEDROCK_CLAUDE)
        monkeypatch.setitem(models.DEFAULTS["search"], "effort", "")
        source = f"models.gateway.yaml search.model={BEDROCK_CLAUDE!r}"
        assert web_search_problem().startswith(source)

    def test_claude_on_bedrock_is_built_as_chat_anthropic_on_the_standard_endpoint(
        self, monkeypatch
    ):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_offline")
        monkeypatch.setenv("ROOT_MODEL", BEDROCK_CLAUDE)
        model = root_model()
        assert type(model).__name__ == "ChatAnthropic"
        assert model.model == BEDROCK_CLAUDE
        assert model.anthropic_api_url == "https://gateway.smith.langchain.com"

    def test_the_standard_endpoint_follows_a_regional_gateway(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_GATEWAY_BASE_URL", "https://eu.gateway.smith.langchain.com/v1/")
        assert _messages_base_url(BEDROCK_CLAUDE) == "https://eu.gateway.smith.langchain.com"
        assert _messages_base_url("claude-sonnet-5").endswith("/anthropic")

    def test_effort_is_checked_against_the_model_behind_the_bedrock_id(self, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "bedrock/openai.gpt-5.6-terra")
        monkeypatch.setenv("ROOT_EFFORT", "minimal")  # not a GPT-5.6 level
        with pytest.raises(SystemExit, match=r"for 'bedrock/openai\.gpt-5\.6-terra'"):
            _resolve("root")
        monkeypatch.setenv("ROOT_EFFORT", "none")
        assert _resolve("root")[2] == "none"

    def test_claude_on_bedrock_builds_like_native_claude(self, monkeypatch):
        """The bare id's profile: its output cap, and thinking turned on by effort."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_offline")
        monkeypatch.setenv("ROOT_EFFORT", "high")
        built = {}
        for model in ("claude-sonnet-5", "bedrock/us.anthropic.claude-sonnet-5"):
            monkeypatch.setenv("ROOT_MODEL", model)
            built[model] = root_model()
        native, bedrock = built.values()
        assert bedrock.max_tokens == native.max_tokens > 4096
        assert bedrock.profile == native.profile
        payload = bedrock._get_request_payload([HumanMessage("hi")])
        assert payload["thinking"] == native._get_request_payload([HumanMessage("hi")])["thinking"]

    @pytest.mark.parametrize(
        "model",
        ["bedrock/openai.gpt-5.6-luna", "bedrock/openai.gpt-5.7-terra", "bedrock/openai.gpt-6"],
    )
    def test_a_bedrock_gpt_model_searches_with_bedrocks_own_tool(self, model, monkeypatch):
        monkeypatch.setenv("SEARCH_MODEL", model)
        monkeypatch.setenv("SEARCH_EFFORT", "")
        model, provider, _ = _resolve("search")
        assert _web_search_spec(model, provider) == _BEDROCK_WEB_SEARCH_SPEC

    @pytest.mark.parametrize(
        "model",
        ["bedrock/openai.gpt-5.3", "bedrock/openai.gpt-oss-120b-1:0", "bedrock/openai.gpt-4.1"],
    )
    def test_a_bedrock_gpt_model_before_5_4_cannot_search(self, model, monkeypatch):
        monkeypatch.setenv("SEARCH_MODEL", model)
        monkeypatch.setenv("SEARCH_EFFORT", "")
        assert "cannot be the search model" in web_search_problem()

    def test_an_empty_gateway_url_reads_as_unset(self, monkeypatch):
        """An empty base URL would send the LangSmith key to the provider's own API."""
        monkeypatch.setenv("LANGSMITH_GATEWAY_BASE_URL", " ")
        monkeypatch.setenv("LANGSMITH_GATEWAY_ANTHROPIC_URL", "")
        assert _messages_base_url(BEDROCK_CLAUDE) == "https://gateway.smith.langchain.com"
        assert _messages_base_url("claude-sonnet-5") == ANTHROPIC_BASE_URL

    def test_a_search_problem_is_logged_once_per_reason(self, monkeypatch, capsys):
        monkeypatch.setattr(models, "_reported_search_problem", None)
        monkeypatch.setenv("SEARCH_EFFORT", "")
        for model in (BEDROCK_CLAUDE, BEDROCK_CLAUDE, "bedrock/amazon.nova-pro-v1:0"):
            monkeypatch.setenv("SEARCH_MODEL", model)
            report_web_search_problem()
        monkeypatch.setenv("SEARCH_MODEL", "bedrock/openai.gpt-5.6-luna")
        report_web_search_problem()
        monkeypatch.setenv("SEARCH_MODEL", BEDROCK_CLAUDE)
        report_web_search_problem()
        logged = capsys.readouterr().out.splitlines()
        assert [BEDROCK_CLAUDE in line for line in logged] == [True, False, True]

    @pytest.mark.parametrize("model", [BEDROCK_CLAUDE, "bedrock/amazon.nova-pro-v1:0"])
    def test_a_model_bedrock_cannot_search_with_starts_and_reports_why(
        self, model, monkeypatch
    ):
        """Only web search is lost: startup passes, and the reason is kept for each search."""
        monkeypatch.setenv("SEARCH_MODEL", model)
        monkeypatch.setenv("SEARCH_EFFORT", "")
        validate("search")
        assert "SEARCH_MODEL=" in web_search_problem()
        assert "cannot be the search model" in web_search_problem()
        with pytest.raises(WebSearchUnavailable):
            web_search_model()

    def test_a_search_model_that_can_search_reports_no_problem(self, monkeypatch):
        monkeypatch.setenv("SEARCH_MODEL", "bedrock/openai.gpt-5.6-luna")
        assert web_search_problem() is None

