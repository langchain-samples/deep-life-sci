"""`models.py`: the four roles x three axes, and the rules that keep a swap honest.

Every assertion here is about configuration resolution, not about calling a model. The
two paths are not cosmetically different — an Anthropic id sent down the OpenAI-compatible
path silently loses prompt caching, and a prefixed id sent down the native path returns a
501 that reads like an outage — so `_provider_for` raising on a contradiction is a feature
and is tested as one.

The subtlest behaviour in the module, and the one most likely to be broken by a
well-intentioned edit: a **default** provider describes the default model it sits beside,
so it must not survive that model being replaced. Without that, `ROOT_MODEL=claude-sonnet-5`
alone would contradict the default `ROOT_PROVIDER=openai` and refuse to run.
"""

from __future__ import annotations

import pytest

from deep_life_sci.models import (
    DEFAULTS,
    ENV_VARS,
    LABELS,
    PROVIDERS,
    ROLES,
    WEB_SEARCH_SPECS,
    _effort,
    _infer_provider,
    _load,
    _provider_for,
    _resolve,
    _setting,
    check_gateway_config,
    describe,
    gateway_key,
    rejection_message,
    slug,
    summary,
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
    def test_any_level_is_left_for_the_provider_to_judge(self, level: str, monkeypatch):
        """Levels differ by model, so a local list would refuse valid ones (`none`)."""
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
            # models.yaml may leave the path to the id's form.
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
            check_gateway_config()

    def test_says_a_provider_key_is_not_what_it_wants(self):
        """The commonest mistake: pasting an sk-... here gets a 403 at the gateway."""
        with pytest.raises(SystemExit, match="provider credential"):
            check_gateway_config()

    def test_passes_once_a_key_is_present(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_x")
        check_gateway_config()


class TestWebSearchSpecs:
    def test_there_is_a_spec_for_every_gateway_path(self):
        """A missing one is a KeyError inside `web_search_model`, at run time."""
        assert set(WEB_SEARCH_SPECS) == set(PROVIDERS)

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
    """models.yaml is what a user edits, so a mistake in it must fail loudly at startup."""

    def _load(self, tmp_path, text: str):
        path = tmp_path / "models.yaml"
        path.write_text(text, encoding="utf-8")
        return _load(path)

    def test_the_shipped_file_names_every_role(self):
        assert set(DEFAULTS) == set(ROLES)
        assert all(DEFAULTS[role]["model"] for role in ROLES)

    def test_reads_each_axis_and_the_labels(self, tmp_path):
        defaults, labels = self._load(tmp_path, VALID_YAML)
        assert defaults["root"] == {
            "model": "openai/gpt-5.6-terra", "provider": "", "effort": "high"
        }
        assert defaults["search"]["provider"] == "openai"
        assert labels == {"openai/gpt-5.6-terra": "GPT-5.6 Terra"}

    def test_an_empty_or_absent_effort_is_none_rather_than_a_default(self, tmp_path):
        """Empty is a value on this axis: Haiku 4.5 400s on any effort at all."""
        defaults, _ = self._load(tmp_path, VALID_YAML)
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
    """A refused call must say that model and effort have to be valid together."""

    @pytest.mark.parametrize("status", [400, 404, 422])
    def test_names_the_role_setting_and_both_axes(self, status: int, monkeypatch):
        monkeypatch.setenv("ROOT_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("ROOT_EFFORT", "xhigh")
        message = rejection_message("root", _Refused(status))
        assert f"({status})" in message
        assert "Unsupported value: 'xhigh'" in message  # the provider's own words first
        assert "'claude-sonnet-4-6'" in message and "'xhigh'" in message
        assert "models.yaml" in message and "ROOT_EFFORT" in message
        assert "valid together" in message

    def test_an_empty_effort_is_shown_as_none_rather_than_blank(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_EFFORT", "")
        assert "effort '(none)'" in rejection_message("subagent", _Refused(400))

    @pytest.mark.parametrize("exc", [_Refused(500), _Refused(429), TimeoutError("slow")])
    def test_anything_but_a_refusal_is_left_alone(self, exc: Exception):
        assert rejection_message("root", exc) is None
