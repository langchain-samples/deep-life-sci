"""`scripts/setup.py`'s choice of how model calls authenticate, and the models.yaml switch
that follows choosing a provider key the file does not use."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from unittest.mock import Mock

import pytest

from deep_life_sci import models, paths

sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))

import _common

# Loaded under another name: `import setup` could find setuptools' convention instead.
_spec = importlib.util.spec_from_file_location(
    "setup_script", paths.REPO_ROOT / "scripts" / "setup.py"
)
setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=lsv2_x\nMODEL_ACCESS=\nANTHROPIC_API_KEY=\n")
    monkeypatch.setattr(_common, "ENV_FILE", env)
    monkeypatch.setattr(setup, "ENV_FILE", env)
    return env


@pytest.fixture
def answers(monkeypatch):
    """Answer setup's prompts in order, as someone at a terminal would."""
    def use(*replies: str) -> None:
        queue = list(replies)
        monkeypatch.setattr(setup, "interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: queue.pop(0))
        monkeypatch.setattr(setup.getpass, "getpass", lambda _prompt: queue.pop(0))

    return use


class TestModelAccess:
    def test_without_a_terminal_it_is_the_gateway(self, env_file, monkeypatch):
        monkeypatch.setattr(setup, "interactive", lambda: False)
        assert setup.ensure_model_access() == ()
        assert _common.env_value("MODEL_ACCESS") == "gateway"

    def test_enter_takes_the_gateway(self, env_file, answers):
        answers("")
        assert setup.ensure_model_access() == ()
        assert _common.env_value("MODEL_ACCESS") == "gateway"

    def test_choosing_a_provider_asks_for_its_key(self, env_file, answers):
        answers("2", "sk-ant-abc")
        assert setup.ensure_model_access() == ("anthropic",)
        assert _common.env_value("MODEL_ACCESS") == "direct"
        assert _common.env_value("ANTHROPIC_API_KEY") == "sk-ant-abc"

    def test_a_key_already_in_the_shell_is_adopted(self, env_file, answers, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        answers("3", "")
        assert setup.ensure_model_access() == ("openai",)
        assert _common.env_value("OPENAI_API_KEY") == "sk-shell"

    def test_an_answered_choice_is_not_asked_again(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\nANTHROPIC_API_KEY=sk-ant-abc\n")
        answers()  # any prompt would pop from an empty queue
        assert setup.ensure_model_access() == ("anthropic",)

    def test_bedrock_asks_for_its_key_and_a_region(self, env_file, answers):
        answers("4", "ABSKbedrock", "")
        assert setup.ensure_model_access() == ("bedrock",)
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == "ABSKbedrock"
        assert _common.env_value("AWS_REGION") == setup.BEDROCK_REGIONS[0]

    def test_bedrock_takes_a_signed_in_aws_profile_instead(self, env_file, answers, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "research")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        # Enter to each: yes to the profile, then to the shell's region.
        answers("4", "", "")
        assert setup.ensure_model_access() == ("bedrock",)
        assert _common.env_value("AWS_PROFILE") == "research"
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == ""
        assert _common.env_value("AWS_REGION") == "us-west-2"

    def test_a_region_outside_the_list_is_refused_and_asked_again(self, env_file, answers):
        answers("4", "ABSKbedrock", "us-west-1", "us-west-2")
        setup.ensure_model_access()
        assert _common.env_value("AWS_REGION") == "us-west-2"

    def test_an_unsupported_region_in_the_shell_is_not_adopted(
        self, env_file, answers, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "us-west-1")
        answers("4", "ABSKbedrock", "")
        setup.ensure_model_access()
        assert _common.env_value("AWS_REGION") == setup.BEDROCK_REGIONS[0]

    def test_the_region_list_is_models_own(self):
        assert setup.BEDROCK_REGIONS == models.BEDROCK_REGIONS

    def test_a_setup_stopped_at_the_key_prompt_offers_the_whole_menu_again(
        self, env_file, answers, monkeypatch
    ):
        answers("2")
        monkeypatch.setattr(setup, "ask_key", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            setup.ensure_model_access()
        assert _common.env_value("MODEL_ACCESS") == ""

    def test_direct_without_any_key_and_no_terminal_stops(self, env_file, monkeypatch):
        env_file.write_text("MODEL_ACCESS=direct\n")
        monkeypatch.setattr(setup, "interactive", lambda: False)
        with pytest.raises(SystemExit):
            setup.ensure_model_access()


class TestSwitchModels:
    @pytest.fixture
    def models_file(self, tmp_path):
        """A copy of the shipped models.yaml, which is what a user's setup would edit."""
        path = tmp_path / "models.yaml"
        shutil.copy(paths.REPO_ROOT / "models.yaml", path)
        return path

    def test_an_anthropic_key_moves_every_role_to_claude(self, models_file):
        stranded = models.roles_without_key(("anthropic",), models_file)
        setup.switch_models(models_file, "anthropic", stranded)
        defaults, _ = models._load(models_file)
        for role, (model, effort) in models.RECOMMENDED_MODELS["anthropic"].items():
            assert defaults[role]["model"] == model
            assert defaults[role]["effort"] == effort
        assert models.roles_without_key(("anthropic",), models_file) == {}

    def test_comments_survive_and_keep_the_old_model(self, models_file):
        before = models_file.read_text()
        setup.switch_models(models_file, "anthropic", {"root": "openai"})
        after = models_file.read_text()
        assert "  model: claude-sonnet-5" in after
        assert "# OpenAI: openai/gpt-5.6-terra" in after
        # Only the root's lines changed.
        pairs = zip(after.splitlines(), before.splitlines(), strict=True)
        assert sum(a != b for a, b in pairs) == 2

    def test_only_the_stranded_roles_change(self, models_file):
        setup.switch_models(models_file, "anthropic", {"search": "openai"})
        defaults, _ = models._load(models_file)
        assert defaults["search"]["model"] == "claude-haiku-4-5-20251001"
        assert defaults["root"]["model"] == "openai/gpt-5.6-terra"

    def test_bedrock_credentials_move_every_role_to_bedrock(self, models_file):
        stranded = models.roles_without_key(("bedrock",), models_file)
        assert stranded == dict.fromkeys(models.ROLES, "openai")
        setup.switch_models(models_file, "bedrock", stranded)
        assert models.roles_without_key(("bedrock",), models_file) == {}
        assert "  model: bedrock/openai.gpt-5.6-terra" in models_file.read_text()

    def test_a_declared_provider_follows_the_model(self, models_file):
        text = models_file.read_text().replace(
            "  effort: high ", "  provider: openai\n  effort: high ", 1
        )
        models_file.write_text(text)
        setup.switch_models(models_file, "anthropic", {"root": "openai"})
        defaults, _ = models._load(models_file)
        assert defaults["root"]["provider"] == "anthropic"
