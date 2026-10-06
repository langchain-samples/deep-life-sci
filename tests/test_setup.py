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

    def test_a_key_in_the_shell_is_not_used(self, env_file, answers, monkeypatch):
        """What .env holds is always what the user typed here."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        answers("3", "sk-typed")
        assert setup.ensure_model_access() == ("openai",)
        assert _common.env_value("OPENAI_API_KEY") == "sk-typed"

    def test_an_answered_choice_is_not_asked_again(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\nANTHROPIC_API_KEY=sk-ant-abc\n")
        answers()  # any prompt would pop from an empty queue
        assert setup.ensure_model_access() == ("anthropic",)

    def test_bedrock_asks_for_its_key_and_a_region(self, env_file, answers):
        answers("4", "ABSKbedrock", "")
        assert setup.ensure_model_access() == ("bedrock",)
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == "ABSKbedrock"
        assert _common.env_value("AWS_REGION") == setup.RECOMMENDED_BEDROCK_REGIONS[0]

    def test_an_aws_profile_in_env_stands_in_for_the_key(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\nAWS_PROFILE=research\nAWS_REGION=us-west-2\n")
        answers()  # nothing left to ask
        assert setup.ensure_model_access() == ("bedrock",)
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == ""

    def test_a_recommended_region_gets_a_note(self, env_file, answers, capsys):
        answers("4", "ABSKbedrock", "us-west-2")
        setup.ensure_model_access()
        assert _common.env_value("AWS_REGION") == "us-west-2"
        out = capsys.readouterr().out
        assert "note: Bedrock serves different models" in out and "WARNING" not in out

    def test_any_other_region_is_kept_with_a_warning(self, env_file, answers, capsys):
        answers("4", "ABSKbedrock", "eu-west-1")
        setup.ensure_model_access()
        assert _common.env_value("AWS_REGION") == "eu-west-1"
        assert "WARNING: the default Bedrock models do not run in eu-west-1" in (
            capsys.readouterr().out
        )

    def test_a_region_in_the_shell_is_not_used(
        self, env_file, answers, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "us-east-2")
        answers("4", "ABSKbedrock", "")
        setup.ensure_model_access()
        assert _common.env_value("AWS_REGION") == setup.RECOMMENDED_BEDROCK_REGIONS[0]

    def test_a_provider_key_is_taken_as_typed(self, env_file, answers, capsys):
        """No prefix check: key formats are the providers' to change."""
        answers("2", "sk-proj-not-anthropic")
        setup.ensure_model_access()
        assert _common.env_value("ANTHROPIC_API_KEY") == "sk-proj-not-anthropic"
        assert "use it anyway" not in capsys.readouterr().out

    def test_the_provider_only_menu_does_not_repeat_your_own(self, env_file, answers, capsys):
        env_file.write_text("MODEL_ACCESS=direct\n")
        answers("1", "sk-ant-abc")
        setup.ensure_model_access()
        menu = capsys.readouterr().out
        assert "which provider" in menu and "your own" not in menu

    def test_ncbi_credentials_say_nothing_without_a_terminal(self, env_file, monkeypatch, capsys):
        monkeypatch.setattr(setup, "interactive", lambda: False)
        setup.ask_ncbi_credentials()
        assert capsys.readouterr().out == ""

    def test_the_recommended_regions_are_models_own(self):
        assert setup.RECOMMENDED_BEDROCK_REGIONS == models.RECOMMENDED_BEDROCK_REGIONS

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
        """A copy of the pinned defaults (tests/conftest.py), not the repository's
        models.yaml: setup itself rewrites that one in a clone set up for Anthropic or
        Bedrock, which would decide these assertions."""
        path = tmp_path / "models.yaml"
        shutil.copy(paths.MODELS_FILE, path)
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
        # The shipped file's shape: a comment above each role, and the alternative trailing.
        before = models_file.read_text().replace(
            "root:\n  model: openai/gpt-5.6-terra\n  effort: high\n",
            "# Main agent\nroot:\n  model: openai/gpt-5.6-terra  # Anthropic: claude-sonnet-5\n"
            "  effort: high                 # Anthropic: high\n",
        )
        models_file.write_text(before)
        setup.switch_models(models_file, "anthropic", {"root": "openai"})
        after = models_file.read_text()
        assert "# Main agent\nroot:\n" in after
        assert "  model: claude-sonnet-5       # OpenAI: openai/gpt-5.6-terra" in after
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
            "  effort: high\n", "  provider: openai\n  effort: high\n", 1
        )
        models_file.write_text(text)
        setup.switch_models(models_file, "anthropic", {"root": "openai"})
        defaults, _ = models._load(models_file)
        assert defaults["root"]["provider"] == "anthropic"
