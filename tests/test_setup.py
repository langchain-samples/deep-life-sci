"""`scripts/setup.py`'s one question about models: which models file to run (MODELS_FILE),
and so how model calls are made, with the credentials that file needs."""

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


@pytest.fixture(autouse=True)
def repo(tmp_path, monkeypatch):
    """A repository root for setup to read: the shipped models files and .env.example, with
    the pinned gateway file (tests/conftest.py) rather than one a developer may have changed."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in models.MODEL_FILES.values():
        shutil.copy(paths.REPO_ROOT / name, root / name)
    shutil.copy(paths.MODELS_FILE, root / models.MODEL_FILES["gateway"])
    shutil.copy(paths.REPO_ROOT / ".env.example", root / ".env.example")
    monkeypatch.setattr(setup, "REPO_ROOT", root)
    return root


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=lsv2_x\nMODELS_FILE=\nANTHROPIC_API_KEY=\n")
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


class TestModelsFile:
    def test_without_a_terminal_it_is_the_gateway(self, env_file, monkeypatch):
        monkeypatch.setattr(setup, "interactive", lambda: False)
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.gateway.yaml"

    def test_enter_takes_the_gateway(self, env_file, answers):
        answers("")
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.gateway.yaml"

    def test_choosing_a_provider_runs_its_file_and_asks_for_its_key(self, env_file, answers):
        answers("2", "sk-ant-abc")
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.anthropic.yaml"
        assert _common.env_value("ANTHROPIC_API_KEY") == "sk-ant-abc"

    def test_a_key_in_the_shell_is_not_used(self, env_file, answers, monkeypatch):
        """What .env holds is always what the user typed here."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        answers("3", "sk-typed")
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.openai.yaml"
        assert _common.env_value("OPENAI_API_KEY") == "sk-typed"

    def test_an_answered_choice_is_not_asked_again(self, env_file, answers):
        env_file.write_text("MODELS_FILE=models.anthropic.yaml\nANTHROPIC_API_KEY=sk-ant-abc\n")
        answers()  # any prompt would pop from an empty queue
        setup.ensure_models_file()

    def test_another_file_later_asks_for_the_key_it_needs(self, env_file, answers):
        """How to switch: set MODELS_FILE and run setup again."""
        env_file.write_text("MODELS_FILE=models.bedrock.yaml\nANTHROPIC_API_KEY=sk-ant-abc\n")
        answers("ABSKbedrock", "")
        setup.ensure_models_file()
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == "ABSKbedrock"
        assert _common.env_value("AWS_REGION") == setup.RECOMMENDED_BEDROCK_REGIONS[0]

    def test_a_file_without_its_key_and_no_terminal_stops(self, env_file, monkeypatch):
        env_file.write_text("MODELS_FILE=models.openai.yaml\n")
        monkeypatch.setattr(setup, "interactive", lambda: False)
        with pytest.raises(SystemExit):
            setup.ensure_models_file()

    def test_bedrock_asks_for_its_key_and_a_region(self, env_file, answers):
        answers("4", "ABSKbedrock", "")
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.bedrock.yaml"
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == "ABSKbedrock"
        assert _common.env_value("AWS_REGION") == setup.RECOMMENDED_BEDROCK_REGIONS[0]

    def test_an_aws_profile_in_env_stands_in_for_the_key(self, env_file, answers):
        env_file.write_text("MODELS_FILE=models.bedrock.yaml\nAWS_PROFILE=research\n"
                            "AWS_REGION=us-west-2\n")
        answers()  # nothing left to ask
        setup.ensure_models_file()
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == ""

    def test_aws_access_keys_in_env_stand_in_for_the_key(self, env_file, monkeypatch):
        """The server accepts them, so setup must neither refuse them nor ask for a key."""
        env_file.write_text("MODELS_FILE=models.bedrock.yaml\nAWS_ACCESS_KEY_ID=AKIAEXAMPLE\n"
                            "AWS_SECRET_ACCESS_KEY=secret\nAWS_REGION=us-west-2\n")
        monkeypatch.setattr(setup, "interactive", lambda: False)
        setup.ensure_models_file()
        assert _common.env_value("AWS_BEARER_TOKEN_BEDROCK") == ""

    def test_aws_default_region_is_a_region(self, env_file, monkeypatch):
        """AWS_REGION written beside it would win, and silently move the user's region."""
        env_file.write_text("MODELS_FILE=models.bedrock.yaml\nAWS_BEARER_TOKEN_BEDROCK=k\n"
                            "AWS_DEFAULT_REGION=us-west-2\n")
        monkeypatch.setattr(setup, "interactive", lambda: False)
        setup.ensure_models_file()
        assert _common.env_value("AWS_REGION") == ""

    def test_setup_and_the_server_count_credentials_alike(self, env_file):
        """Setup reads .env by the server's own rule, so neither accepts what the other
        refuses."""
        cases = [
            {"AWS_ACCESS_KEY_ID": "AKIA"},  # half a pair is nothing
            {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s"},
            {"AWS_PROFILE": "research"},
            {"OPENAI_API_KEY": "sk-x"},
        ]
        for values in cases:
            env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
            for destination in models.DIRECT_KEYS:
                assert setup.has_own_credentials(destination) == (
                    models.has_direct_credentials(destination, values.get)
                ), (values, destination)
        assert {d: names[0] for d, names in setup.OWN_KEYS.items()} == models.DIRECT_KEYS

    def test_every_answer_is_a_shipped_file_that_says_how_it_calls(self):
        """The menu, the files and what each needs, read from the repository as shipped."""
        for choice, _label in setup.ACCESS_CHOICES:
            access, destinations = models.file_needs(
                paths.REPO_ROOT / models.MODEL_FILES[choice]
            )
            assert (access, destinations) == (
                ("gateway", ()) if choice == "gateway" else ("direct", (choice,))
            ), choice

    def test_a_recommended_region_gets_a_note(self, env_file, answers, capsys):
        answers("4", "ABSKbedrock", "us-west-2")
        setup.ensure_models_file()
        assert _common.env_value("AWS_REGION") == "us-west-2"
        out = capsys.readouterr().out
        assert "note: Bedrock serves different models" in out and "WARNING" not in out

    def test_any_other_region_is_kept_with_a_warning(self, env_file, answers, capsys):
        answers("4", "ABSKbedrock", "eu-west-1")
        setup.ensure_models_file()
        assert _common.env_value("AWS_REGION") == "eu-west-1"
        assert "WARNING: the default Bedrock models do not run in eu-west-1" in (
            capsys.readouterr().out
        )

    def test_a_region_in_the_shell_is_not_used(
        self, env_file, answers, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "us-east-2")
        answers("4", "ABSKbedrock", "")
        setup.ensure_models_file()
        assert _common.env_value("AWS_REGION") == setup.RECOMMENDED_BEDROCK_REGIONS[0]

    def test_a_provider_key_is_taken_as_typed(self, env_file, answers, capsys):
        """No prefix check: key formats are the providers' to change."""
        answers("2", "sk-proj-not-anthropic")
        setup.ensure_models_file()
        assert _common.env_value("ANTHROPIC_API_KEY") == "sk-proj-not-anthropic"
        assert "not 'sk-ant-'" not in capsys.readouterr().out

    def test_the_recommended_regions_are_models_own(self):
        assert setup.RECOMMENDED_BEDROCK_REGIONS == models.RECOMMENDED_BEDROCK_REGIONS

    def test_a_setup_stopped_at_the_key_prompt_offers_the_whole_menu_again(
        self, env_file, answers, monkeypatch
    ):
        answers("2")
        monkeypatch.setattr(setup, "ask_key", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == ""


class TestModelAccessMigration:
    """An .env from an earlier draft of this setup: MODEL_ACCESS, and the old gateway file name."""

    def test_direct_runs_the_file_for_the_key_env_holds(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\nOPENAI_API_KEY=sk-x\n")
        answers()
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.openai.yaml"
        assert "MODEL_ACCESS" not in env_file.read_text()

    def test_gateway_runs_the_gateway_file(self, env_file, answers):
        env_file.write_text("# kept\nMODEL_ACCESS=gateway\n")
        answers()
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.gateway.yaml"
        assert env_file.read_text().startswith("# kept\n")

    def test_a_models_file_already_named_wins(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\nMODELS_FILE=models.anthropic.yaml\n"
                            "ANTHROPIC_API_KEY=sk-ant-x\nOPENAI_API_KEY=sk-x\n")
        answers()
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.anthropic.yaml"
        assert "MODEL_ACCESS" not in env_file.read_text()

    def test_direct_without_a_key_asks_the_question_again(self, env_file, answers):
        env_file.write_text("MODEL_ACCESS=direct\n")
        answers("2", "sk-ant-abc")
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.anthropic.yaml"

    def test_the_gateway_files_old_name_becomes_its_new_one(self, env_file, repo, answers):
        env_file.write_text("MODELS_FILE=models.yaml\n")
        answers()
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.gateway.yaml"
        # Unless the user has a models.yaml of their own.
        env_file.write_text("MODELS_FILE=models.yaml\n")
        (repo / "models.yaml").write_text((repo / "models.gateway.yaml").read_text())
        setup.ensure_models_file()
        assert _common.env_value("MODELS_FILE") == "models.yaml"


class TestFirstRun:
    @pytest.fixture
    def fresh(self, tmp_path, monkeypatch):
        """A clone with no .env yet, and the NCBI prompts recorded rather than asked."""
        env = tmp_path / ".env"
        monkeypatch.setattr(_common, "ENV_FILE", env)
        monkeypatch.setattr(setup, "ENV_FILE", env)
        asked = []
        monkeypatch.setattr(setup, "ask_optional", lambda key, *_a, **_k: asked.append(key))
        return asked

    def test_a_first_run_stopped_partway_still_asks_for_ncbi_next_time(
        self, fresh, answers, monkeypatch
    ):
        """Stopping at the model menu, to go and find a key, is ordinary."""
        menus, choose = [], setup.choose

        def stopped_once(*args):
            menus.append(args[0])
            if len(menus) == 1:
                raise KeyboardInterrupt
            return choose(*args)

        monkeypatch.setattr(setup, "choose", stopped_once)
        answers("lsv2_x")
        with pytest.raises(KeyboardInterrupt):
            setup.ensure_env()
        assert fresh == []
        answers("")  # the menu again; Enter takes the gateway
        setup.ensure_env()
        assert len(menus) == 2
        assert fresh == ["NCBI_API_KEY", "NCBI_EMAIL"]

    def test_a_later_run_does_not_ask_again(self, fresh, answers):
        answers("lsv2_x", "")
        setup.ensure_env()
        assert fresh == ["NCBI_API_KEY", "NCBI_EMAIL"]
        answers()
        setup.ensure_env()
        assert fresh == ["NCBI_API_KEY", "NCBI_EMAIL"]

    def test_a_value_already_in_env_is_not_asked_for(self, fresh, env_file, answers):
        """A clone from before the model question meets it once, and the NCBI ones with it."""
        env_file.write_text("LANGSMITH_API_KEY=lsv2_x\nNCBI_API_KEY=abc\n")
        answers("")
        setup.ensure_env()
        assert fresh == ["NCBI_EMAIL"]

    def test_ncbi_credentials_say_nothing_without_a_terminal(self, env_file, monkeypatch, capsys):
        monkeypatch.setattr(setup, "interactive", lambda: False)
        setup.ask_ncbi_credentials()
        assert capsys.readouterr().out == ""


class TestGatewayKeyMigration:
    def test_an_old_gateway_key_is_renamed_whichever_models_file_runs(self, env_file):
        """Under models.openai.yaml it would otherwise be sent to OpenAI as the user's key."""
        env_file.write_text("LANGSMITH_API_KEY=lsv2_x\nMODELS_FILE=models.openai.yaml\n"
                            "OPENAI_API_KEY=lsv2_old\n")
        setup.migrate_gateway_key()
        assert _common.env_value("LANGSMITH_GATEWAY_API_KEY") == "lsv2_old"
        assert _common.env_value("OPENAI_API_KEY") == ""

    def test_the_old_placeholder_is_renamed_too(self, env_file):
        env_file.write_text("LANGSMITH_API_KEY=lsv2_x\nOPENAI_API_KEY=lsv2_...\n")
        setup.migrate_gateway_key()
        assert "OPENAI_API_KEY" not in env_file.read_text()

    def test_the_users_own_openai_key_is_never_renamed(self, env_file):
        env_file.write_text("LANGSMITH_API_KEY=lsv2_x\nOPENAI_API_KEY=sk-proj-mine\n")
        setup.migrate_gateway_key()
        assert _common.env_value("OPENAI_API_KEY") == "sk-proj-mine"
        assert _common.env_value("LANGSMITH_GATEWAY_API_KEY") == ""
