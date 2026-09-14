"""The cross-file rules `CLAUDE.md` states, which no single module can enforce.

Every rule here is one the repo already documents and that fails *silently* when broken:
a tool the model cannot discover, an upload format registered in three of its four places,
a hand-copied env var list that drifts. None of them is catchable by ruff, by a type
checker, or by any test that stays inside one module.

Two of these read source with `ast` rather than importing and inspecting. That is
deliberate: `build_agent` needs a sandbox backend and a gateway key to run, and the
allowlist is a literal in its body, so reading the literal is both cheaper and closer to
what a reviewer would check by eye.
"""

from __future__ import annotations

import ast
import re

import pytest

from deep_life_sci import paths
from deep_life_sci.middleware.upload_probe import PROBES
from deep_life_sci.middleware.uploads import UPLOAD_KINDS
from deep_life_sci.models import DEFAULTS, ENV_VARS
from deep_life_sci.prompts.system import _TEMPLATE, build_system_prompt

REPO = paths.REPO_ROOT


def _camel(name: str) -> str:
    """`pubmed_search` -> `pubmedSearch`, the form PTC renders a tool into JS as."""
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _ptc_allowlist() -> list[str]:
    """The `ptc=[...]` literal out of `agent.py:build_agent`."""
    tree = ast.parse((REPO / "deep_life_sci" / "agent.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "ptc":
            return [element.value for element in node.value.elts]
    raise AssertionError("no ptc=[...] allowlist found in agent.py")


def _root_tool_names() -> list[str]:
    """The tools handed to `create_deep_agent`, by the name each is imported under."""
    source = (REPO / "deep_life_sci" / "agent.py").read_text(encoding="utf-8")
    block = re.search(r"with_progress\(\[(.*?)\]\)", source, re.S)
    assert block, "no with_progress([...]) tool list found in agent.py"
    return [line.strip().rstrip(",") for line in block.group(1).splitlines() if line.strip()]


class TestEveryPtcToolIsDiscoverable:
    """A tool the prompt does not mention is a tool the model has no way to reach.

    "Adding a tool means adding it to that allowlist and writing a prompt segment for
    it in `prompts/system.py` — the model has no other way to discover it."
    """

    @pytest.mark.parametrize("name", _ptc_allowlist())
    def test_each_allowlisted_tool_appears_in_the_prompt_in_its_js_form(self, name: str):
        assert f"tools.{_camel(name)}" in _TEMPLATE

    def test_the_allowlist_is_not_empty(self):
        assert len(_ptc_allowlist()) > 5

    def test_every_source_tool_bound_to_the_agent_is_also_in_the_allowlist(self):
        """A tool bound but not allowlisted is invisible inside `eval`, where the work is."""
        allowlisted = set(_ptc_allowlist())
        for name in _root_tool_names():
            assert name in allowlisted, f"{name} is bound but not reachable from eval"

    def test_web_search_is_a_ptc_tool_rather_than_a_spec_on_the_root_model(self):
        """Server-side search bound to the root lands every page in root context —
        27.9k input tokens for one question, measured."""
        assert "web_search" in _ptc_allowlist()

    def test_the_sandbox_and_filesystem_tools_are_reachable_from_eval(self):
        assert {"execute", "read_file", "write_file", "edit_file", "ls", "glob"} <= set(
            _ptc_allowlist()
        )


class TestEveryProgressLabelNamesARealTool:
    """`STARTED` is the run's only visible output while it works."""

    def test_every_label_corresponds_to_a_tool_the_agent_actually_has(self):
        from deep_life_sci.middleware.progress import STARTED

        assert set(STARTED) <= set(_ptc_allowlist())

    def test_every_source_tool_has_a_label_rather_than_falling_back_to_its_name(self):
        from deep_life_sci.middleware.progress import STARTED

        unlabelled = [n for n in _root_tool_names() if n not in STARTED and n != "web_search"]
        assert unlabelled == [], f"no progress label for {unlabelled}"


class TestAnUploadFormatLivesInFourPlaces:
    """"An upload format lives in four places": `UPLOAD_KINDS`, a probe branch, a prompt
    segment, and the composer's allowlist in `scripts/setup.py`.
    """

    @pytest.fixture
    def composer_suffixes(self) -> set[str]:
        source = (REPO / "scripts" / "setup.py").read_text(encoding="utf-8")
        block = re.search(r"export const UPLOAD_SUFFIXES = \[(.*?)\];", source, re.S)
        assert block, "no UPLOAD_SUFFIXES list found in scripts/setup.py"
        return set(re.findall(r'"(\.[a-z0-9.]+)"', block.group(1)))

    @pytest.mark.parametrize("suffix", sorted(UPLOAD_KINDS))
    def test_every_accepted_suffix_is_offered_by_the_composer(
        self, suffix: str, composer_suffixes: set[str]
    ):
        """Otherwise the agent has a reader for a file the UI will not let the user attach."""
        assert suffix in composer_suffixes

    def test_the_composer_offers_nothing_the_sandbox_cannot_read(
        self, composer_suffixes: set[str]
    ):
        """A `.gz` is the exception: it is stripped before the reader is chosen."""
        unreadable = {
            s for s in composer_suffixes if not s.endswith(".gz") and s not in UPLOAD_KINDS
        }
        assert unreadable == set()

    def test_every_gz_entry_the_composer_offers_strips_back_to_a_real_reader(
        self, composer_suffixes: set[str]
    ):
        """`.gz` has no entry of its own; the host strips it to pick the reader."""
        from deep_life_sci.middleware.uploads import _suffix

        gzipped = {s for s in composer_suffixes if s.endswith(".gz")}
        assert gzipped, "the composer should accept gzipped tables"
        for entry in gzipped:
            assert _suffix(f"upload{entry}") in UPLOAD_KINDS

    @pytest.mark.parametrize("kind", sorted(set(UPLOAD_KINDS.values())))
    def test_every_kind_has_a_probe_branch(self, kind: str):
        assert kind in PROBES

    def test_no_probe_branch_is_unreachable(self):
        assert set(PROBES) == set(UPLOAD_KINDS.values())

    @pytest.mark.parametrize("kind", sorted(set(UPLOAD_KINDS.values())))
    def test_every_kind_is_mentioned_in_the_root_prompt(self, kind: str):
        """The prompt is how the agent knows what a manifest line means."""
        assert kind in _TEMPLATE

    def test_the_declined_formats_are_declined_and_not_merely_absent(self):
        """A `.xls` left in reaches a provider with no such type and 400s the whole run."""
        from deep_life_sci.middleware.uploads import _REJECTED

        assert set(_REJECTED) & {".xls"}
        assert not set(_REJECTED) & set(UPLOAD_KINDS)


class TestOneAxisOnePlace:
    """"A new axis is preserved by adding it *there and nowhere else*" — `models.ENV_VARS`."""

    def test_the_two_callers_import_the_list_rather_than_copying_it(self):
        """A copy that drifts is how a `ROOT_MODEL=...` on the command line loses to .env."""
        for relative in ("deep_life_sci/cli.py", "evals/run.py"):
            source = (REPO / relative).read_text(encoding="utf-8")
            assert "from deep_life_sci.models import ENV_VARS" in source, relative

    @pytest.mark.parametrize("relative", ["deep_life_sci/cli.py", "evals/run.py"])
    def test_neither_caller_hard_codes_a_role_axis_name(self, relative: str):
        source = (REPO / relative).read_text(encoding="utf-8")
        for role in DEFAULTS:
            for axis in ("MODEL", "PROVIDER", "EFFORT"):
                assert f'"{role.upper()}_{axis}"' not in source, relative

    def test_every_documented_env_var_is_in_the_list(self):
        assert len(ENV_VARS) == len(DEFAULTS) * 3


class TestPathsAreAnchoredToTheRepoRoot:
    """"Getting that anchor wrong silently starts a second empty cache instead of failing." """

    def test_the_repo_root_is_the_directory_holding_pyproject_toml(self):
        assert (paths.REPO_ROOT / "pyproject.toml").is_file()
        assert (paths.REPO_ROOT / "deep_life_sci" / "__init__.py").is_file()

    def test_every_cache_root_lives_under_the_data_directory(self):
        for root in paths.CACHE_ROOTS:
            assert paths.DATA_DIR in root.parents

    def test_the_sweep_scope_names_every_cache_rather_than_walking_data_dir(self):
        """DEEP_LIFE_SCI_DATA_DIR can point anywhere; a walk would delete what is there."""
        assert set(paths.CACHE_ROOTS) == {
            paths.ABSTRACT_CACHE,
            paths.PMC_CACHE,
            paths.CTGOV_CACHE,
        }

    def test_the_uploads_directory_is_not_under_the_deliverables_directory(self):
        """Or a file the user gave us comes back as a deliverable of their own question."""
        assert not paths.UPLOAD_DIR.startswith(paths.OUT_DIR + "/")
        assert paths.UPLOAD_DIR != paths.OUT_DIR

    def test_the_derived_sidecars_live_under_the_uploads_directory(self):
        """`uploads.py` inventories by filename; a subdirectory falls out of its isfile check."""
        assert paths.UPLOAD_DERIVED_DIR.startswith(paths.UPLOAD_DIR + "/")

    @pytest.mark.parametrize(
        "name", ["WORKSPACE", "OUT_DIR", "UPLOAD_DIR", "UPLOAD_DERIVED_DIR"]
    )
    def test_each_sandbox_path_is_mirrored_in_the_prompt(self, name: str):
        """The agent only knows where to write because the prompt says so."""
        assert getattr(paths, name) in _TEMPLATE

    def test_the_cache_and_the_sandbox_share_one_idle_window(self):
        """Past it a returning thread finds neither its container nor its cache."""
        from deep_life_sci.sources import cache_io

        assert cache_io.ttl_seconds() == float(paths.IDLE_TTL_SECONDS)

    def test_a_stopped_sandbox_outlives_the_idle_window_by_a_working_day_either_side(self):
        assert paths.DELETE_AFTER_STOP_SECONDS > paths.IDLE_TTL_SECONDS
        assert paths.DELETE_AFTER_STOP_SECONDS == 48 * 60 * 60


class TestTheSystemPromptIsBuiltPerRun:
    def test_todays_date_is_substituted(self):
        from datetime import date

        assert date.today().isoformat() in build_system_prompt()

    def test_an_explicit_date_wins_so_a_dev_server_can_outlive_midnight(self):
        from datetime import date

        assert "2026-01-02" in build_system_prompt(date(2026, 1, 2))

    def test_no_placeholder_survives_into_the_prompt(self):
        assert "{{TODAY}}" not in build_system_prompt()

    def test_the_template_is_substituted_rather_than_formatted(self):
        """The body is full of JS object literals; `.format()` reads every `{...}` as a field."""
        assert "{" in _TEMPLATE, "the prompt contains JS literals"
        build_system_prompt()  # would raise on a stray brace if `.format()` were used

    def test_the_date_has_a_days_granularity_so_the_cached_prefix_is_stable(self):
        from datetime import date

        assert build_system_prompt(date(2026, 1, 2)) == build_system_prompt(date(2026, 1, 2))


class TestThePromptKeepsThePayloadRulesItEnforces:
    """The prompt is production code — "one line ... cut root context from 115k to 31k"."""

    def test_the_triage_step_before_any_full_text_call_is_named(self):
        assert "pmcLocate" in _TEMPLATE
        assert "fetchFullText" in _TEMPLATE

    def test_reading_a_deliverable_back_is_forbidden(self):
        """Reading a PNG back can cost more context than an entire run."""
        assert paths.OUT_DIR in _TEMPLATE
        assert "readFile" in _TEMPLATE

    def test_subagents_are_told_not_to_fetch_for_themselves(self):
        """NCBI allows 3 req/sec; N subagents each fetching would collect 429s."""
        assert "subagent" in _TEMPLATE.lower()


class TestThePackageShipsNoTestFramework:
    """"Nothing shipped at deploy time should carry a test framework." """

    def test_the_declared_packages_are_the_runtime_ones_only(self):
        import tomllib

        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(config["tool"]["setuptools"]["packages"])
        assert all(name.startswith("deep_life_sci") for name in declared)
        assert not {"tests", "evals", "scripts", "ui"} & declared

    def test_pytest_is_confined_to_the_optional_groups(self):
        import tomllib

        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert not any("pytest" in dep for dep in config["project"]["dependencies"])
        assert any("pytest" in dep for dep in config["dependency-groups"]["test"])

    def test_the_unit_suite_is_on_the_default_test_path(self):
        import tomllib

        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        assert "tests" in config["tool"]["pytest"]["ini_options"]["testpaths"]

    @pytest.mark.parametrize("relative", ["tests", "evals"])
    def test_neither_suite_is_importable_from_the_package(self, relative: str):
        """`deep_life_sci` must not reach into either — they measure it, they aren't it."""
        for path in (REPO / "deep_life_sci").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert f"import {relative}" not in source, path
