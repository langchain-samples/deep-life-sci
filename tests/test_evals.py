"""`evals/`: the scoring conventions, which are where a sweep quietly lies.

Three rules from `evals/README.md` are the ones worth defending, and all three are about
what a *number* means rather than about whether an evaluator runs:

* **All three evaluators are boolean.** Every fraction they could return was misleading —
  0.95 on `citations_exist` sorts next to a clean run while reading as a rounding error,
  and one invented citation in twenty is the whole point of the check.
* **`score: None` means not applicable, not `False`.** A question with no required
  artifact is excluded from that evaluator's aggregate rather than given a free pass.
* **An errored run is scored `None` by all three.** Without the guard, a batch that died
  on infrastructure reads as a quality regression — or, worse, as an improvement, because
  one evaluator dropped the dead run from its own denominator.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evals import dataset_name, dataset_prefix
from evals.evaluators import DEFAULT, STRUCTURAL
from evals.evaluators._guard import scores_only_completed_runs
from evals.evaluators.citations import citations_exist, cited_pmids
from evals.evaluators.deliverables import _KINDS, produced_expected_artifacts
from evals.evaluators.judge import _as_bool
from evals.sync import DATASETS_DIR, _load, _split


def _run(**outputs) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs)


def _example(**outputs) -> SimpleNamespace:
    return SimpleNamespace(inputs={"question": "Q"}, outputs=outputs)


class TestDatasetName:
    def test_defaults_to_the_repos_prefix(self):
        assert dataset_name() == "deep-life-sci-default"

    def test_the_prefix_is_an_env_override_rather_than_a_code_edit(self, monkeypatch):
        """Demo hygiene in a shared workspace asks for a `-<username>` suffix."""
        monkeypatch.setenv("EVALS_DATASET_PREFIX", "deep-life-sci-mc")
        assert dataset_name() == "deep-life-sci-mc-default"

    def test_a_whitespace_prefix_reads_as_unset(self, monkeypatch):
        monkeypatch.setenv("EVALS_DATASET_PREFIX", "   ")
        assert dataset_prefix() == "deep-life-sci"

    def test_is_resolved_per_call_because_run_py_loads_dotenv_late(self, monkeypatch):
        assert dataset_name() == "deep-life-sci-default"
        monkeypatch.setenv("EVALS_DATASET_PREFIX", "later")
        assert dataset_name() == "later-default"

    def test_names_one_dataset_per_seed_file_by_its_stem(self):
        assert dataset_name("smoke") == "deep-life-sci-smoke"


class TestCitedPmids:
    def test_finds_a_labelled_pmid(self):
        assert cited_pmids("see PMID: 33567185") == {"33567185"}

    def test_finds_a_bare_seven_or_eight_digit_number(self):
        assert cited_pmids("Wilding et al. (33567185)") == {"33567185"}

    def test_a_bare_year_is_not_a_citation(self):
        """A 4-digit run in prose is almost always a year."""
        assert cited_pmids("published in 2021") == set()

    def test_a_bare_sample_size_is_not_a_citation(self):
        assert cited_pmids("n = 1961 participants") == set()

    def test_a_labelled_short_number_is_still_read_as_a_citation(self):
        """The explicit label is what makes it unambiguous."""
        assert cited_pmids("PMID 12345") == {"12345"}

    def test_finds_every_citation_in_a_paragraph(self):
        text = "STEP 1 (PMID: 33567185) and STEP 2 (33667417) both reported."
        assert cited_pmids(text) == {"33567185", "33667417"}

    def test_an_answer_citing_nothing_yields_nothing(self):
        assert cited_pmids("No papers were found.") == set()


class TestCitationsExist:
    def test_passes_when_every_cited_pmid_is_in_the_fetch_cache(self, abstract_cache,
                                                                monkeypatch):
        monkeypatch.setattr(
            "evals.evaluators.citations.ABSTRACT_CACHE", abstract_cache
        )
        (abstract_cache / "33567185.json").write_text("{}")
        result = citations_exist(_run(answer="see PMID: 33567185"), _example())
        assert result["score"] is True

    def test_fails_when_any_cited_pmid_is_absent(self, abstract_cache, monkeypatch):
        """One invented citation in twenty is the failure this exists to catch."""
        monkeypatch.setattr(
            "evals.evaluators.citations.ABSTRACT_CACHE", abstract_cache
        )
        (abstract_cache / "33567185.json").write_text("{}")
        result = citations_exist(
            _run(answer="PMID: 33567185 and PMID: 99999999"), _example()
        )
        assert result["score"] is False
        assert "99999999" in result["comment"]

    def test_an_answer_citing_nothing_is_not_applicable_rather_than_failing(self):
        """The metadata-only questions are answerable without citing a paper."""
        result = citations_exist(_run(answer="Nature publishes the most."), _example())
        assert result["score"] is None

    def test_the_verdict_is_a_bool_not_a_fraction(self, abstract_cache, monkeypatch):
        """0.95 sorts next to a clean run while reading as a rounding error."""
        monkeypatch.setattr(
            "evals.evaluators.citations.ABSTRACT_CACHE", abstract_cache
        )
        result = citations_exist(_run(answer="PMID: 99999999"), _example())
        assert isinstance(result["score"], bool)

    def test_the_comment_carries_what_the_fraction_used_to(self, abstract_cache,
                                                           monkeypatch):
        monkeypatch.setattr(
            "evals.evaluators.citations.ABSTRACT_CACHE", abstract_cache
        )
        result = citations_exist(_run(answer="PMID: 99999999"), _example())
        assert "0/1 cited PMIDs" in result["comment"]

    def test_the_feedback_key_is_stable(self):
        assert citations_exist(_run(answer=""), _example())["key"] == "citations_exist"


class TestProducedExpectedArtifacts:
    def test_passes_when_the_expected_kind_was_published(self):
        result = produced_expected_artifacts(
            _run(artifact_names=["chart"]), _example(expects_artifact=["image"])
        )
        assert result["score"] is True

    def test_fails_when_a_plot_question_was_answered_in_prose(self):
        """The bytes never appear in the answer text; this is the only place it is visible."""
        result = produced_expected_artifacts(
            _run(artifact_names=[]), _example(expects_artifact=["image"])
        )
        assert result["score"] is False
        assert "missing ['image']" in result["comment"]

    def test_a_question_mandating_nothing_is_not_applicable(self):
        """Returning None keeps it out of the aggregate instead of a free 1.0."""
        result = produced_expected_artifacts(_run(artifact_names=[]), _example())
        assert result["score"] is None

    @pytest.mark.parametrize(
        ("kind", "name"), [("image", "chart"), ("image", "image"), ("table", "table"),
                           ("file", "download")]
    )
    def test_each_kind_accepts_the_component_names_the_middleware_assigns(
        self, kind: str, name: str
    ):
        result = produced_expected_artifacts(
            _run(artifact_names=[name]), _example(expects_artifact=[kind])
        )
        assert result["score"] is True

    def test_component_names_are_matched_case_insensitively(self):
        result = produced_expected_artifacts(
            _run(artifact_names=["Chart"]), _example(expects_artifact=["image"])
        )
        assert result["score"] is True

    def test_a_missing_kind_fails_even_when_another_was_published(self):
        result = produced_expected_artifacts(
            _run(artifact_names=["chart"]), _example(expects_artifact=["image", "table"])
        )
        assert result["score"] is False

    def test_the_kinds_table_covers_what_the_middleware_can_assign(self):
        """`_component_for` returns exactly these three."""
        from deep_life_sci.middleware.artifacts import _component_for

        assigned = {_component_for(s) for s in (".png", ".csv", ".md")}
        assert assigned <= set().union(*_KINDS.values())


class TestAsBool:
    @pytest.mark.parametrize("value", [True, False])
    def test_a_real_bool_passes_through(self, value: bool):
        assert _as_bool(value) is value

    @pytest.mark.parametrize(("text", "expected"),
                             [("true", True), ("FALSE", False), (" True ", True)])
    def test_the_two_string_spellings_are_accepted(self, text: str, expected: bool):
        assert _as_bool(text) is expected

    @pytest.mark.parametrize("value", [0.75, 1, 0, None, "maybe", [], {}])
    def test_a_number_is_not_a_verdict(self, value):
        """Reading 0.75 as True restores the split-the-difference grading this avoids."""
        with pytest.raises(ValueError, match="not a boolean verdict"):
            _as_bool(value)


class TestErroredRunGuard:
    """A run that died is not a run that scored badly."""

    def test_a_failed_run_is_unscoreable_rather_than_false(self):
        result = citations_exist(_run(answer="", error="sandbox boot timeout"), _example())
        assert result["score"] is None
        assert "sandbox boot timeout" in result["comment"]

    def test_the_guard_covers_the_artifact_evaluator_too(self):
        """Without it, a dead run reads as a missing deliverable."""
        result = produced_expected_artifacts(
            _run(answer="", error="boom"), _example(expects_artifact=["image"])
        )
        assert result["score"] is None

    def test_a_dead_run_does_not_land_in_the_not_applicable_bucket_by_accident(self):
        """That was the worst of the three: it raised the aggregate."""
        result = produced_expected_artifacts(_run(answer="", error="boom"), _example())
        assert "run failed" in result["comment"]

    def test_the_guard_files_the_skip_under_the_key_it_was_given(self):
        """`rubric_judge` writes under `rubric`; guessing from `__name__` would misfile it."""

        @scores_only_completed_runs("rubric")
        def evaluator(run, example):  # pragma: no cover - never reached
            raise AssertionError("short-circuited")

        assert evaluator(_run(error="boom"), _example())["key"] == "rubric"

    async def test_an_async_evaluator_is_wrapped_too(self):
        @scores_only_completed_runs("rubric")
        async def evaluator(run, example):  # pragma: no cover - never reached
            raise AssertionError("short-circuited")

        assert (await evaluator(_run(error="boom"), _example()))["score"] is None

    async def test_a_healthy_run_reaches_the_async_evaluator(self):
        @scores_only_completed_runs("rubric")
        async def evaluator(run, example):
            return {"key": "rubric", "score": True}

        assert (await evaluator(_run(answer="fine"), _example()))["score"] is True

    def test_the_signature_stays_visible_to_the_sdks_introspection(self):
        """`functools.wraps` sets `__wrapped__`, which is what keeps `(run, example)` legible."""
        import inspect

        assert list(inspect.signature(citations_exist).parameters) == ["run", "example"]


class TestEvaluatorRegistry:
    def test_the_structural_evaluators_cost_no_model_call(self):
        assert len(STRUCTURAL) == 2
        assert all(e.__name__ != "rubric_judge" for e in STRUCTURAL)

    def test_the_judge_is_listed_last_so_it_can_be_dropped(self):
        assert DEFAULT[: len(STRUCTURAL)] == STRUCTURAL
        assert DEFAULT[-1].__name__ == "rubric_judge"


class TestSeedLoader:
    def test_accepts_a_minimal_example(self, tmp_path: Path):
        path = tmp_path / "s.yaml"
        path.write_text("- id: a\n  question: Why?\n")
        assert _load(path) == [{"id": "a", "question": "Why?"}]

    def test_an_empty_file_is_no_examples_rather_than_an_error(self, tmp_path: Path):
        path = tmp_path / "s.yaml"
        path.write_text("")
        assert _load(path) == []

    def test_a_non_list_top_level_is_refused(self, tmp_path: Path):
        path = tmp_path / "s.yaml"
        path.write_text("id: a\n")
        with pytest.raises(SystemExit, match="expected a top-level list"):
            _load(path)

    def test_an_example_missing_an_id_is_refused_by_number(self, tmp_path: Path):
        """A partial sync leaves LangSmith holding some of the edit."""
        path = tmp_path / "s.yaml"
        path.write_text("- id: a\n  question: Q\n- question: no id\n")
        with pytest.raises(SystemExit, match="example 2 needs an 'id'"):
            _load(path)

    def test_duplicate_ids_are_refused_by_name(self, tmp_path: Path):
        """Matching is by `id`; duplicates would overwrite each other in LangSmith."""
        path = tmp_path / "s.yaml"
        path.write_text("- id: a\n  question: Q\n- id: a\n  question: R\n")
        with pytest.raises(SystemExit, match=re.escape("duplicate ids ['a']")):
            _load(path)

    def test_malformed_yaml_names_the_file(self, tmp_path: Path):
        path = tmp_path / "s.yaml"
        path.write_text("- id: [unclosed\n")
        with pytest.raises(SystemExit, match=re.escape("s.yaml")):
            _load(path)


class TestSplit:
    def test_question_is_the_only_input(self):
        inputs, _, _ = _split({"id": "a", "question": "Why?"})
        assert inputs == {"question": "Why?"}

    def test_the_references_the_evaluators_read_become_outputs(self):
        _, outputs, _ = _split(
            {"id": "a", "question": "Q", "expects_artifact": ["image"], "rubric": "R"}
        )
        assert outputs == {"expects_artifact": ["image"], "rubric": "R"}

    def test_absent_references_default_rather_than_raising(self):
        _, outputs, _ = _split({"id": "a", "question": "Q"})
        assert outputs == {"expects_artifact": [], "rubric": ""}

    def test_the_seed_id_is_what_makes_a_re_sync_an_update(self):
        _, _, metadata = _split({"id": "semaglutide-weightloss-boxplot", "question": "Q"})
        assert metadata["seed_id"] == "semaglutide-weightloss-boxplot"

    def test_provenance_fields_are_metadata_and_never_scored(self):
        _, outputs, metadata = _split(
            {"id": "a", "question": "Q", "domain": "endocrinology", "surface": "pubmed"}
        )
        assert metadata["domain"] == "endocrinology"
        assert metadata["surface"] == "pubmed"
        assert "domain" not in outputs


class TestTheCheckedInSeedFiles:
    """The YAML is the source of truth and is hand-edited, so it is validated here."""

    @pytest.fixture(params=sorted(DATASETS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
    def seed_file(self, request) -> Path:
        return request.param

    def test_there_is_at_least_one(self):
        assert list(DATASETS_DIR.glob("*.yaml"))

    def test_it_loads(self, seed_file: Path):
        assert _load(seed_file)

    def test_every_example_splits_into_the_three_parts(self, seed_file: Path):
        for row in _load(seed_file):
            inputs, _outputs, metadata = _split(row)
            assert inputs["question"].strip()
            assert metadata["seed_id"]

    def test_every_expected_artifact_is_a_kind_the_evaluator_knows(self, seed_file: Path):
        """`_KINDS.get(kind, {kind})` means a typo silently matches nothing, forever."""
        for row in _load(seed_file):
            for kind in row.get("expects_artifact") or []:
                assert kind in _KINDS, f"{row['id']}: unknown artifact kind {kind!r}"

    def test_every_example_carries_a_grading_criterion(self, seed_file: Path):
        """An example with neither a rubric nor assertions is silently never scored.

        Two shapes, one rule. `rubric` is prose for `evals/evaluators.py`'s judge;
        `assertions` is a list of behavioural claims for the Assertions evaluator template
        in LangSmith, which is what the workshop seeds use. Either satisfies the guard, and
        an example with neither passes every experiment by scoring nothing.
        """
        for row in yaml.safe_load(seed_file.read_text(encoding="utf-8")):
            rubric = row.get("rubric")
            assertions = row.get("assertions")
            if assertions is not None:
                assert isinstance(assertions, list) and assertions, row["id"]
                assert all(
                    isinstance(a, str) and a.strip() for a in assertions
                ), row["id"]
                continue
            assert isinstance(rubric, str), row["id"]
            assert rubric.strip(), row["id"]

    def test_no_example_carries_a_key_sync_would_silently_drop(self, seed_file: Path):
        known = {"id", "question", "expects_artifact", "rubric", "assertions", "domain",
                 "surface", "notes"}
        for row in _load(seed_file):
            assert set(row) <= known, f"{row['id']}: unknown key(s) {set(row) - known}"


@pytest.mark.parametrize("reply,expected", [
    ('{"pass": true, "reason": "supported"}', True),
    ('```json\n{"pass": false, "reason": "missing chart"}\n```', False),
    ('{"pass": 0.75}', None), ('[]', None), ('not json', None), ('{}', None),
])
async def test_judge_parses_real_message_text_and_includes_artifacts(monkeypatch, reply, expected):
    from unittest.mock import AsyncMock

    from langchain_core.messages import AIMessage

    from evals.evaluators import judge

    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(reply)))
    monkeypatch.setattr(judge, "judge_model", lambda: model)
    result = await judge.rubric_judge(
        _run(answer="Answer evidence", artifact_names=["chart", None]),
        _example(rubric="Must plot evidence"),
    )
    assert result["score"] is expected
    prompt = model.ainvoke.call_args.args[0]
    assert "Answer evidence" in prompt
    assert "Must plot evidence" in prompt
    assert "Deliverables this run published: chart" in prompt
    if expected is None:
        assert "unparseable" in result["comment"]
