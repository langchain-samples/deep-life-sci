"""`middleware/uploads.py`: harvesting attachments and describing them to the model.

The rule this module exists to enforce is the one in CLAUDE.md: **the manifest is shapes
and identifiers, never contents.** An uploaded PDF's body in root context costs exactly
what `pmc_locate` exists to save, so the manifest carries a row count, a PMID list and a
sidecar path, and the bytes stay in the sandbox for a subagent or `execute`.

`_read_block` is the other half and is tested hardest, because everything it declines to
recognise is left in place and reaches the provider as-is. Getting that wrong is not a
missing feature — a `.xls` block sent to a provider with no such document type answers
with a 400 for the whole run.
"""

from __future__ import annotations

import base64

import pytest

from deep_life_sci.middleware.uploads import (
    MAX_UPLOAD_BYTES,
    UPLOAD_KINDS,
    UPLOAD_SUFFIXES,
    UploadMiddleware,
    _count,
    _detail,
    _extra_lines,
    _heredoc,
    _human_size,
    _ids,
    _parse_lines,
    _render_manifest,
    _safe_name,
    _suffix,
    _thread_key,
)
from deep_life_sci.paths import OUT_DIR, UPLOAD_DERIVED_DIR, UPLOAD_DIR


class TestSafeName:
    def test_an_ordinary_name_is_unchanged(self):
        assert _safe_name("counts.csv") == "counts.csv"

    def test_a_path_is_reduced_to_its_basename(self):
        assert _safe_name("/etc/passwd") == "passwd"

    def test_a_traversal_attempt_cannot_escape_the_upload_directory(self):
        assert "/" not in _safe_name("../../etc/passwd")

    def test_leading_dots_are_stripped_so_nothing_lands_hidden(self):
        assert _safe_name("...hidden.csv") == "hidden.csv"

    def test_shell_metacharacters_are_replaced(self):
        """The name is interpolated into a sandbox path."""
        assert _safe_name("deseq2 results;rm -rf.csv") == "deseq2_results_rm_-rf.csv"

    def test_a_very_long_name_is_truncated(self):
        assert len(_safe_name("x" * 500 + ".csv")) == 80

    @pytest.mark.parametrize("raw", ["", "   ", None, "..."])
    def test_an_unusable_name_gets_a_placeholder(self, raw):
        assert _safe_name(raw) == "upload"


class TestSuffix:
    @pytest.mark.parametrize("name", ["counts.CSV", "REFS.NBIB"])
    def test_is_case_insensitive(self, name: str):
        assert _suffix(name) == "." + name.split(".")[1].lower()

    def test_sees_through_gz_to_the_reader_that_matters(self):
        """The file stays compressed on disk; this only picks the reader."""
        assert _suffix("deseq2_results.csv.gz") == ".csv"

    def test_a_bare_gz_has_no_reader_suffix(self):
        assert _suffix("archive.gz") == ""

    def test_a_name_with_no_extension_has_no_suffix(self):
        assert _suffix("README") == ""


class TestHumanSize:
    @pytest.mark.parametrize(
        ("size", "expected"), [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2,
                                                                             "5.0 MB")]
    )
    def test_formats_at_the_right_scale(self, size: int, expected: str):
        assert _human_size(size) == expected

    def test_anything_larger_stays_in_megabytes(self):
        assert _human_size(2 * 1024**3).endswith("MB")

    @pytest.mark.parametrize("bad", [None, "abc", object()])
    def test_an_unreadable_size_says_so_rather_than_raising(self, bad):
        assert _human_size(bad) == "unknown size"


class TestCount:
    def test_pluralises(self):
        assert _count(1, "molecule") == "1 molecule"
        assert _count(12, "molecule") == "12 molecules"

    def test_takes_an_explicit_plural(self):
        assert _count(2, "entry", "entries") == "2 entries"

    def test_groups_thousands_because_the_manifest_is_prose(self):
        assert _count(12345, "reference") == "12,345 references"

    def test_none_counts_as_zero(self):
        assert _count(None, "reference") == "0 references"


class TestIds:
    def test_joins_the_list(self):
        assert _ids({"pmids": ["1", "2"]}, "pmids") == "1, 2"

    def test_says_how_many_were_left_out_and_where_they_are(self):
        record = {"pmids": ["1"], "more_pmids": 900}
        text = _ids(record, "pmids")
        assert "+900 more" in text
        assert "sidecar" in text

    def test_an_absent_key_is_an_empty_string(self):
        assert _ids({}, "pmids") == ""


class TestParseLines:
    def test_reads_one_json_object_per_line(self):
        assert _parse_lines('{"name": "a.csv", "bytes": 10}') == [
            {"name": "a.csv", "bytes": 10}
        ]

    def test_ignores_anything_the_shell_printed_alongside(self):
        output = 'bash: warning\n{"name": "a.csv"}\n[Command succeeded]'
        assert _parse_lines(output) == [{"name": "a.csv"}]

    def test_a_malformed_line_is_skipped(self):
        assert _parse_lines('{"broken\n{"name": "a"}') == [{"name": "a"}]

    def test_empty_output_yields_nothing(self):
        assert _parse_lines("") == []
        assert _parse_lines(None) == []


class TestHeredoc:
    def test_wraps_a_script_so_the_shell_expands_nothing_in_it(self):
        command = _heredoc("print('$HOME')")
        assert "<<'__UPLOADS_EOF__'" in command
        assert "print('$HOME')" in command

    def test_configuration_goes_in_as_environment_rather_than_interpolation(self):
        """`upload_probe.py` is a real file this repo lints; a `%(root)s` would ruin it."""
        command = _heredoc("import os", UPLOADS_ROOT="/workspace/uploads")
        assert command.startswith("UPLOADS_ROOT=/workspace/uploads python3 -")

    def test_a_value_needing_quoting_is_quoted(self):
        command = _heredoc("x", UPLOADS_ROOT="/tmp/a b")
        assert "'/tmp/a b'" in command

    def test_a_script_without_a_trailing_newline_still_closes_the_heredoc(self):
        assert _heredoc("x").endswith("__UPLOADS_EOF__")


class TestThreadKey:
    def test_falls_back_to_a_constant_outside_a_runnable_context(self):
        """The CLI has no thread."""
        assert _thread_key() == "default"


class TestDetail:
    def test_a_tabular_record_reports_its_shape(self):
        record = {"kind": "tabular", "bytes": 2048, "rows": 1200, "columns": ["a", "b"]}
        assert _detail(record) == ["2.0 KB", "1,200 rows x 2 columns"]

    def test_elided_columns_still_count_toward_the_width(self):
        record = {"kind": "tabular", "bytes": 0, "rows": 1, "columns": ["a"],
                  "more_columns": 60}
        assert "61 columns" in _detail(record)[1]

    def test_a_citations_record_reports_how_many_carry_a_pmid(self):
        record = {"kind": "citations", "bytes": 0, "references": 40, "with_pmid": 38,
                  "with_doi_only": 2}
        detail = _detail(record)
        assert "40 references" in detail
        assert "38 with a PMID" in detail
        assert "2 with a DOI but no PMID" in detail

    def test_a_pdf_reports_pages_and_character_count_not_text(self):
        record = {"kind": "pdf", "bytes": 0, "pages": 12, "chars": 41_000}
        assert _detail(record) == ["0 B", "12 pages", "41,000 chars of text"]

    def test_a_sequence_record_names_the_molecule_type(self):
        record = {
            "kind": "sequence",
            "bytes": 0,
            "sequences": 3,
            "format": "fasta",
            "molecule": "nucleotide",
            "total_residues": 9000,
        }
        detail = _detail(record)
        assert "3 fasta records" in detail
        assert "nucleotide" in detail

    def test_an_image_reports_its_dimensions(self):
        record = {"kind": "image", "bytes": 0, "width": 800, "height": 600,
                  "format": "png"}
        assert "800x600 png" in _detail(record)

    def test_a_record_with_no_kind_is_described_by_its_size_alone(self):
        assert _detail({"bytes": 1024}) == ["1.0 KB"]


class TestExtraLines:
    def test_a_citations_record_offers_the_pmids_and_the_sidecar(self):
        record = {
            "kind": "citations",
            "pmids": ["33567185"],
            "refs_path": f"{UPLOAD_DERIVED_DIR}/refs.nbib.refs.json",
        }
        lines = _extra_lines(record)
        assert lines[0] == "PMIDs: 33567185"
        assert lines[1].endswith("refs.nbib.refs.json")

    def test_a_pdf_offers_the_sidecar_path_and_never_the_text(self):
        """A median paper is ~40k chars — exactly the payload kept out of root context."""
        record = {
            "kind": "pdf",
            "title": "Once-Weekly Semaglutide",
            "doi": "10.1056/NEJMoa2032183",
            "text_path": f"{UPLOAD_DERIVED_DIR}/paper.pdf.txt",
            "lines": 900,
        }
        lines = _extra_lines(record)
        assert any("paper.pdf.txt" in line for line in lines)
        assert any("10.1056/NEJMoa2032183" in line for line in lines)

    def test_pdf_page_images_are_labelled_for_figure_analyst_not_read_file(self):
        """Reading a PNG back can cost more context than an entire run."""
        record = {"kind": "pdf", "figures": 2, "figure_paths": ["/a.png", "/b.png"]}
        assert any("figure-analyst, not readFile" in line for line in _extra_lines(record))

    def test_a_chem_sample_is_rendered_from_whichever_keys_it_has(self):
        """A `.smi` names molecules with SMILES; an SDF with a title and an atom count."""
        smi = _extra_lines({"kind": "chem", "samples": [{"name": "caffeine",
                                                         "smiles": "CN1C=NC2"}]})
        sdf = _extra_lines({"kind": "chem", "samples": [{"name": "caffeine", "atoms": 24}]})
        assert "CN1C=NC2" in smi[0]
        assert "24 atoms" in sdf[0]

    def test_an_older_manifests_extra_keys_stay_readable(self):
        """A thread can span a deploy; the sample is rendered from what it carries."""
        lines = _extra_lines(
            {"kind": "chem", "samples": [{"name": "caffeine", "formula": "C8H10N4O2",
                                          "mw": 194.19}]}
        )
        assert "C8H10N4O2" in lines[0]
        assert "194.19 g/mol" in lines[0]

    def test_a_note_is_appended_whatever_the_kind(self):
        assert _extra_lines({"kind": "image", "note": "truncated"})[-1] == "note: truncated"

    def test_an_elided_column_list_says_how_many_are_missing(self):
        record = {"kind": "tabular", "columns": ["a", "b"], "more_columns": 38}
        assert _extra_lines(record)[0].endswith("(+38 more)")


class TestRenderManifest:
    def test_one_line_per_file_with_its_path_and_kind(self):
        manifest = [
            {"path": f"{UPLOAD_DIR}/counts.csv", "kind": "tabular", "bytes": 1024,
             "rows": 10, "columns": ["a"]}
        ]
        rendered = _render_manifest(manifest)
        assert rendered.startswith("<uploaded_files>")
        assert rendered.endswith("</uploaded_files>")
        assert f"- {UPLOAD_DIR}/counts.csv [tabular] — 1.0 KB, 10 rows x 1 columns" in rendered

    def test_an_undescribed_file_is_labelled_unread_rather_than_defaulted(self):
        """Labelling it `tabular` invites code written against a shape that is not there."""
        rendered = _render_manifest([{"name": "mystery.csv", "bytes": 10}])
        assert "[unread]" in rendered

    def test_extra_lines_are_indented_under_their_record(self):
        manifest = [{"path": "/a.nbib", "kind": "citations", "bytes": 0, "references": 1,
                     "pmids": ["33567185"]}]
        assert "\n  PMIDs: 33567185" in _render_manifest(manifest)

    def test_the_manifest_never_carries_a_files_contents(self):
        """The rule from CLAUDE.md: shapes and identifiers, never contents."""
        manifest = [
            {
                "path": "/paper.pdf",
                "kind": "pdf",
                "bytes": 400_000,
                "pages": 12,
                "chars": 41_000,
                "text_path": f"{UPLOAD_DERIVED_DIR}/paper.pdf.txt",
                "lines": 900,
            }
        ]
        rendered = _render_manifest(manifest)
        assert "41,000 chars of text" in rendered
        assert f"{UPLOAD_DERIVED_DIR}/paper.pdf.txt" in rendered
        assert len(rendered) < 400


class TestReadBlock:
    @pytest.fixture
    def middleware(self) -> UploadMiddleware:
        return UploadMiddleware(backend=None)

    def _block(self, name: str, payload: bytes = b"gene,n\nTP53,1\n", **extra) -> dict:
        block = {
            "type": "file",
            "mimeType": "text/csv",
            "data": base64.b64encode(payload).decode(),
            "metadata": {"filename": name},
        }
        block.update(extra)
        return block

    def test_an_accepted_upload_is_staged_under_the_upload_directory(self, middleware):
        result = middleware._read_block(self._block("counts.csv"))
        assert result["path"] == f"{UPLOAD_DIR}/counts.csv"
        assert result["bytes"] == 14

    def test_the_upload_directory_is_not_under_the_deliverables_directory(self):
        """Or a file the user gave us comes back as a deliverable of their own question."""
        assert not UPLOAD_DIR.startswith(OUT_DIR)

    def test_the_marker_left_in_the_message_carries_no_sandbox_path(self, middleware):
        """The UI joins these into the user's own chat bubble."""
        result = middleware._read_block(self._block("counts.csv"))
        assert UPLOAD_DIR not in result["marker"]
        assert result["marker"] == "[attached counts.csv (14 B)]"

    def test_a_pasted_screenshot_with_no_filename_is_named_off_its_mime_type(
        self, middleware
    ):
        """Otherwise the whole image rides into root context for want of an extension."""
        block = {
            "type": "image",
            "mimeType": "image/png",
            "data": base64.b64encode(b"\x89PNG").decode(),
            "metadata": {},
        }
        result = middleware._read_block(block)
        assert result["name"].endswith(".png")
        assert result["path"].startswith(UPLOAD_DIR)

    def test_an_image_block_is_intercepted_like_a_file_block(self, middleware):
        block = self._block("gel.png", b"\x89PNG", type="image", mimeType="image/png")
        assert middleware._read_block(block)["path"].endswith("gel.png")

    def test_a_legacy_xls_is_declined_with_the_reason_in_place_of_a_path(self, middleware):
        """Left in, the block reaches a provider with no such type and 400s the run."""
        result = middleware._read_block(self._block("old.xls"))
        assert "path" not in result
        assert "re-save it as .xlsx" in result["marker"]

    def test_a_format_with_no_reader_is_left_in_place(self, middleware):
        """It then fails — or works — on its own terms rather than being swallowed here."""
        assert middleware._read_block(self._block("archive.zip")) is None

    def test_a_non_upload_block_is_left_alone(self, middleware):
        assert middleware._read_block({"type": "text", "text": "hi"}) is None
        assert middleware._read_block("not a block") is None

    def test_a_block_with_no_payload_says_so(self, middleware):
        block = self._block("counts.csv")
        del block["data"]
        assert "no payload" in middleware._read_block(block)["marker"]

    def test_an_undecodable_payload_says_so(self, middleware):
        block = self._block("counts.csv")
        block["data"] = "not base64!!"
        assert "undecodable" in middleware._read_block(block)["marker"]

    def test_an_oversize_upload_is_refused_with_both_sizes(self, middleware):
        block = self._block("big.csv", b"x" * (MAX_UPLOAD_BYTES + 1))
        marker = middleware._read_block(block)["marker"]
        assert "exceeds the" in marker
        assert "15.0 MB" in marker

    def test_the_snake_case_mime_key_is_read_too(self, middleware):
        block = self._block("counts.csv")
        block["mime_type"] = block.pop("mimeType")
        assert middleware._read_block(block)["mime"] == "text/csv"

    @pytest.mark.parametrize(
        "name", ["../../etc/passwd.csv", "/etc/passwd.csv", "a/b/../c.csv"]
    )
    def test_a_traversal_filename_cannot_escape_the_upload_directory(
        self, middleware, name: str
    ):
        path = middleware._read_block(self._block(name))["path"]
        assert path.startswith(f"{UPLOAD_DIR}/")
        assert ".." not in path
        assert path.count("/") == UPLOAD_DIR.count("/") + 1


class TestUploadKindsTable:
    def test_the_suffix_set_is_derived_from_the_table(self):
        assert frozenset(UPLOAD_KINDS) == UPLOAD_SUFFIXES

    def test_txt_is_the_one_suffix_the_probe_may_override(self):
        """A PMID list and a headerless TSV both arrive as `.txt`."""
        assert UPLOAD_KINDS[".txt"] == "tabular"

    def test_xls_is_deliberately_absent(self):
        """Reading it needs xlrd, and the sandbox blocks runtime installs."""
        assert ".xls" not in UPLOAD_KINDS

    @pytest.mark.parametrize("suffix", sorted(UPLOAD_KINDS))
    def test_every_suffix_is_lowercase_and_dotted(self, suffix: str):
        assert suffix.startswith(".") and suffix == suffix.lower()

    def test_the_upload_limit_sits_above_the_artifact_inline_cap(self):
        """Different directions, different costs — see the note on each constant."""
        from deep_life_sci.middleware.artifacts import MAX_INLINE_BYTES

        assert MAX_UPLOAD_BYTES > MAX_INLINE_BYTES
