"""`middleware/upload_probe.py`: the readers that run in the sandbox before the first token.

This module is the one place in the repo that is deliberately on the standard library
only, and the reason is latency: it runs *before* the first model call, and the first
`import pandas` or `import rdkit` in a fresh container is 2-11s of lazy snapshot restore.
That constraint is what makes these hand-written parsers worth testing — they replace
library readers, so the bar is the library's output, not "close enough".

The module is imported directly rather than executed in a container. It reads its
configuration from the environment at import time, so the tests that need a sidecar
directory rebind the module attribute, exactly as the sandbox would set the env var.
"""

from __future__ import annotations

import gzip
import json
import struct
import zlib

import pytest

from deep_life_sci.middleware import upload_probe as probe
from deep_life_sci.middleware.upload_probe import (
    MAX_COLS,
    MAX_IDS,
    base_suffix,
    cap_ids,
    delimiter_for,
    header_size,
    looks_like_id_list,
    medline_entries,
    molfile_atoms,
    parse_bibtex,
    parse_fasta,
    parse_id_list,
    parse_medline,
    parse_ris,
    probe_delimited,
    resolve_kind,
    set_columns,
)


@pytest.fixture
def derived(tmp_path, monkeypatch):
    """Point the sidecar directory at tmp_path, as `UPLOADS_DERIVED` would in the sandbox."""
    target = tmp_path / "derived"
    monkeypatch.setattr(probe, "DERIVED", str(target))
    return target


class TestBaseSuffix:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("counts.csv", ".csv"),
            ("COUNTS.CSV", ".csv"),
            ("refs.nbib", ".nbib"),
            ("paper.pdf", ".pdf"),
        ],
    )
    def test_reads_the_extension(self, name: str, expected: str):
        assert base_suffix(name) == expected

    @pytest.mark.parametrize(
        "name", ["deseq2.csv.gz", "series_matrix.txt.gz", "compounds.sdf.gz"]
    )
    def test_sees_through_gz_to_the_reader_that_matters(self, name: str):
        """A DESeq2 table, a MAF and a GEO series matrix all routinely arrive gzipped."""
        assert base_suffix(name) == "." + name.split(".")[-2]

    @pytest.mark.parametrize(
        "name", ["counts.csv", "counts.csv.gz", "refs.RIS", "x.tar.gz", "paper.pdf"]
    )
    def test_matches_the_host_sides_own_suffix_rule(self, name: str):
        """`uploads._suffix` mirrors this; the two disagreeing picks the wrong reader."""
        from deep_life_sci.middleware.uploads import _suffix

        assert base_suffix(name) == _suffix(name)

    def test_an_extensionless_name_is_where_the_two_rules_diverge(self):
        """Recorded rather than asserted equal, because today it cannot be reached.

        `rpartition` has no "no separator" branch here, so `README` comes back as
        `.readme` while `uploads._suffix` returns `""`. It does not matter as things
        stand: `""` is not in `UPLOAD_SUFFIXES`, so the host rejects an extensionless
        attachment and the probe never sees one. It would matter the moment anything
        started calling the probe on a file the composer did not gate, so the divergence
        is pinned here rather than left to be discovered then.
        """
        from deep_life_sci.middleware.uploads import UPLOAD_SUFFIXES, _suffix

        assert base_suffix("README") == ".readme"
        assert _suffix("README") == ""
        assert "" not in UPLOAD_SUFFIXES


class TestOpenText:
    def test_reads_a_plain_file(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("hello")
        with probe.open_text(str(path)) as handle:
            assert handle.read() == "hello"

    def test_decompresses_a_gz_transparently(self, tmp_path):
        path = tmp_path / "a.csv.gz"
        path.write_bytes(gzip.compress(b"a,b\n1,2\n"))
        with probe.open_text(str(path)) as handle:
            assert handle.read() == "a,b\n1,2\n"

    def test_a_stray_latin1_byte_does_not_cost_the_upload(self, tmp_path):
        """These are files off a user's laptop; an author name must not break the run."""
        path = tmp_path / "a.txt"
        path.write_bytes(b"Bj\xf8rk")
        with probe.open_text(str(path)) as handle:
            assert "Bj" in handle.read()


class TestCapsAndColumns:
    def test_cap_ids_bounds_the_list_and_counts_what_was_left_out(self):
        record: dict = {}
        cap_ids([str(n) for n in range(MAX_IDS + 7)], record, "pmids")
        assert len(record["pmids"]) == MAX_IDS
        assert record["more_pmids"] == 7

    def test_cap_ids_deduplicates_before_capping(self):
        record: dict = {}
        cap_ids(["1", "1", "2"], record, "pmids")
        assert record["pmids"] == ["1", "2"]
        assert "more_pmids" not in record

    def test_cap_ids_drops_empties(self):
        record: dict = {}
        cap_ids(["1", "", None], record, "pmids")  # type: ignore[list-item]
        assert record["pmids"] == ["1"]

    def test_a_short_list_carries_no_overflow_key(self):
        record: dict = {}
        cap_ids(["1", "2"], record, "pmids")
        assert "more_pmids" not in record

    def test_set_columns_bounds_a_wide_frames_column_list(self):
        record: dict = {}
        set_columns(record, [f"c{n}" for n in range(MAX_COLS + 3)])
        assert len(record["columns"]) == MAX_COLS
        assert record["more_columns"] == 3


class TestDelimiterFor:
    def test_a_named_suffix_is_not_sniffed(self):
        assert delimiter_for("a\tb\tc", ".csv") == ","
        assert delimiter_for("a,b,c", ".tsv") == "\t"

    def test_a_txt_is_sniffed(self):
        """Only `.txt` genuinely needs it — a tool wrote it with whatever it liked."""
        assert delimiter_for("gene\tbaseMean\tlog2FC\nA\t1\t2\n", ".txt") == "\t"

    def test_a_one_column_txt_falls_back_by_counting_rather_than_raising(self):
        """`csv.Sniffer` raises on a file with no consistent delimiter, which this is."""
        assert delimiter_for("alpha\nbeta\ngamma\n", ".txt") in (",", "\t")

    def test_the_fallback_prefers_whichever_candidate_appears_more(self):
        assert delimiter_for("a\tb\nc\td\n", ".txt") == "\t"


class TestProbeDelimited:
    def test_counts_rows_excluding_the_header(self, tmp_path):
        path = tmp_path / "counts.csv"
        path.write_text("gene,baseMean\nTP53,10\nMDM2,20\n")
        record: dict = {}
        probe_delimited(str(path), ".csv", record)
        assert record["rows"] == 2
        assert record["columns"] == ["gene", "baseMean"]

    def test_a_quoted_newline_does_not_inflate_the_row_count(self, tmp_path):
        """Counting newlines would disagree with the agent's own `read_csv`."""
        path = tmp_path / "a.csv"
        path.write_text('name,note\n"A","line one\nline two"\n')
        record: dict = {}
        probe_delimited(str(path), ".csv", record)
        assert record["rows"] == 1

    def test_an_empty_file_says_so_rather_than_reporting_a_shape(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("   \n")
        record: dict = {}
        probe_delimited(str(path), ".csv", record)
        assert record["rows"] == 0
        assert "empty" in record["note"]

    def test_a_gzipped_table_is_read_through(self, tmp_path):
        path = tmp_path / "a.csv.gz"
        path.write_bytes(gzip.compress(b"gene,baseMean\nTP53,10\n"))
        record: dict = {}
        probe_delimited(str(path), ".csv", record)
        assert record["rows"] == 1


# --------------------------------------------------------------------------------
# Bibliographies
# --------------------------------------------------------------------------------

_NBIB = """\
PMID- 33567185
TI  - Once-Weekly Semaglutide in Adults with Overweight
      or Obesity
TA  - N Engl J Med
DP  - 2021 Mar 18
AU  - Wilding JPH
AU  - Batterham RL
AU  - Calanna S
AU  - Davies M
AID - 10.1056/NEJMoa2032183 [doi]

PMID- 33667417
TI  - Effect of Subcutaneous Semaglutide vs Placebo
TA  - JAMA
DP  - 2021 Apr 13
AU  - Davies M
"""


class TestMedline:
    def test_splits_records_on_the_blank_line(self):
        assert len(medline_entries(_NBIB)) == 2

    def test_a_repeated_tag_becomes_a_list(self):
        assert medline_entries(_NBIB)[0]["AU"] == [
            "Wilding JPH",
            "Batterham RL",
            "Calanna S",
            "Davies M",
        ]

    def test_a_wrapped_title_is_rejoined_with_the_space_the_break_stood_for(self):
        title = parse_medline(_NBIB)[0]["title"]
        assert title == "Once-Weekly Semaglutide in Adults with Overweight or Obesity"

    def test_reads_the_fields_the_agent_hydrates_from(self):
        record = parse_medline(_NBIB)[0]
        assert record["pmid"] == "33567185"
        assert record["journal"] == "N Engl J Med"
        assert record["year"] == "2021"

    def test_the_doi_is_taken_from_the_aid_tagged_doi(self):
        assert parse_medline(_NBIB)[0]["doi"] == "10.1056/NEJMoa2032183"

    def test_authors_are_capped_at_three(self):
        assert len(parse_medline(_NBIB)[0]["authors"]) == 3

    def test_a_trailing_record_with_no_blank_line_is_still_read(self):
        assert len(parse_medline("PMID- 1\nTI  - A\n")) == 1

    def test_an_empty_file_parses_to_nothing(self):
        assert parse_medline("") == []


_RIS = """\
TY  - JOUR
AU  - Wilding, J P H
AU  - Batterham, R L
TI  - Once-Weekly Semaglutide in Adults with Overweight
  or Obesity
JO  - N Engl J Med
PY  - 2021
DO  - 10.1056/NEJMoa2032183
UR  - https://pubmed.ncbi.nlm.nih.gov/33567185/
ER  -

TY  - JOUR
T1  - A second paper
T2  - JAMA
Y1  - 2022
AN  - 33667417
ER  -
"""


class TestRis:
    def test_reads_both_records(self):
        assert len(parse_ris(_RIS)) == 2

    def test_a_pmid_is_recovered_from_a_pubmed_url(self):
        assert parse_ris(_RIS)[0]["pmid"] == "33567185"

    def test_a_bare_pmid_in_an_accession_field_is_recovered(self):
        assert parse_ris(_RIS)[1]["pmid"] == "33667417"

    def test_the_alternate_title_and_journal_tags_are_read(self):
        record = parse_ris(_RIS)[1]
        assert record["title"] == "A second paper"
        assert record["journal"] == "JAMA"
        assert record["year"] == "2022"

    def test_a_wrapped_value_with_no_continuation_marker_is_rejoined(self):
        """RIS has none, so anything that is not a tag line belongs to the tag above."""
        assert parse_ris(_RIS)[0]["title"].endswith("Overweight or Obesity")

    def test_authors_are_read_in_order(self):
        assert parse_ris(_RIS)[0]["authors"] == ["Wilding, J P H", "Batterham, R L"]

    def test_a_record_with_nothing_in_it_is_dropped(self):
        assert parse_ris("TY  - JOUR\nER  -\n") == []


_BIBTEX = """\
@article{wilding2021,
  title = {Once-Weekly {Semaglutide} in Adults with Overweight or Obesity},
  author = {Wilding, John P H and Batterham, Rachel L and Calanna, Salvatore
            and Davies, Melanie},
  journal = {New England Journal of Medicine},
  year = {2021},
  doi = {10.1056/NEJMoa2032183},
}

@inproceedings{quoted2022,
  title = "A quoted title",
  booktitle = "Proceedings of Something",
  year = 2022,
  note = {See https://pubmed.ncbi.nlm.nih.gov/33667417/ for details},
}
"""


class TestBibtex:
    def test_reads_both_entry_types(self):
        assert len(parse_bibtex(_BIBTEX)) == 2

    def test_a_braced_value_containing_braces_is_matched_rather_than_split(self):
        """A BibTeX title routinely contains braces, so any split on `}` cuts it in half."""
        title = parse_bibtex(_BIBTEX)[0]["title"]
        assert title == "Once-Weekly Semaglutide in Adults with Overweight or Obesity"

    def test_a_doi_does_not_come_out_with_a_trailing_brace(self):
        """It would then match nothing in PubMed."""
        assert parse_bibtex(_BIBTEX)[0]["doi"] == "10.1056/NEJMoa2032183"

    def test_a_quoted_value_is_read(self):
        assert parse_bibtex(_BIBTEX)[1]["title"] == "A quoted title"

    def test_a_bare_value_runs_to_the_next_comma(self):
        assert parse_bibtex(_BIBTEX)[1]["year"] == "2022"

    def test_booktitle_stands_in_for_a_missing_journal(self):
        assert parse_bibtex(_BIBTEX)[1]["journal"] == "Proceedings of Something"

    def test_authors_are_split_on_and_and_capped(self):
        authors = parse_bibtex(_BIBTEX)[0]["authors"]
        assert len(authors) == 3
        assert authors[0] == "Wilding, John P H"

    def test_a_pmid_url_anywhere_in_the_entry_is_found(self):
        assert parse_bibtex(_BIBTEX)[1]["pmid"] == "33667417"

    def test_an_explicit_pmid_field_is_reduced_to_its_digits(self):
        assert parse_bibtex("@article{x, pmid = {PMID: 123456} }")[0]["pmid"] == "123456"

    def test_an_entry_with_no_fields_is_skipped(self):
        assert parse_bibtex("@article{empty}") == []


class TestIdList:
    def test_reads_bare_pmids(self):
        assert [r["pmid"] for r in parse_id_list("33567185\n33667417\n")] == [
            "33567185",
            "33667417",
        ]

    def test_reads_a_labelled_pmid(self):
        assert parse_id_list("PMID: 33567185")[0]["pmid"] == "33567185"

    def test_reads_a_pubmed_url(self):
        line = "https://pubmed.ncbi.nlm.nih.gov/33567185/"
        assert parse_id_list(line)[0]["pmid"] == "33567185"

    def test_reads_a_doi_with_no_pmid(self):
        record = parse_id_list("10.1056/NEJMoa2032183")[0]
        assert record["doi"] == "10.1056/NEJMoa2032183"
        assert record["pmid"] == ""

    def test_a_trailing_sentence_period_is_not_part_of_the_doi(self):
        assert parse_id_list("10.1056/NEJMoa2032183.")[0]["doi"] == "10.1056/NEJMoa2032183"

    def test_comments_and_blank_lines_are_skipped(self):
        assert parse_id_list("# my list\n\n33567185\n") == [
            {"pmid": "33567185", "doi": "", "title": ""}
        ]


class TestLooksLikeIdList:
    def test_a_column_of_pmids_is_a_citation_list(self):
        """Checked before the tabular reader, which reads it as a one-column table."""
        assert looks_like_id_list("33567185\n33667417\n33625476\n") is True

    def test_a_column_of_dois_is_too(self):
        text = "\n".join(f"10.1056/NEJMoa20321{n:02d}" for n in range(5))
        assert looks_like_id_list(text) is True

    def test_a_delimited_table_is_not(self):
        assert looks_like_id_list("gene\tbaseMean\nTP53\t10\nMDM2\t20\n") is False

    def test_a_mostly_prose_file_is_not(self):
        text = "33567185\n" + "\n".join("some prose line" for _ in range(10))
        assert looks_like_id_list(text) is False

    def test_an_empty_file_is_not(self):
        assert looks_like_id_list("") is False

    def test_two_ids_alone_are_not_enough(self):
        assert looks_like_id_list("33567185\n33667417\n") is False


class TestProbeCitations:
    def test_writes_the_pmids_and_a_sidecar(self, tmp_path, derived):
        path = tmp_path / "refs.nbib"
        path.write_text(_NBIB)
        record: dict = {}
        probe.probe_citations(str(path), ".nbib", record, "refs.nbib")
        assert record["references"] == 2
        assert record["with_pmid"] == 2
        assert record["pmids"] == ["33567185", "33667417"]
        assert json.loads((derived / "refs.nbib.refs.json").read_text())[0]["pmid"] == (
            "33567185"
        )

    def test_a_nbib_that_is_really_ris_falls_back_rather_than_reporting_zero(
        self, tmp_path, derived
    ):
        """One reparse turns '0 references' into a working corpus."""
        path = tmp_path / "refs.nbib"
        path.write_text(_RIS)
        record: dict = {}
        probe.probe_citations(str(path), ".nbib", record, "refs.nbib")
        assert record["references"] == 2

    def test_a_doi_only_reference_is_counted_separately(self, tmp_path, derived):
        path = tmp_path / "refs.txt"
        path.write_text("10.1056/NEJMoa2032183\n33567185\n")
        record: dict = {}
        probe.probe_citations(str(path), ".txt", record, "refs.txt")
        assert record["with_pmid"] == 1
        assert record["with_doi_only"] == 1

    def test_an_unparseable_file_says_so_and_writes_no_sidecar(self, tmp_path, derived):
        path = tmp_path / "refs.ris"
        path.write_text("nothing resembling a bibliography")
        record: dict = {}
        probe.probe_citations(str(path), ".ris", record, "refs.ris")
        assert record["references"] == 0
        assert "no references" in record["note"]
        assert "refs_path" not in record


# --------------------------------------------------------------------------------
# Chemistry and sequences
# --------------------------------------------------------------------------------


class TestMolfileAtoms:
    def test_reads_a_v2000_fixed_width_counts_line(self):
        lines = ["caffeine", "  ChemDraw", "", " 24 25  0  0  0  0  0  0  0  0999 V2000"]
        assert molfile_atoms(lines) == 24

    def test_reads_a_v3000_counts_line(self):
        lines = [
            "big",
            "",
            "",
            "  0  0  0  0  0  0            999 V3000",
            "M  V30 BEGIN CTAB",
            "M  V30 COUNTS 42 45 0 0 0",
        ]
        assert molfile_atoms(lines) == 42

    def test_a_file_that_is_not_a_molfile_reports_nothing_rather_than_guessing(self):
        assert molfile_atoms(["just", "some", "text", "not a counts line"]) is None

    def test_a_block_too_short_to_have_a_counts_line_reports_nothing(self):
        assert molfile_atoms(["a", "b"]) is None


class TestParseFasta:
    def test_reads_ids_descriptions_and_lengths(self, tmp_path):
        path = tmp_path / "s.fasta"
        path.write_text(">sp|P04637|P53_HUMAN Cellular tumor antigen p53\nMEEPQSDPSV\nKEPGG\n")
        (record,) = parse_fasta(str(path))
        assert record["id"] == "sp|P04637|P53_HUMAN"
        assert record["description"] == "Cellular tumor antigen p53"
        assert record["length"] == 15

    def test_reads_several_records(self, tmp_path):
        path = tmp_path / "s.fasta"
        path.write_text(">a\nACGT\n>b\nACGTACGT\n")
        assert [r["length"] for r in parse_fasta(str(path))] == [4, 8]

    def test_residues_are_sampled_rather_than_held(self, tmp_path):
        """`SeqIO.parse` held every residue of every record; this streams."""
        path = tmp_path / "s.fasta"
        path.write_text(">a\n" + ("ACGT" * 1000) + "\n")
        (record,) = parse_fasta(str(path))
        assert record["length"] == 4000
        assert len(record["residues"]) == 60

    def test_residues_before_any_header_are_ignored(self, tmp_path):
        path = tmp_path / "s.fasta"
        path.write_text("ACGT\n>a\nACGT\n")
        assert len(parse_fasta(str(path))) == 1

    def test_a_header_with_no_description_still_reads(self, tmp_path):
        path = tmp_path / "s.fasta"
        path.write_text(">a\nACGT\n")
        assert parse_fasta(str(path))[0]["description"] == ""


class TestProbeSequence:
    def test_a_nucleotide_alphabet_is_detected(self, tmp_path, derived):
        """Nucleotide and protein lead to completely different literature searches."""
        path = tmp_path / "s.fasta"
        path.write_text(">a\nACGTACGTNN\n")
        record: dict = {}
        probe.probe_sequence(str(path), ".fasta", record, "s.fasta")
        assert record["molecule"] == "nucleotide"
        assert record["sequences"] == 1
        assert record["total_residues"] == 10

    def test_a_protein_alphabet_is_detected(self, tmp_path, derived):
        path = tmp_path / "s.fasta"
        path.write_text(">p53\nMEEPQSDPSVEPPLSQETFSDLWKLL\n")
        record: dict = {}
        probe.probe_sequence(str(path), ".fasta", record, "s.fasta")
        assert record["molecule"] == "protein"

    def test_the_sidecar_carries_the_records_without_their_residues(
        self, tmp_path, derived
    ):
        path = tmp_path / "s.fasta"
        path.write_text(">a\nACGT\n")
        record: dict = {}
        probe.probe_sequence(str(path), ".fasta", record, "s.fasta")
        (entry,) = json.loads((derived / "s.fasta.seqs.json").read_text())
        assert "residues" not in entry
        assert entry["length"] == 4

    def test_an_unparseable_file_says_so(self, tmp_path, derived):
        path = tmp_path / "s.fasta"
        path.write_text("not a fasta file at all\n")
        record: dict = {}
        probe.probe_sequence(str(path), ".fasta", record, "s.fasta")
        assert record["sequences"] == 0
        assert "no fasta records" in record["note"]


class TestProbeChem:
    def test_a_smi_needs_no_library_because_its_smiles_are_the_file(
        self, tmp_path, derived
    ):
        path = tmp_path / "c.smi"
        path.write_text("CN1C=NC2=C1C(=O)N(C)C(=O)N2C caffeine\nCCO ethanol\n")
        record: dict = {}
        probe.probe_chem(str(path), ".smi", record, "c.smi")
        assert record["molecules"] == 2
        assert record["samples"][0] == {
            "name": "caffeine",
            "smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        }

    def test_an_unnamed_smiles_line_gets_a_positional_name(self, tmp_path, derived):
        path = tmp_path / "c.smi"
        path.write_text("CCO\n")
        record: dict = {}
        probe.probe_chem(str(path), ".smi", record, "c.smi")
        assert record["samples"][0]["name"] == "mol_1"

    def test_an_sdf_is_counted_and_its_property_columns_listed(self, tmp_path, derived):
        block = (
            "caffeine\n  ChemDraw\n\n"
            " 24 25  0  0  0  0  0  0  0  0999 V2000\n"
            "M  END\n"
            "> <PUBCHEM_CID>\n2519\n\n"
            "> <PUBCHEM_IUPAC_NAME>\ncaffeine\n\n"
            "$$$$\n"
        )
        path = tmp_path / "c.sdf"
        path.write_text(block * 2)
        record: dict = {}
        probe.probe_chem(str(path), ".sdf", record, "c.sdf")
        assert record["molecules"] == 2
        assert record["properties"] == ["PUBCHEM_CID", "PUBCHEM_IUPAC_NAME"]
        assert record["samples"][0] == {"name": "caffeine", "atoms": 24}

    def test_a_block_that_is_not_a_molfile_is_counted_as_unparsed(self, tmp_path, derived):
        path = tmp_path / "c.sdf"
        path.write_text("junk\nmore junk\n$$$$\n")
        record: dict = {}
        probe.probe_chem(str(path), ".sdf", record, "c.sdf")
        assert record["molecules"] == 0
        assert record["unparsed"] == 1
        assert "no molecule records" in record["note"]


# --------------------------------------------------------------------------------
# Images: dimensions off the header, never through PIL
# --------------------------------------------------------------------------------


def _png(width: int, height: int) -> bytes:
    ihdr = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr) - 4)
        + ihdr
        + struct.pack(">I", zlib.crc32(ihdr))
    )


def _gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 20


def _webp_vp8(width: int, height: int) -> bytes:
    # 4-byte chunk size, 3-byte frame tag, then the sync code, then the dimensions — which
    # is what puts them at bytes 26-30 of the file.
    body = (
        b"VP8 "
        + b"\x00" * 4
        + b"\x00" * 3
        + b"\x9d\x01\x2a"
        + struct.pack("<HH", width, height)
    )
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


def _jpeg(width: int, height: int) -> bytes:
    comment = b"\xff\xfe" + struct.pack(">H", 10) + b"\x00" * 8
    sof0 = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width)
    return b"\xff\xd8" + comment + sof0 + b"\xff\xd9"


class TestHeaderSize:
    @pytest.mark.parametrize(
        ("maker", "fmt"),
        [(_png, "png"), (_gif, "gif"), (_webp_vp8, "webp"), (_jpeg, "jpeg")],
    )
    def test_reads_dimensions_without_a_decoder(self, tmp_path, maker, fmt: str):
        """PIL is 2s of cold import for two integers."""
        path = tmp_path / "img"
        path.write_bytes(maker(640, 480))
        assert header_size(str(path)) == (fmt, 640, 480)

    def test_a_jpeg_frame_is_found_past_intervening_segments(self, tmp_path):
        path = tmp_path / "img.jpg"
        path.write_bytes(_jpeg(1024, 768))
        assert header_size(str(path)) == ("jpeg", 1024, 768)

    def test_a_lossless_webp_defers_to_pil_rather_than_guessing(self, tmp_path):
        """VP8L's header is bit-packed; reading it here would report a wrong size."""
        body = b"VP8L" + b"\x00" * 20
        path = tmp_path / "img.webp"
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body)
        assert header_size(str(path)) is None

    def test_an_unrecognised_format_reports_nothing(self, tmp_path):
        path = tmp_path / "img.tif"
        path.write_bytes(b"II*\x00" + b"\x00" * 40)
        assert header_size(str(path)) is None


class TestProbeImage:
    def test_records_format_and_dimensions(self, tmp_path):
        path = tmp_path / "gel.png"
        path.write_bytes(_png(800, 600))
        record: dict = {}
        probe.probe_image(str(path), record)
        assert record == {"format": "png", "width": 800, "height": 600}


# --------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------


class TestResolveKind:
    @pytest.fixture(autouse=True)
    def kinds(self, monkeypatch):
        from deep_life_sci.middleware.uploads import UPLOAD_KINDS

        monkeypatch.setattr(probe, "KINDS", dict(UPLOAD_KINDS))

    @pytest.mark.parametrize(
        ("suffix", "kind"),
        [
            (".csv", "tabular"),
            (".xlsx", "tabular"),
            (".nbib", "citations"),
            (".ris", "citations"),
            (".pdf", "pdf"),
            (".sdf", "chem"),
            (".fasta", "sequence"),
            (".png", "image"),
        ],
    )
    def test_every_other_suffix_is_decided_by_name_alone(
        self, tmp_path, suffix: str, kind: str
    ):
        path = tmp_path / f"f{suffix}"
        path.write_bytes(b"")
        assert resolve_kind(str(path), suffix) == kind

    def test_a_txt_of_pmids_is_sniffed_as_a_bibliography(self, tmp_path):
        """A PMID list and a headerless TSV both arrive as `.txt`."""
        path = tmp_path / "f.txt"
        path.write_text("33567185\n33667417\n33625476\n33755728\n")
        assert resolve_kind(str(path), ".txt") == "citations"

    def test_a_txt_of_table_rows_is_read_as_tabular(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("gene\tbaseMean\nTP53\t10\nMDM2\t20\n")
        assert resolve_kind(str(path), ".txt") == "tabular"

    def test_an_unreadable_txt_falls_through_to_the_tabular_reader(self, tmp_path):
        assert resolve_kind(str(tmp_path / "absent.txt"), ".txt") == "tabular"

    def test_an_unregistered_suffix_defaults_to_tabular(self, tmp_path):
        path = tmp_path / "f.dat"
        path.write_bytes(b"")
        assert resolve_kind(str(path), ".dat") == "tabular"


class TestMain:
    def test_emits_one_json_record_per_file_and_never_crashes(
        self, tmp_path, monkeypatch, capsys, derived
    ):
        """Every failure is a note on the record; a crash costs the whole turn."""
        (tmp_path / "counts.csv").write_text("gene,n\nTP53,1\n")
        (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
        (tmp_path / "derived").mkdir(exist_ok=True)

        monkeypatch.setattr(probe, "ROOT", str(tmp_path))
        from deep_life_sci.middleware.uploads import UPLOAD_KINDS

        monkeypatch.setattr(probe, "KINDS", dict(UPLOAD_KINDS))
        probe.main()

        records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        by_name = {r["name"]: r for r in records}
        assert by_name["counts.csv"]["rows"] == 1
        assert "could not read" in by_name["broken.pdf"]["note"]

    def test_a_subdirectory_is_not_probed_as_an_upload(
        self, tmp_path, monkeypatch, capsys
    ):
        """`derived/` falls out of the `isfile` check, so a sidecar is never re-probed."""
        (tmp_path / "derived").mkdir()
        (tmp_path / "derived" / "refs.json").write_text("[]")
        monkeypatch.setattr(probe, "ROOT", str(tmp_path))
        probe.main()
        assert capsys.readouterr().out == ""
