"""PDF reader boundary contracts; actual PDF/image decoding needs the sandbox image."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deep_life_sci.middleware import upload_probe as probe


@pytest.fixture
def pdf_reader(monkeypatch, tmp_path):
    reader = SimpleNamespace(
        is_encrypted=False,
        metadata={"/Title": "Paper"},
        pages=[SimpleNamespace(extract_text=lambda: "Evidence. PMID: 33567185", images=[])],
    )
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Mock(return_value=reader)))
    monkeypatch.setattr(probe, "DERIVED", str(tmp_path / "derived"))
    # Decoding is the external boundary. With no embedded images this must not be called.
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=Mock()))
    return reader


def test_pdf_body_goes_to_a_sidecar_and_only_metadata_returns(pdf_reader):
    record = {}
    probe.probe_pdf("unused.pdf", record, "paper.pdf")
    assert record["pages"] == 1
    assert record["title"] == "Paper"
    assert record["pmid"] == "33567185"
    assert record["chars"] == len("Evidence. PMID: 33567185")
    assert Path(record["text_path"]).read_text() == "Evidence. PMID: 33567185"
    assert "Evidence." not in json.dumps(record)


def test_scanned_pdf_reports_no_text_and_writes_no_text_sidecar(pdf_reader):
    pdf_reader.pages[0].extract_text = lambda: None
    record = {}
    probe.probe_pdf("unused.pdf", record, "scan.pdf")
    assert record["chars"] == 0
    assert "scanned PDF" in record["note"]
    assert "text_path" not in record


@pytest.mark.parametrize("result", [0, ValueError("locked")])
def test_encrypted_pdf_reports_failed_decryption(pdf_reader, result):
    pdf_reader.is_encrypted = True
    pdf_reader.decrypt = (
        Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    )
    record = {}
    probe.probe_pdf("unused.pdf", record, "locked.pdf")
    assert "password-protected" in record["note"]
    assert "text_path" not in record
    pdf_reader.decrypt.assert_called_once_with("")


def test_figure_extraction_failure_does_not_lose_text(pdf_reader, monkeypatch):
    monkeypatch.setattr(probe, "extract_figures", Mock(side_effect=ValueError("decoder")))
    record = {}
    probe.probe_pdf("unused.pdf", record, "paper.pdf")
    assert Path(record["text_path"]).read_text().startswith("Evidence.")
    assert "ValueError" in record["figures_note"]


def test_extracted_figures_are_reencoded_capped_and_returned_as_paths(pdf_reader, monkeypatch):
    class DecodedImage:
        size = (1000, 1000)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def save(self, buffer, *, format):
            assert format == "PNG"
            buffer.write(b"encoded PNG")

    decoder = Mock(return_value=DecodedImage())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=SimpleNamespace(open=decoder)))
    monkeypatch.setattr(probe, "MAX_FIGURES", 1)
    pdf_reader.pages[0].images = [SimpleNamespace(data=b"original image")] * 2
    record = {}
    probe.extract_figures(pdf_reader, "paper.pdf", record)
    assert record["figures"] == 1
    assert record["more_figures"] == 1
    assert Path(record["figure_paths"][0]).read_bytes() == b"encoded PNG"
    assert "encoded PNG" not in json.dumps(record)
    assert "original image" not in json.dumps(record)
    decoder.assert_called_once()
