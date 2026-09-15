"""Read real OOXML ZIPs using the probe's standard-library workbook reader."""

import zipfile

import pytest

from deep_life_sci.middleware import upload_probe as probe

NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
REL = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


@pytest.mark.parametrize("target", ["worksheets/other.xml", "/xl/worksheets/other.xml"])
def test_workbook_resolves_relationships_and_shared_inline_and_literal_cells(tmp_path, target):
    path = tmp_path / "data.xlsx"
    with zipfile.ZipFile(path, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f"<workbook {NS} {REL}><sheets>"
            '<sheet name="Results" r:id="rId7"/>'
            '<sheet name="Notes" r:id="rId8"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships><Relationship Id="rId7" Target="{target}"/></Relationships>',
        )
        book.writestr(
            "xl/sharedStrings.xml", f"<sst {NS}><si><r><t>gene</t></r><r><t>_id</t></r></si></sst>"
        )
        book.writestr(
            "xl/worksheets/other.xml",
            f"<worksheet {NS}><sheetData>"
            '<row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="inlineStr"><is><t>count</t></is></c>'
            '<c r="C1"><v>2026</v></c></row>'
            '<row r="2"><c r="A2"><v>1</v></c></row>'
            '<row r="3"><c r="A3"><v>2</v></c></row>'
            "</sheetData></worksheet>",
        )
    record = {}
    probe.probe_tabular(str(path), ".xlsx", record)
    assert record == {
        "sheets": ["Results", "Notes"],
        "rows": 2,
        "columns": ["gene_id", "count", "2026"],
    }


def test_empty_workbook_reports_no_sheets(tmp_path):
    path = tmp_path / "empty.xlsx"
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/workbook.xml", f"<workbook {NS}><sheets/></workbook>")
    record = {}
    probe.probe_workbook(str(path), record)
    assert record == {"sheets": [], "note": "the workbook has no sheets"}


def test_sparse_header_keeps_blank_column_positions(tmp_path):
    path = tmp_path / "sparse.xlsx"
    with zipfile.ZipFile(path, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook {NS} {REL}><sheets><sheet name="Data" r:id="rId1"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            "<Relationships>"
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f"<worksheet {NS}><sheetData>"
            '<row r="1"><c r="A1" t="inlineStr"><is><t>gene</t></is></c>'
            '<c r="C1" t="inlineStr"><is><t>count</t></is></c></row>'
            '<row r="2"><c r="B2"><v>42</v></c></row>'
            "</sheetData></worksheet>",
        )
    record = {}
    probe.probe_workbook(str(path), record)
    assert record["columns"] == ["gene", "", "count"]
    assert record["rows"] == 1
