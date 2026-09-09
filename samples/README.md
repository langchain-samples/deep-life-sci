# Sample uploads

One real file per upload kind `research_agent/middleware/uploads.py` accepts, for trying
the composer by hand and for the checks under `scripts/`. Everything here was fetched from
a public API or an open-access journal — nothing is synthetic, because a hand-made file
tests the parser against what we imagined rather than against what a user will actually
attach. Provenance is below; re-fetch rather than edit.

| File | Kind | Source |
|---|---|---|
| `airway-dex-vs-untreated.txt.gz` | tabular | GEO [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778) supplementary — cuffdiff output, 23k rows × 14 columns, gzipped and named `.txt`, which is the shape the reader has to sniff |
| `base-editing-liver.nbib` | citations | PubMed export of 30 records for `"base editing" AND liver`, via E-utilities `efetch&rettype=medline` |
| `zotero-library.ris` | citations | DOI content negotiation (`Accept: application/x-research-info-systems`) for 12 prime-editing papers — DOIs, no PMIDs, which is what a reference manager export usually looks like |
| `manuscript-refs.bib` | citations | the same 12 DOIs as BibTeX (`Accept: application/x-bibtex`) |
| `pmid-list.txt` | citations | a bare PMID list, the `.txt` a user saves out of PubMed's clipboard |
| `tale-base-editing-paper.pdf` | pdf | PLOS ONE [10.1371/journal.pone.0289509](https://doi.org/10.1371/journal.pone.0289509), CC-BY |
| `tale-assembly-figure.png` | image | figure 1 of the same paper, CC-BY |
| `kinase-inhibitors.sdf` | chem | PubChem PUG-REST, 8 CIDs as 2D SDF (public domain) |
| `compound-set.smi` | chem | PubChem canonical SMILES + titles for 12 compounds |
| `aspirin.mol` | chem | the first record of the PubChem SDF for CID 2244 |
| `tp53-pathway.fasta` | sequence | NCBI protein `efetch&rettype=fasta` — p53, BRCA2, BARD1, TET2 |
| `pten-mrna.gb` | sequence | NCBI nuccore `efetch&rettype=gb` — `NM_000314.8` |

The set deliberately includes files that exercise the awkward paths: a `.txt` that is a
table and a `.txt` that is a PMID list, a gzipped table, a bibliography with no PMIDs in
it at all, and an SDF whose 36 property fields have to be capped before they reach the
prompt.
