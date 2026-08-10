"""External data sources: NCBI E-utilities and the PMC open-data bucket.

The guards in `pubmed.py` and `pmc.py` are not boilerplate. Each one corresponds to a
verified API failure mode that returns a *wrong answer rather than an error* — PMID
tokenization, silent query rewriting, esummary's 500-UID cap answering HTTP 200, closed
articles served as a complete `<front>` with no `<body>`. See `docs/pubmed_api_notes/`
and `docs/pmc_api_notes/` for the probe results behind each.
"""

from research_agent.sources.pmc import fetch_full_text, make_sandbox_tools, pmc_locate
from research_agent.sources.pubmed import fetch_abstracts, pubmed_search

__all__ = [
    "fetch_abstracts",
    "fetch_full_text",
    "make_sandbox_tools",
    "pmc_locate",
    "pubmed_search",
]
