"""External data sources: NCBI E-utilities, the PMC open-data bucket, and the
ClinicalTrials.gov registry.

The guards in `pubmed.py` and `pmc.py` are not boilerplate. Each one corresponds to a
verified API failure mode that returns a *wrong answer rather than an error* — PMID
tokenization, silent query rewriting, esummary's 500-UID cap answering HTTP 200, closed
articles served as a complete `<front>` with no `<body>`. See `docs/pubmed_api_notes/`
and `docs/pmc_api_notes/` for the probe results behind each.

`ctgov.py` needs far fewer of them, and for a reason worth knowing before editing it:
that API rejects bad input with a 400 naming the offending token instead of quietly
returning something plausible. What it needs instead is *rate discipline* — roughly one
request per second, measured, with no `Retry-After` to obey. See `docs/ctgov_api_notes/`.
"""

from research_agent.sources.ctgov import ctgov_fetch, ctgov_search
from research_agent.sources.pmc import fetch_full_text, make_sandbox_tools, pmc_locate
from research_agent.sources.pubmed import fetch_abstracts, pubmed_search

__all__ = [
    "ctgov_fetch",
    "ctgov_search",
    "fetch_abstracts",
    "fetch_full_text",
    "make_sandbox_tools",
    "pmc_locate",
    "pubmed_search",
]
