"""Did the agent cite PMIDs that actually exist in the corpus it fetched?

The cheapest useful grounding check in this repo, and it needs no judge model.

`sources/pubmed.py` writes every abstract it fetches to `data/abstracts/<pmid>.json` and
never evicts. So a PMID the agent genuinely read is on disk, and one it hallucinated is
not. That turns "is this citation real?" into a file-existence test.

The known limit, stated rather than papered over: the cache is shared across every run
ever made on this machine, so a PMID fetched by an *earlier* run also passes. This
catches invention, not misattribution — an agent citing a real paper for a claim that
paper doesn't make scores 1.0 here. `judge.py` is what covers that.

The prompt tells the model to cite PMIDs and to keep the `pmid` alongside each answer so
citations can't drift, which is what makes a zero here a real signal rather than a
formatting quibble.
"""

from __future__ import annotations

import re

from research_agent.paths import ABSTRACT_CACHE

# PubMed ids are 1-8 digits, but a bare 4-digit run in prose is almost always a year and
# a bare 2-3 digit one is a sample size. Requiring 7+ digits, or an explicit `PMID:`
# label, is what keeps the false-positive rate near zero on text full of numbers.
_LABELLED = re.compile(r"PMID[:\s]*(\d{1,8})", re.IGNORECASE)
_BARE = re.compile(r"\b(\d{7,8})\b")


def cited_pmids(text: str) -> set[str]:
    """Every PMID the answer appears to cite."""
    return {m for m in _LABELLED.findall(text)} | {m for m in _BARE.findall(text)}


def citations_exist(run, example) -> dict:
    """Fraction of cited PMIDs that are present in the host-side abstract cache."""
    answer = (run.outputs or {}).get("answer", "")
    pmids = cited_pmids(answer)

    if not pmids:
        # Not automatically a failure: the metadata-only questions ("which journals
        # publish the most...") are answerable without citing a single paper. Scored
        # None so it is excluded from the aggregate rather than dragging it down.
        return {
            "key": "citations_exist",
            "score": None,
            "comment": "no PMIDs cited — not applicable to this answer",
        }

    found = {p for p in pmids if (ABSTRACT_CACHE / f"{p}.json").exists()}
    missing = sorted(pmids - found)
    return {
        "key": "citations_exist",
        "score": len(found) / len(pmids),
        "comment": (
            f"{len(found)}/{len(pmids)} cited PMIDs in the fetch cache"
            + (f"; unverifiable: {missing[:10]}" if missing else "")
        ),
    }
