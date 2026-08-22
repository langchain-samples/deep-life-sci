"""ClinicalTrials.gov API v2 client and the two agent tools.

The registry answers what PubMed structurally cannot: what was *registered*, including
the trials that never produced a paper. See `docs/ctgov_concept.md` for why that is worth
wiring in, and `docs/ctgov_api_notes/` for the probe results behind every number here.

This module reads almost nothing like `pubmed.py`, for two reasons that invert:

1. **The API fails loudly.** An unknown field name, an unknown enum value, an unknown
   Essie area, a malformed id and an invalid sort are all HTTP 400 with a message naming
   the offending token. There is no `check_field_tags()` analog here and none is wanted —
   `pubmed.py` carries 80 lines of local tag validation only because
   `cancer[nosuchfield]` returns 5.7M hits and reports nothing. A 400 is a better error
   than anything this module could synthesise, so 4xx bodies are surfaced verbatim to the
   caller, which can fix the value and retry.
2. **The rate limit is the binding constraint.** Undocumented, and measured at roughly a
   10-token bucket refilling at 1/sec: 12 concurrent requests returned ten 429s, and the
   429 carries **no `Retry-After`**. That is 3x slower than unkeyed NCBI. Everything below
   is shaped by making per-trial fetching unnecessary rather than merely discouraged —
   `pageSize` reaches 1,000 and `filter.ids` takes 300 ids in one call, so a 1,000-trial
   corpus is four requests.

Three behaviours still return a *wrong answer rather than an error*, and each has a guard:

1. `pageSize` silently clamps at 1,000 (`pageSize=5000` -> HTTP 200, 1,000 studies).
2. Well-formed but nonexistent ids vanish from a `filter.ids` batch — 2 requested, 1
   returned, `totalCount: 1`, no missing list. Duplicates collapse the same way.
3. `countTotal` is opt-in *and first-page-only*: omit it and there is no `totalCount` key
   at all, and a page-2 response never carries one even when page 1 asked.

A fourth is a footgun rather than an API bug: an unfiltered `/studies` is legal and
returns all 599k studies, so a tool assembling params from optional arguments can scan the
whole registry by accident. `ctgov_search` refuses an empty query outright.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from datetime import date
from typing import Any

import httpx
from langchain_core.tools import tool

from research_agent.paths import CTGOV_CACHE
from research_agent.sources import cache_io

BASE_URL = "https://clinicaltrials.gov/api/v2"

# Verified: 300 ids in ~3.6 kB of URL returns 300 studies. There is no documented POST
# form, so URL length is the only ceiling — chunk at the same 200 `pubmed.py` uses.
ID_CHUNK = 200
# The API clamps here silently. Clamp explicitly so the caller is told.
MAX_PAGE_SIZE = 1000
# Not an API limit. Each page costs a second of the rate budget, and a search this wide
# means the query needed narrowing, not more pages.
MAX_RETMAX = 5000

NCT_RE = re.compile(r"^NCT\d{8}$")

STUDY_URL = "https://clinicaltrials.gov/study/{}"


class ClinicalTrialsError(RuntimeError):
    """A ClinicalTrials.gov call failed in a way the caller needs to know about.

    4xx messages from this API are precise and actionable (`Invalid value in parameter
    `overallStatus`: `NOTASTATUS``), so the body travels with the exception rather than
    being flattened into a status code. The agent can usually fix the value and retry.
    """


# --------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------

def _min_interval() -> float:
    """~1 request/second. Measured, not documented — NLM publishes no number.

    The ladder behind it: 3/5/8 concurrent all returned 200; 12 concurrent returned ten
    429s; 30 unpaced sequential requests (5.4 req/s) returned 20 errors; 12 requests at
    2.0 req/s returned 2 errors; 12 at 1.0 req/s returned none. There is no API key to
    raise it with.

    No separate concurrency cap is needed, unlike `pmc.py`'s semaphore: every request in
    this module passes through `_throttle`, which serialises them process-wide, so a
    `Promise.all` of ten searches is paced the same as ten sequential ones.
    """
    return 1.0


RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 4
# Higher than pubmed.py's 1.0. Against a bucket that refills at 1/sec, a one-second
# retry is just the next request in the burst that caused the 429.
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 30.0

_rate_lock = asyncio.Lock()
_last_call = 0.0


async def _throttle() -> None:
    global _last_call
    async with _rate_lock:
        wait = _min_interval() - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


def _backoff_delay(attempt: int) -> float:
    """Seconds before retry `attempt` + 1.

    No `Retry-After` branch, unlike `pubmed.py`: this API's 429 carries no such header
    (nor any `X-RateLimit-*`), so there is nothing to obey and the client's own backoff
    is the only thing between a fan-out and a dead run. Jittered because the callers
    that trip a 429 are concurrent by construction.
    """
    ceiling = min(RETRY_BASE_DELAY * 2 ** (attempt - 1), RETRY_MAX_DELAY)
    return random.uniform(ceiling / 2, ceiling)


async def _request(path: str, **params: Any) -> httpx.Response:
    """Call the API, retrying 429/5xx with jittered backoff. Raises on anything else.

    Retries are safe: every call here is a read. GET only — there is no POST form for
    long id lists, which is why `ID_CHUNK` exists.
    """
    payload = {k: v for k, v in params.items() if v is not None and v != ""}
    url = f"{BASE_URL}{path}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            await _throttle()
            resp = await client.get(url, params=payload)
            if resp.status_code == 200:
                return resp
            if resp.status_code in RETRY_STATUSES and attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(_backoff_delay(attempt))
                continue
            detail = " ".join(resp.text.split())[:300]
            raise ClinicalTrialsError(
                f"ClinicalTrials.gov returned HTTP {resp.status_code}"
                + (f" on all {attempt} attempts" if resp.status_code in RETRY_STATUSES else "")
                + (f": {detail}" if detail else "")
            )
    raise ClinicalTrialsError(f"exhausted {RETRY_ATTEMPTS} attempts against {path}")


def validate_nct_ids(ids: list[str]) -> tuple[list[str], list[str]]:
    """Split into (valid, invalid), uppercased and deduplicated in order.

    A malformed id is a 400 here rather than `efetch`'s silent tokenisation into the
    wrong record, so this is not the load-bearing guard `validate_pmids` is. It earns
    its place anyway: at one request per second, a round trip spent learning that item
    47 was a typo is expensive.
    """
    valid, invalid = [], []
    for raw in ids:
        s = str(raw).strip().upper()
        (valid if NCT_RE.match(s) else invalid).append(s)
    return list(dict.fromkeys(valid)), invalid


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------------
# Projection and flattening
# --------------------------------------------------------------------------------

# The lean projection, requested on every call. Server-side, so it costs nothing to ask
# for and the caller cannot accidentally receive a whole record. Measured at ~750 bytes
# (~185 tokens) per study against a corpus mean of 17,279 bytes — a 23x reduction.
#
# `pmc_locate` learned this the hard way: it returns 2,019 tokens per paper and the
# system prompt has to beg the model to project it down before returning it. Here the
# projection is unavoidable, which is the point.
CORE_FIELDS = (
    "NCTId", "BriefTitle", "Acronym", "OverallStatus", "WhyStopped",
    "StudyType", "Phase", "EnrollmentCount", "EnrollmentType",
    "LeadSponsorName", "LeadSponsorClass",
    "StartDate", "PrimaryCompletionDate", "CompletionDate", "LastUpdatePostDate",
    "Condition", "InterventionName", "HasResults",
)

# Opt-in field groups for `ctgov_fetch`. Names are the API's own, verified against the
# live endpoint — an unknown one is a 400, so a typo here breaks loudly at the first call.
INCLUDE_FIELDS: dict[str, tuple[str, ...]] = {
    "description": ("BriefSummary", "DetailedDescription"),
    "eligibility": (
        "EligibilityCriteria", "Sex", "MinimumAge", "MaximumAge", "StdAge",
        "HealthyVolunteers",
    ),
    "design": (
        "DesignAllocation", "DesignInterventionModel", "DesignPrimaryPurpose",
        "DesignMasking", "ArmGroupLabel", "ArmGroupType", "ArmGroupDescription",
        "ArmGroupInterventionName", "InterventionType", "InterventionDescription",
    ),
    "outcomes": (
        "PrimaryOutcomeMeasure", "PrimaryOutcomeTimeFrame", "PrimaryOutcomeDescription",
        "SecondaryOutcomeMeasure", "SecondaryOutcomeTimeFrame",
    ),
    "references": ("ReferencePMID", "ReferenceType", "ReferenceCitation"),
    "mesh": (
        "ConditionMeshId", "ConditionMeshTerm",
        "InterventionMeshId", "InterventionMeshTerm",
    ),
    "locations": ("LocationCountry",),
}

DEFAULT_INCLUDE = ("description", "eligibility")


def _dig(obj: Any, *keys: str) -> Any:
    """Walk nested dicts, returning None the moment anything is absent.

    Projection means most modules are simply not in the response, so absence is the
    normal case rather than an error.
    """
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _study_to_record(study: dict) -> dict | None:
    """Flatten one API study into a flat-ish record. The analog of `_summary_to_record`.

    Roughly 30% of the response bytes are module structure
    (`protocolSection.statusModule.startDateStruct.date` for a date string), which is
    unusable in the JS heap and pointless to carry. Optional keys appear only when their
    module came back, which is governed by the `include` groups the caller asked for.
    """
    proto = study.get("protocolSection") or {}
    nct_id = _dig(proto, "identificationModule", "nctId")
    if not nct_id:
        return None

    ident = proto.get("identificationModule") or {}
    status = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    sponsor = proto.get("sponsorCollaboratorsModule") or {}

    rec: dict[str, Any] = {
        "nct_id": nct_id,
        "title": ident.get("briefTitle"),
        "acronym": ident.get("acronym"),
        "status": status.get("overallStatus"),
        "why_stopped": status.get("whyStopped"),
        "study_type": design.get("studyType"),
        # A list: a trial can be registered as PHASE2 and PHASE3 at once.
        "phases": design.get("phases") or [],
        "enrollment": _dig(design, "enrollmentInfo", "count"),
        # ESTIMATED vs ACTUAL. Reading an estimate as a real number is the most common
        # way to be wrong about a registry record, so the discriminator travels with it.
        "enrollment_type": _dig(design, "enrollmentInfo", "type"),
        "lead_sponsor": _dig(sponsor, "leadSponsor", "name"),
        "sponsor_class": _dig(sponsor, "leadSponsor", "class"),
        "start_date": _dig(status, "startDateStruct", "date"),
        "primary_completion_date": _dig(status, "primaryCompletionDateStruct", "date"),
        "completion_date": _dig(status, "completionDateStruct", "date"),
        "last_updated": _dig(status, "lastUpdatePostDateStruct", "date"),
        "conditions": _dig(proto, "conditionsModule", "conditions") or [],
        "interventions": [
            i.get("name") for i in _dig(proto, "armsInterventionsModule", "interventions") or []
            if i.get("name")
        ],
        # Whether the 45,000-token results tier exists at all. True for 13.3% of studies.
        "has_results": study.get("hasResults"),
        "url": STUDY_URL.format(nct_id),
    }

    if desc := proto.get("descriptionModule"):
        rec["brief_summary"] = desc.get("briefSummary")
        rec["detailed_description"] = desc.get("detailedDescription")

    if elig := proto.get("eligibilityModule"):
        rec["eligibility_criteria"] = elig.get("eligibilityCriteria")
        rec["sex"] = elig.get("sex")
        rec["min_age"] = elig.get("minimumAge")
        rec["max_age"] = elig.get("maximumAge")
        rec["std_ages"] = elig.get("stdAges") or []
        rec["healthy_volunteers"] = elig.get("healthyVolunteers")

    if info := _dig(design, "designInfo"):
        rec["allocation"] = info.get("allocation")
        rec["intervention_model"] = info.get("interventionModel")
        rec["primary_purpose"] = info.get("primaryPurpose")
        rec["masking"] = _dig(info, "maskingInfo", "masking")

    if arms := proto.get("armsInterventionsModule"):
        if "armGroups" in arms:
            rec["arms"] = [
                {
                    "label": a.get("label"),
                    "type": a.get("type"),
                    "description": a.get("description"),
                    "interventions": a.get("interventionNames") or [],
                }
                for a in arms.get("armGroups") or []
            ]
        # Only when the caller asked for `design`; the core projection returns names only.
        if any(i.get("type") or i.get("description") for i in arms.get("interventions") or []):
            rec["intervention_details"] = arms.get("interventions")

    if outcomes := proto.get("outcomesModule"):
        rec["primary_outcomes"] = outcomes.get("primaryOutcomes") or []
        rec["secondary_outcomes"] = outcomes.get("secondaryOutcomes") or []

    if refs := _dig(proto, "referencesModule", "references"):
        rec["references"] = refs
        # Bridge 1, pre-extracted so the model never has to filter on `type` itself —
        # and the split matters, because the three types mean genuinely different things:
        #
        # * RESULT is the sponsor's own designation of the trial's publication, and it is
        #   sparse: sponsors simply do not fill it in. Measured over four phase 3 sets,
        #   RESULT appeared on 1/126, 3/82, 15/151 and 7/126 references.
        # * DERIVED is NLM's automatic back-link, harvested from the `[si]` secondary
        #   source ids on the PubMed record. It costs a sponsor nothing, so it is where
        #   the coverage actually is — the bulk of every sample above.
        # * BACKGROUND is prior literature the sponsor cited when registering. **These
        #   are other people's papers.** A union of all three passed to `fetch_abstracts`
        #   would answer "what has this trial published" with a reading list.
        typed = lambda *kinds: [  # noqa: E731
            r["pmid"] for r in refs if r.get("pmid") and r.get("type") in kinds
        ]
        rec["result_pmids"] = typed("RESULT")
        rec["trial_pmids"] = list(dict.fromkeys(typed("RESULT", "DERIVED")))
        rec["background_pmids"] = typed("BACKGROUND")

    derived = study.get("derivedSection") or {}
    if browse := derived.get("conditionBrowseModule"):
        # Bridge 3: NLM-assigned MeSH descriptors, which feed the `[mh]`/`[majr]` tags
        # `pubmed_search` already takes.
        rec["condition_mesh"] = browse.get("meshes") or []
    if browse := derived.get("interventionBrowseModule"):
        rec["intervention_mesh"] = browse.get("meshes") or []

    if locs := _dig(proto, "contactsLocationsModule", "locations"):
        rec["countries"] = sorted({loc["country"] for loc in locs if loc.get("country")})

    return rec


# --------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------

def _scan_cache(
    ids: list[str], groups: frozenset[str]
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Partition NCT ids into cached records and ones still needing a fetch.

    A single `to_thread` hop for the whole batch, as in `pubmed.py:_scan_cache`, and the
    same freshness/touch helpers so the TTL semantics cannot drift.

    A trial record is mutable in a way a PMID is not — status changes, results get
    posted, enrollment is revised — so the entry stores the field groups it was fetched
    with alongside `last_updated`. A hit requires the cached groups to *cover* what is
    being asked for; a request for more refetches and overwrites. Extra keys on a
    covering entry are returned as-is rather than projected away: they land in the JS
    heap, never the model's context, so trimming them would cost work and save nothing.

    Returns `(records, from_cache, to_fetch)`.
    """
    CTGOV_CACHE.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    from_cache: list[str] = []
    to_fetch: list[str] = []

    ttl = cache_io.ttl_seconds()

    for nct_id in ids:
        path = CTGOV_CACHE / f"{nct_id}.json"
        if cache_io.is_fresh(path, ttl):
            try:
                cached = json.loads(path.read_text())
                if groups.issubset(set(cached.get("groups") or [])):
                    records[nct_id] = cached["record"]
                    from_cache.append(nct_id)
                    cache_io.touch(path)
                    continue
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                pass  # corrupt or pre-schema entry, refetch
        to_fetch.append(nct_id)

    return records, from_cache, to_fetch


def _write_cache(records: dict[str, dict], groups: frozenset[str]) -> None:
    """Persist freshly fetched records. Blocking; call via `to_thread`."""
    payload_groups = sorted(groups)
    for nct_id, rec in records.items():
        (CTGOV_CACHE / f"{nct_id}.json").write_text(
            json.dumps({"groups": payload_groups, "record": rec}, indent=2)
        )


# --------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------

async def _paged_studies(params: dict[str, Any], limit: int) -> tuple[list[dict], int]:
    """Walk `nextPageToken` until `limit` studies or the results run out.

    `countTotal=true` is sent on the first request only and the answer is carried
    forward: the key is absent from every later page regardless of what was asked, so a
    caller reading it off page 2 would get None rather than an error.
    """
    studies: list[dict] = []
    total = 0
    token = None
    first = True

    while len(studies) < limit:
        page = dict(params)
        page["pageSize"] = min(MAX_PAGE_SIZE, limit - len(studies))
        if first:
            page["countTotal"] = "true"
        else:
            page["pageToken"] = token
        payload = (await _request("/studies", **page)).json()
        if first:
            total = int(payload.get("totalCount") or 0)
            first = False
        page_studies = payload.get("studies") or []
        studies.extend(page_studies)
        token = payload.get("nextPageToken")
        # A token with an empty page would otherwise spin; the API has not been seen to
        # do it, but at one request per second a loop is expensive to notice.
        if not token or not page_studies:
            break

    return studies, total


@tool
async def ctgov_search(
    condition: str | None = None,
    intervention: str | None = None,
    term: str | None = None,
    title: str | None = None,
    sponsor: str | None = None,
    status: list[str] | None = None,
    filter_advanced: str | None = None,
    retmax: int = 50,
    sort: str = "@relevance",
) -> dict:
    """Search the ClinicalTrials.gov registry and return matching trials with metadata.

    Covers registered studies including the many that never produced a paper, which is
    what PubMed cannot answer. Records come back as a lean projection (~185 tokens each),
    never whole study records.

    Use `retmax=0` to probe: it returns `count` without fetching any records, so it is
    cheap. Probe, narrow, then pull the records — the same discipline as `pubmed_search`.

    At least one search argument is required. An unfiltered search is legal at the API
    and would return all ~600,000 registered studies, so it is rejected here instead.

    Args:
        condition: Condition or disease, e.g. 'obesity'. Synonym-expanded.
        intervention: Drug, device or procedure, e.g. 'semaglutide'.
        term: Free text across all other fields.
        title: Title or acronym, e.g. 'STEP 1'.
        sponsor: Sponsor or collaborator name, e.g. 'Novo Nordisk'.
        status: Overall statuses to accept, ORed together. Valid values are
            RECRUITING, NOT_YET_RECRUITING, ENROLLING_BY_INVITATION,
            ACTIVE_NOT_RECRUITING, COMPLETED, SUSPENDED, TERMINATED, WITHDRAWN,
            UNKNOWN, AVAILABLE, NO_LONGER_AVAILABLE, TEMPORARILY_NOT_AVAILABLE,
            APPROVED_FOR_MARKETING, WITHHELD. An invalid value is rejected by the API.
        filter_advanced: An Essie expression, e.g. 'AREA[Phase]PHASE3' or
            'AREA[StartDate]RANGE[2020-01-01,2025-12-31]'. ANDed with everything above.
            This is NOT PubMed syntax — field tags like [tiab] are not valid here.
        retmax: Max trials to return, capped at 5000. 0 = count-only probe.
        sort: '@relevance', or 'FieldName:asc|desc' such as 'EnrollmentCount:desc' or
            'LastUpdatePostDate:desc'. An invalid value is rejected by the API.

    Returns:
        count: total matches in the registry (may far exceed the records returned)
        returned: how many records are in this response
        current_date: today, so a projected completion date can be told from a past one
        query_sent: the parameters actually sent, for the record
        warnings: ways the request was altered. Empty means it ran as asked
        records: list of {nct_id, title, acronym, status, why_stopped, study_type,
            phases, enrollment, enrollment_type, lead_sponsor, sponsor_class,
            start_date, primary_completion_date, completion_date, last_updated,
            conditions, interventions, has_results, url}
    """
    params: dict[str, Any] = {
        "query.cond": condition,
        "query.intr": intervention,
        "query.term": term,
        "query.titles": title,
        "query.spons": sponsor,
        "filter.advanced": filter_advanced,
        "filter.overallStatus": "|".join(status) if status else None,
    }
    params = {k: v for k, v in params.items() if v}
    if not params:
        raise ClinicalTrialsError(
            "ctgov_search needs at least one of condition, intervention, term, title, "
            "sponsor, status or filter_advanced. An unfiltered search would return all "
            "~600,000 registered studies."
        )

    warnings: list[str] = []
    probe_only = int(retmax) == 0
    limit = 1 if probe_only else max(1, min(int(retmax), MAX_RETMAX))
    if not probe_only and int(retmax) > MAX_RETMAX:
        warnings.append(
            f"retmax was reduced from {int(retmax)} to {MAX_RETMAX}. `count` is still "
            "the true total — narrow the query rather than paging further."
        )

    params["fields"] = ",".join(CORE_FIELDS)
    params["sort"] = sort

    studies, count = await _paged_studies(params, limit)
    records = [] if probe_only else [
        r for r in (_study_to_record(s) for s in studies) if r
    ]
    return {
        "count": count,
        "returned": len(records),
        # The interpreter's `Date.now()` is stubbed, so JS comparing a completion date
        # against "now" has no other source for it. Costs one field per call, in the JS
        # heap, where it reaches the model only if the model returns it.
        "current_date": date.today().isoformat(),
        # There is no `query_translation` equivalent on this API — synonym expansion is
        # applied and invisible — so the honest thing to echo is what was sent.
        "query_sent": {k: v for k, v in params.items() if k != "fields"},
        "warnings": warnings,
        # Every record, not a truncated head: these live in the interpreter's JS heap and
        # only what `eval` returns reaches the model.
        "records": records,
    }


@tool
async def ctgov_fetch(nct_ids: list[str], include: list[str] | None = None) -> dict:
    """Retrieve detailed registry records for a list of NCT ids, batched and cached.

    Pass every id you need in one call — 200 trials is one request, and the API allows
    only about one request per second. Never call this once per trial in a loop, and
    never have subagents call it; fetch here, then put the text into subagent prompts.

    Args:
        nct_ids: Ids like 'NCT03548935'. Anything else is rejected rather than sent.
        include: Field groups to add on top of the core record. Defaults to
            ['description', 'eligibility'] — the fan-out payload. Available:
            - description: brief_summary, detailed_description
            - eligibility: eligibility_criteria, sex, min_age, max_age, std_ages,
              healthy_volunteers
            - design: allocation, intervention_model, primary_purpose, masking, arms,
              intervention_details
            - outcomes: primary_outcomes, secondary_outcomes — what the protocol says
              will be measured, NOT the measured values
            - references: references [{pmid, type, citation}], plus the pmids split
              by type — trial_pmids (papers about this trial), result_pmids (the
              sponsor-designated primary, often absent), background_pmids (prior
              literature the sponsor cited, NOT this trial's own output)
            - mesh: condition_mesh, intervention_mesh — NLM MeSH descriptors
            - locations: countries

    Returns:
        records: {nct_id: {...core fields plus the groups requested}}. `enrollment_type`
            distinguishes ESTIMATED from ACTUAL; `has_results` says whether posted
            results exist. Free text runs ~440 chars for brief_summary and ~910 for
            eligibility_criteria, so this tier is safe for a large fan-out.
        missing: requested ids the registry returned nothing for
        invalid: inputs rejected as malformed
        from_cache: ids served from the local cache
    """
    valid, invalid = validate_nct_ids([str(i) for i in nct_ids])

    requested = list(include) if include is not None else list(DEFAULT_INCLUDE)
    unknown = [g for g in requested if g not in INCLUDE_FIELDS]
    if unknown:
        raise ClinicalTrialsError(
            f"unknown include group(s) {unknown}. Choose from "
            f"{sorted(INCLUDE_FIELDS)}."
        )
    groups = frozenset(requested)

    records, from_cache, to_fetch = await asyncio.to_thread(_scan_cache, valid, groups)

    fields = list(CORE_FIELDS)
    for group in sorted(groups):
        fields.extend(INCLUDE_FIELDS[group])

    for chunk in _chunks(to_fetch, ID_CHUNK):
        # `filter.ids` drops unknown ids and collapses duplicates without saying so, so
        # the request is never trusted to define the response — `missing` below is a diff
        # of what was asked against what came back.
        payload = (await _request(
            "/studies",
            **{"filter.ids": ",".join(chunk)},
            fields=",".join(fields),
            pageSize=MAX_PAGE_SIZE,
        )).json()
        fetched = {
            r["nct_id"]: r
            for r in (_study_to_record(s) for s in payload.get("studies") or [])
            if r
        }
        records.update(fetched)
        await asyncio.to_thread(_write_cache, fetched, groups)

    return {
        "records": records,
        "missing": [i for i in valid if i not in records],
        "invalid": invalid,
        "from_cache": from_cache,
    }
