---
name: clinical-trials
description: Instructions on using ClinicalTrials.gov registry tools. Read before searching or fetching trials.

---

## Clinical trials

`tools.ctgovSearch` and `tools.ctgovFetch` reach the ClinicalTrials.gov registry,
**including the trials that never produced a paper**.

A registry record is a plan, not a result. Enrollment may be a target
(`enrollment_type: "ESTIMATED"`), completion dates on an unfinished trial are projections,
and `primary_outcomes` lists what will be measured, never a measured value. Say "planned"
when that is what the field means.

### Rate limit

**About one request per second, and there is no API key that raises it.** Three times
tighter than PubMed, so batching is not a preference here:

- `ctgovFetch` takes 200 ids in one request. Pass every id at once.
- `ctgovSearch` returns up to 5000 records in one call, paginating internally.
- Never loop one trial at a time, and never have a subagent fetch anything.

### Searching

Probe with `retmax: 0` first, the same discipline as `pubmedSearch`:

```js
const probe = await tools.ctgovSearch({
  condition: "HIV", intervention: "lenacapavir",
  filterAdvanced: "AREA[Phase]PHASE3", retmax: 0,
});
probe.count;  // too many? add a filter, but beware over-narrowing. zero? check spelling — see below
```

Search arguments, all optional but **at least one required** (an unfiltered search would
return the whole registry, so it is rejected):

| argument | matches |
|---|---|
| `condition` | condition or disease — `"HIV"` |
| `intervention` | drug, device or procedure — `"lenacapavir"` |
| `term` | free text across everything else |
| `title` | title or acronym — `"STEP 1"` |
| `sponsor` | sponsor or collaborator — `"Gilead Sciences"` |
| `status` | array of statuses, ORed — `["RECRUITING", "NOT_YET_RECRUITING"]` |
| `filterAdvanced` | an Essie expression, ANDed with the rest |

Then fetch the records:

```js
const res = await tools.ctgovSearch({
  condition: "HIV", filterAdvanced: "AREA[Phase]PHASE3",
  status: ["COMPLETED"], retmax: 500, sort: "EnrollmentCount:desc",
});
res.records;  // [{ nct_id, title, acronym, status, why_stopped, study_type, phases,
              //    enrollment, enrollment_type, lead_sponsor, sponsor_class, start_date,
              //    primary_completion_date, completion_date, last_updated, conditions,
              //    interventions, has_results, url }]
```

`sort` is `"@relevance"` or `"FieldName:asc|desc"` — `EnrollmentCount:desc`,
`LastUpdatePostDate:desc`, `StartDate:desc`.

**This is not PubMed syntax, and the two do not mix.** `[tiab]` and `[mesh]` mean nothing
here; `AREA[Phase]PHASE3` means nothing to `pubmedSearch`. Keep them apart.

`filterAdvanced` takes `AREA[FieldName]value` with AND/OR/NOT:

```
AREA[Phase]PHASE3 AND AREA[LeadSponsorClass]INDUSTRY
AREA[StartDate]RANGE[2020-01-01,2025-12-31]
AREA[LocationCountry]Japan
```

Enum values are exact and uppercase:

| field | values |
|---|---|
| status | `RECRUITING` `NOT_YET_RECRUITING` `ENROLLING_BY_INVITATION` `ACTIVE_NOT_RECRUITING` `COMPLETED` `SUSPENDED` `TERMINATED` `WITHDRAWN` `UNKNOWN` |
| `Phase` | `EARLY_PHASE1` `PHASE1` `PHASE2` `PHASE3` `PHASE4` `NA` |
| `StudyType` | `INTERVENTIONAL` `OBSERVATIONAL` `EXPANDED_ACCESS` |
| `LeadSponsorClass` | `INDUSTRY` `NIH` `FED` `OTHER_GOV` `NETWORK` `INDIV` `OTHER` |
| `DesignPrimaryPurpose` | `TREATMENT` `PREVENTION` `DIAGNOSTIC` `SCREENING` `SUPPORTIVE_CARE` `BASIC_SCIENCE` `HEALTH_SERVICES_RESEARCH` `DEVICE_FEASIBILITY` `OTHER` |

**Unlike PubMed, this API rejects bad input instead of quietly working around it.** A
wrong field, enum value, area name or sort is an error whose message names the offending
token — read it, fix that token, and search again. The flip side: there is no
`query_translation` to check and no spelling repair. Terms *are* synonym-expanded
(`condition: "heart attack"` and `"myocardial infarction"` return overlapping but
different sets) and you cannot see how, so `count` is the only handle you have. **A count
of 0 means zero, not "close enough" — check your spelling before concluding a trial does
not exist.**

There is no truncation wildcard, and a trailing `*` is not an error — it just returns 0
(`MK-347*` finds nothing, `MK-3475` finds 3080). Terms match only as written, so a drug
code and a fragment of one are unrelated searches. When you need a substring, or a concept
the registry has no single term for, search broadly and filter the records in JS — they
are in your heap, not your context.

`conditions` lists every condition a trial studies, comorbidities included, so a keyword or 
blocklist pass over `title` and `conditions` silently drops trials that do belong — a 
lenacapavir HIV trial registered under Kaposi's sarcoma, cytomegalovirus infection or 
"healthy participants" is still a lenacapavir HIV trial.

### Fetching trial detail

```js
const nctIds = res.records.map(r => r.nct_id);
const { records, missing, invalid } = await tools.ctgovFetch({
  nct_ids: nctIds, include: ["description", "eligibility"],
});
```

`include` adds field groups on top of the record above. Default is
`["description", "eligibility"]`.

| group | adds |
|---|---|
| `description` | `brief_summary`, `detailed_description` |
| `eligibility` | `eligibility_criteria`, `sex`, `min_age`, `max_age`, `std_ages`, `healthy_volunteers` |
| `design` | `allocation`, `intervention_model`, `primary_purpose`, `masking`, `arms`, `intervention_details` |
| `outcomes` | `primary_outcomes`, `secondary_outcomes` — planned measures, no values |
| `references` | `references` `[{pmid, type, citation}]`, `trial_pmids`, `result_pmids`, `background_pmids` |
| `mesh` | `condition_mesh`, `intervention_mesh` |
| `locations` | `countries` |
| `results` | `posted_results` — measured outcomes, participant flow, baseline characteristics and adverse events |

Cost per trial, roughly:

| rung | cost | use when |
|---|---|---|
| `ctgovSearch` record | ~175 tokens | always — this is how you build the shortlist |
| `+ description, eligibility` | ~350 tokens | the fan-out payload |
| `+ design, outcomes` | ~800 tokens | protocol-level questions about a few named trials |

Request `include: ["results"]` when you need posted measurements or adverse events;
`outcomes` alone contains planned measures. `has_results` indicates availability, not
what was retrieved. Missing posted results are not evidence of no effect or no publication.

Records over 24 KB automatically become file manifests (`storage: "file"`) with
`nct_id`, `title`, `status`, `has_results`, `url`, `path`, `index_path`, `bytes` and
`section_count`. The complete JSON and indexed sections are already staged under
`/workspace/retrieved/ctgov`, outside deliverables. Small records keep their usual shape.
Pass either representation to `trial-analyst` as below: it reads a file manifest's
index and only the relevant sections. Do not read the full record into root context.
Pass returned file manifests unchanged; never reconstruct their paths.
For counting or joining file-backed records, use Python via `tools.execute` on `path`;
return only the needed values. Files are restaged on every fetch, including cache hits;
if an old path is missing after sandbox replacement, fetch the trial again.

### Asking a question of many trials

Same shape as the abstract fan-out, with `trial-analyst`:

```js
const answers = await Promise.all(Object.values(records).map(async (t) => ({
  nct_id: t.nct_id, title: t.title, status: t.status,
  answer: await task({
    description: `Question: ${question}\n\nTrial record:\n${JSON.stringify(t)}`,
    subagentType: "trial-analyst",
  }),
})));
await tools.writeFile({
  file_path: "/workspace/trial-answers.json", content: JSON.stringify(answers),
});
answers.map(a => ({ nct_id: a.nct_id, answer: a.answer }));
```

Use Python over the records for anything countable — status breakdowns, enrollment
distributions, trials per sponsor. A registry record is mostly structured fields, so most
"how many" questions are `tools.execute`, not a fan-out.

### Joining the two

Three joins, all mechanical, and they are the reason both sources are wired in.

**Trial to papers** — `include: ["references"]` splits the record's citations three ways.
Use `trial_pmids`; it is the papers *about* this trial:
For file manifests, first extract `trial_pmids` from `path` with Python. A field absent
from a manifest is not an empty field in the underlying record. The inline case is:

```js
const { records: trials } = await tools.ctgovFetch({ nct_ids: nctIds, include: ["references"] });
const pmids = [...new Set(Object.values(trials).flatMap(t => t.trial_pmids || []))];
const { records: papers } = await tools.fetchAbstracts({ pmids });
```

- `trial_pmids` — publications reporting this trial. Mostly NLM's automatic back-links
  from PubMed's `[si]` field, so coverage is decent but not complete.
- `result_pmids` — the subset the sponsor explicitly flagged as the results publication.
  **Sparse: most sponsors never fill it in**, so an empty `result_pmids` is not evidence
  that nothing was published. Only ever a hint about which paper is the primary one.
- `background_pmids` — prior literature cited at registration. **Other people's papers.**
  Never count these as the trial's output.

Roughly a third of registered trials carry any linked publication at all. Absence in the
registry is weak evidence; confirm with a `[si]` search before reporting a trial as
unpublished.

**Papers to trial** — PubMed indexes NCT numbers under `[si]`:

```js
await tools.pubmedSearch({ term: "NCT03548935[si]" });   // papers reporting this trial
```

That is what makes "which registered trials have published results, and which have not"
answerable: search the registry, then check each trial both ways.

**Trial to MeSH** — `include: ["mesh"]` returns NLM's own descriptors
(`{id: "D009765", term: "Obesity"}`), which drop straight into `[mh]` and `[majr]`:

```js
const term = trial.condition_mesh.map(m => `"${m.term}"[mh]`).join(" OR ");
```

Cite trials by NCT number with the registry link, e.g.
[NCT03548935](https://clinicaltrials.gov/study/NCT03548935). When you cite both a trial
and its paper, give both ids.
