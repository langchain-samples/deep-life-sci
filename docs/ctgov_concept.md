**Concept addendum: trial registry data via ClinicalTrials.gov**

Extends [`concept.md`](concept.md) and [`pmc_concept.md`](pmc_concept.md). Those two
cover what PubMed knows about *papers*. This covers what the registry knows about
*studies* — including the ones that never produced a paper, which is the gap no amount of
PubMed tooling closes.

Principle, unchanged: **this is a demo. Keep everything as simple as possible.** The
measured facts behind every number here are in [`ctgov_api_notes/`](ctgov_api_notes/),
probed against the live API on 2026-08-20 (API v2.0.5, data timestamp 2026-08-19).

`demo_questions.md` currently lists ClinicalTrials.gov under what the agent explicitly
cannot reach. This is the proposal to remove that line.

---

## What actually changes

Three things, in descending order of how much they constrain the design. The ordering is
almost exactly inverted from PMC, which is the interesting part.

**1. The rate limit is the binding constraint, and it is tight.** PMC lifted the
constraint `concept.md` was built around — 88 concurrent S3 LISTs in 0.4s against
E-utilities' 3/sec. ClinicalTrials.gov puts it back, harder. Measured: **~1 request/second
sustained, bursts of ~10 tolerated, 12 concurrent returns ten 429s** — and the 429 carries
**no `Retry-After` header**, so there is nothing to obey and the client's own backoff is
the only thing between a fan-out and a dead run.

That is 3× slower than unkeyed NCBI and 10× slower than keyed. Every `S3_CONCURRENCY = 16`
instinct from `pmc.py` is wrong here.

The saving grace is that the API is built so you don't need concurrency: `pageSize` goes to
1,000 and `filter.ids` takes **300 ids in one verified call**. A 1,000-trial corpus is four
requests. The design job is to make per-trial fetching *impossible to express*, not merely
discouraged.

**2. The API fails loudly, which deletes most of the expected work.** This is the opposite
of PubMed, and it is worth stating plainly because it changes the effort estimate:

| input | PubMed | ClinicalTrials.gov |
|---|---|---|
| unknown field name | 5.7M hits, no warning anywhere | **400** `invalid field name: 'NoSuchField'` |
| unknown enum value | silently ignored, default used | **400** ``Invalid value in parameter `overallStatus` `` |
| unknown search area | tag dropped, all-fields search | **400** ``Unknown area name: `NoSuchArea` `` |
| malformed identifier | tokenised → *the wrong record* | **400** ``Item 2 in `filter.ids` has incorrect format`` |
| invalid sort | silently ignored | **400** ``Item 1 in parameter `sort` has incorrect format`` |
| typo in a search term | literal search, plausible hits | 200, `totalCount: 0` |

`pubmed.py` carries an 80-line `FIELD_TAGS` frozenset and `check_field_tags()` for exactly
one reason: `cancer[nosuchfield]` returns 5,675,880 hits and reports nothing. **None of
that machinery is needed here.** There is no `querytranslation` equivalent either, and none
is wanted — the query runs as written or it 400s.

Three traps survive and must be encoded; see [Guards](#guards-worth-writing).

**3. Record size is a cliff, and the expensive tier is enormous.** Measured on
`NCT03548935` (STEP 1, completed with results posted):

| tier | bytes | ≈ tokens |
|---|---|---|
| lean projection, 12 fields | **739** | **185** |
| whole `protocolSection` | 57,423 | 14,000 |
| `resultsSection` | **180,052** | **45,000** |
| entire record | **238,737** | **60,000** |

Within `resultsSection`: outcome measures 114,732 b, adverse events 56,917 b, baseline
4,438 b, participant flow 3,253 b.

For scale, `pmc_concept.md`'s headline number is a median full text of 10,056 tokens, and
the prompt already warns that *one paper* overflows `max_result_chars=40_000`. **One
trial's results section is 4.5× a whole paper.** Only **79,755 of 599,324 studies (13.3%)**
have results posted, so the expensive tier is rare — but when it appears it dwarfs
everything else in the run.

Corpus-wide: mean study 17,279 b, median 9,831 b, 95th percentile 49,785 b.

---

## Recommendation 1: three tools, and the search tool projects server-side

Mirroring the PubMed and PMC pairs rather than growing a new subsystem.

**`ctgov_search(...)`** — the workhorse. Takes `query_cond`, `query_intr`, `query_term`,
the structured filters, and a page size. Always requests a **lean `fields=` projection**
and always flattens host-side, so it returns ~185 tokens/trial and *cannot* return a raw
record. Paginates internally via `pageToken`. Supports a `retmax=0`-style probe through
`countTotal=true`, which matches the discipline the prompt already teaches for
`pubmed_search`.

This is the one real difference from `pmc_locate`, and it is deliberate. `pmc_locate`
returns 2,019 tokens/paper and the prompt has to beg the agent to project it down before
returning it — a correction `pmc_concept.md` records after the fact. Here the projection
is free, server-side, and unavoidable. **Do not repeat that mistake by returning whole
studies and trusting the prompt.**

**`ctgov_fetch(nct_ids, include=[...])`** — batch by `filter.ids`, chunked at 200 for URL
headroom (300 verified working at ~3.6 kB of URL; there is no documented POST form). This
is the fan-out payload, and the free-text fields are comfortably abstract-sized — measured
over 40 interventional obesity trials:

| field | present | median | max |
|---|---|---|---|
| `briefSummary` | 40/40 | 441 chars | 2,849 |
| `eligibilityCriteria` | 40/40 | 908 chars | 5,170 |
| `detailedDescription` | 29/40 | 1,197 chars | 8,924 |

An eligibility criteria block is roughly four abstracts. A 200-trial fan-out is entirely
safe, and it is the same shape as the existing one.

**`ctgov_results(nct_ids, outcomes=[...])`** — the 45,000-token tier, deliberately
separate and deliberately narrow. **I would cut this from Phase 1** (see
[Phasing](#phasing)): almost every question that reaches for it is better served by the
*paper* the trial links to, which the bridge below gives you for free.

### Everything needs flattening

The API returns deeply nested modules:

```json
{"protocolSection": {"identificationModule": {"nctId": "NCT06909006", "briefTitle": "..."},
 "statusModule": {"overallStatus": "NOT_YET_RECRUITING",
                  "startDateStruct": {"date": "2025-10"}}}}
```

Unusable in JS-heap ergonomics, and ~30% of the bytes are structure. A
`_study_to_record()` is the direct analog of `pubmed.py:_summary_to_record()` — same job,
same place in the pipeline, more nesting to walk. It is the bulk of the new module's line
count and it is entirely mechanical.

## Recommendation 2: rate discipline is the architecture

`concept.md` named NCBI's 3/sec as *"the constraint driving the above"*. State the CTG
number the same way, because it is stricter:

```
_min_interval()  ->  1.0s      (pubmed.py: 0.34s unkeyed, 0.11s keyed)
concurrency      ->  <= 8      (pmc.py: 16)
Retry-After      ->  never sent; _backoff_delay always falls through to the jittered ceiling
```

The existing **"subagents do no I/O"** invariant is not merely preserved — it becomes
load-bearing in a harder way. Fifty subagents hitting NCBI collect 429s; **twelve** hitting
CTG collect 429s. The measured ladder:

| pattern | result |
|---|---|
| 3 / 5 / 8 concurrent | all 200 |
| **12 concurrent** | **10× 429, 2× 200** |
| 30 sequential, unpaced (5.4 req/s) | 10 ok, **20 errors** |
| 12 requests at 2.0 req/s | 10 ok, 2 errors |
| 12 requests at 1.0 req/s | **12 ok, 0 errors** |

Shared HTTP machinery is a judgment call. `_throttle`, `_backoff_delay`, `RETRY_STATUSES`
and the retry loop in `pubmed.py:_request` are ~60 lines CTG needs almost verbatim, but
with a different interval, no `Retry-After`, and a different base URL. `pmc.py` already
chose to duplicate rather than share. **Follow that precedent** — one more copy is cheaper
than a three-way abstraction over three genuinely different rate-limit regimes.

## Recommendation 3: the bridges are the reason to do this

A second search box is worth little. Three verified joins are worth a lot, and they are
what make the registry more than a parallel corpus.

**Trial → papers.** `protocolSection.referencesModule.references[]` carries
`{pmid, type, citation}` with `type` ∈ `BACKGROUND` / `RESULT` / `DERIVED`. For
`NCT03548935` the `RESULT` entry is PMID 33567185 — *Wilding et al., Once-Weekly
Semaglutide in Adults with Overweight or Obesity*, NEJM 2021. That is row one of the
`semaglutide-weightloss-boxplot` eval rubric, arrived at from the other direction.

**Papers → trial.** PubMed's `[si]` (secondary source id) tag indexes NCT numbers, and it
is already in the prompt's field-tag table under its generic name:

```
NCT03548935[si]        -> 11 PMIDs
NCT04255433[si]        ->  4 PMIDs
clinicaltrials.gov[si] -> 192,339 papers
```

**Trial → MeSH.** `derivedSection.conditionBrowseModule` / `interventionBrowseModule`
carry NLM-assigned MeSH descriptors with ids and ancestors (`D009765 Obesity`,
`D050177 Overweight`, …), which feed straight into the `[mh]` / `[majr]` tags the prompt
already teaches.

Those three unlock a question class the agent currently cannot answer at all:

> *Which registered phase 3 trials of semaglutide for obesity have published results, and
> which have not?*

Registry-only questions (recruiting counts by sponsor, enrollment distributions, endpoint
drift between protocol and paper) are the second class. Both are genuinely new; neither is
reachable from PubMed alone.

## Recommendation 4: the escalation ladder is the prompt's job, again

The prompt already carries the real design work, and it is already 504 lines. Realistically
**+120 lines**, covering:

1. **When to reach for the registry at all.** The likeliest failure mode is the model
   searching CTG for questions PubMed answers better, or the reverse. This needs to be as
   sharp as the existing `eval`-vs-`execute` line, which took real tuning. **This is the
   only item here that is behaviour rather than documentation, and it is where the budget
   goes.**
2. **Essie query syntax**, a second query language next to the boolean/field-tag table.
   Its search areas do synonym expansion (`isSynonyms: true` on `BriefTitle`,
   `OfficialTitle`, `Condition`) — `query.cond=heart attack` returns 3,920 against
   `myocardial infarction`'s 3,864, overlapping but not identical. Same shape of caveat as
   the `[mh]` row already carries, and it deserves the same treatment.
3. **The enum vocabularies.** `OverallStatus` has 14 values, `Phase` 6, `StudyType` 3. A
   wrong value 400s rather than lying, which is recoverable, but it costs a round trip.
4. **A hard cap on results-tier fetches**, at 45,000 tokens each.
5. **The three bridges above**, as reference JS snippets — the way every other capability
   in this prompt is taught.

The cost ladder to state, per trial:

| rung | ≈ tokens | use when |
|---|---|---|
| `/stats/field/values` aggregate | ~100 total | "how many phase 3 trials of X" — no records at all |
| lean search record | 185 | always — this is how you build the shortlist |
| description + eligibility | 350 | the fan-out payload |
| whole `protocolSection` | 14,000 | one named trial, protocol-level question |
| `resultsSection` | 45,000 | one or two trials, and only if the paper won't do |

That first rung has no PubMed analog and is worth teaching explicitly:
`/stats/field/values?fields=Phase&types=ENUM` returns the whole distribution
(NA 234,801 / PHASE2 89,812 / PHASE1 65,443 / PHASE3 49,682 / PHASE4 35,674 /
EARLY_PHASE1 6,446) in one ~400-byte response.

## Recommendation 5: one new leaf, `trial-analyst`

Wrapped by the existing `analyst_leaf()` in `agent.py`, which means it inherits the
`tools: []` + `FilesystemMiddleware(tools=["read_file"])` narrowing for free. No wiring
changes.

Reusing `abstract-analyst` would work and cost nothing. Write the new one anyway — it is
~40 lines, and a registry record is a genuinely different reading task from prose. The
failure modes differ: confusing anticipated enrollment with actual, reading an estimated
completion date as real, treating an arm description as a result, reporting a planned
outcome measure as an observed one. `abstract-analyst`'s *"quote the decisive phrase"*
rule does not fit a record whose answer is usually a field value.

## Recommendation 6: caching, with the one real difference

`cache_io.py` and the `data/abstracts/{pmid}.json` scheme transfer directly. One
difference matters: **a published paper is immutable; a trial record is not.** Trials
change status, post results, and revise enrollment. `NCT04255433`'s `lastUpdatePostDate`
is `2026-07-08`.

The freshness check is cheap and batchable, which is better than PubMed's situation:

```
GET /studies?filter.ids=<300 ids>&fields=NCTId,LastUpdatePostDate,OverallStatus,HasResults
```

~40 bytes per trial, one request. So a two-phase cache is available: compare
`lastUpdatePostDate` against the cached value, refetch only what drifted.

**Do the simple version first.** Cache with a 24h TTL and store `lastUpdatePostDate` in the
entry; add the two-phase check when something demands it. `pubmed.py` already has the
schema-migration precedent (`if "pmcid" in cached`) for evolving this later without a full
cache burn.

New path in `paths.py`: `CTGOV_CACHE = DATA_DIR / "trials"`.

## Guards worth writing

Three, against the API's six loud failures. Each is the *silent wrong answer* class, which
is the bar `sources/__init__.py` sets for a guard earning its place.

1. **`pageSize` silently clamps at 1,000.** `pageSize=5000` → HTTP 200, 1,000 studies, no
   warning. Same class as `esearch`'s `retmax` clamp, same fix: clamp explicitly so the
   caller knows, and surface `totalCount` as the honest number.
2. **Nonexistent-but-well-formed ids vanish.** `filter.ids=NCT03548935,NCT99999999`
   returns `totalCount: 1`. Diff requested against returned into a `missing` list, exactly
   as `fetch_abstracts` does. (`filter.ids` also deduplicates silently: 60 copies of one id
   returned one study.)
3. **`countTotal` is opt-in and first-page-only.** Omit it and the response has no
   `totalCount` key at all; request page 2 via `pageToken` and it is absent regardless. A
   caller that reads `totalCount` off a later page gets `None`, not an error.

One non-guard worth a comment rather than code: `/stats/size` reported 599,324 studies
while a live unfiltered search reported 599,549 in the same session. The stats endpoints
lag slightly. Fine for a denominator, wrong for a count the user will quote.

## Phasing

**Phase 1 — search and fetch.** `ctgov_search` + `ctgov_fetch`, `trial-analyst`, host-side
cache with TTL, prompt segment, rate discipline. No results section, no cross-source
joins in the prompt beyond mentioning `referencesModule.pmid` exists. This alone answers
the registry-only question class.

**Phase 2 — the bridges.** Prompt snippets for all three joins, and the eval seed that
depends on them. This is the phase that justifies the integration; it is almost entirely
prompt work, because the data is already in what Phase 1 returns.

**Phase 3 — `ctgov_results`, if an eval demands it.** 13.3% of trials, 45,000 tokens each,
and usually strictly worse than the linked paper. Do not build it speculatively.

## Evals

`default.yaml` already has `D3 clinical medicine and trials` as a domain and
`fmt-cdiff-placebo-trials` as a seed. Two or three additions:

- **A registry-only question** — recruiting phase 3 trials of X by sponsor. The taxonomy's
  *"pin the dates"* rule matters doubly: trial status changes weekly, so the rubric needs
  a frozen bound (an `AREA[LastUpdatePostDate]RANGE[...]` filter in the question itself, or
  an as-of date in the rubric) or the expected value decays within a month.
- **A cross-source question** — registered phase 3 trials of X that have published results
  versus those that have not. Exercised by nothing currently in the dataset, and the single
  best demonstration of why both sources are wired in.
- **Revisit `semaglutide-weightloss-boxplot`.** It has 19 hand-curated PMIDs with effect
  sizes and is, in substance, a trial-registry question the agent is presently forced to
  answer through publications. With the registry available there is a second trajectory to
  the same rubric. Check whether the rubric still discriminates, or whether it now rewards
  a shortcut that skips reading the papers.

`runner.py`'s `RunResult` needs no change. Root-context size is exactly the number that
would catch a bad projection.

## Files touched

| file | change |
|---|---|
| `research_agent/sources/ctgov.py` | **new**, ~450 lines |
| `research_agent/sources/__init__.py` | export the tools |
| `research_agent/paths.py` | `CTGOV_CACHE` |
| `research_agent/agent.py` | entries in `tools=[]`, `ptc=[]`, `subagents=[]` |
| `research_agent/prompts/system.py` | ~+120 lines |
| `research_agent/prompts/subagents.py` | `TRIAL_ANALYST` |
| `research_agent/prompts/__init__.py` | re-export |
| `evals/datasets/default.yaml` | 2–3 seeds |
| `docs/demo_questions.md` | remove the ClinicalTrials.gov exclusion |
| `CLAUDE.md` | the rate-limit invariant, at minimum |

**Not touched:** `sandbox.py`, `models.py`, `middleware/`, `runner.py`, `graph.py`,
`cli.py`. CTG is a pure host-side HTTP source with no sandbox involvement — no figures, no
binary staging, no `make_sandbox_tools`-style factory. Structurally the easy kind of
source, which is most of why the estimate is a day of implementation plus a day of prompt
tuning against evals rather than the multi-phase build PMC needed.

## Explicitly out of scope

- **CSV output.** `format=csv` uses UI column labels, not API field paths —
  `fields=NCTId&format=csv` is a 400. The agent writes its own spreadsheets in the sandbox
  anyway.
- **The two-phase freshness check.** TTL first; see Recommendation 6.
- **`ctgov_results` in Phase 1.** See Phasing.
- **The WHO ICTRP and EU CTR registries.** Real coverage gaps for non-US trials, but a
  third registry for a demo that has not yet shipped the first one.
- **Bulk dataset download.** There is no working `/studies/download`; 599k studies is not
  a demo-scale corpus regardless.

## Risks

- **The rate limit is undocumented.** NLM publishes no number. Everything in
  Recommendation 2 is measured, not promised, and could move without notice. Build the
  retry path as if the limit will tighten, and re-probe before any demo that fans out.
- **Trial records mutate under the eval set.** Unlike PMIDs, an NCT record's status and
  enrollment change. Every trial-based rubric needs a date bound or it rots. This is a
  standing curation cost the PubMed seeds do not carry.
- **Two query languages in one prompt.** PubMed field tags and Essie `AREA[...]`
  expressions look similar and are not interchangeable. The most likely regression is
  cross-contamination — `AREA[Phase]PHASE3` sent to `pubmed_search`, or `[pt]` sent to
  `ctgov_search`. Both 400 or return zero rather than lying, so it is recoverable, but it
  is worth an explicit line in the prompt and a look in the eval trajectories.
- **`resultsSection` is a context bomb.** 45,000 tokens, and the model has no way to know
  that before asking. If Phase 3 ever happens, the tool must cap or slice by outcome
  measure rather than trusting the prompt.
