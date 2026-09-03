# sources/

Every module here has a docstring listing the verified API failure modes its guards exist for,
and `docs/pubmed_api_notes/`, `docs/pmc_api_notes/`, `docs/ctgov_api_notes/` (all gitignored)
hold the probe results — the ctgov notes ship a `probe.py` that reproduces every measurement,
sleeps included. These are the rules those notes add up to.

- **The guards are not boilerplate.** Each matches a failure that returns a *wrong answer
  rather than an error*: PMID tokenization, silent query rewriting, esummary's 500-UID cap
  returning HTTP 200, unguessable PMC object version suffixes, `itertext()` fusing body
  paragraphs. Don't simplify them away. PubMed field tags are validated locally against the
  documented set, so supporting a new one means extending that list.

- **ClinicalTrials.gov is the tightest rate limit in the repo, and it is undocumented.**
  Measured at roughly a 10-token bucket refilling at ~1 req/sec — 12 concurrent requests
  returned ten 429s — and **the 429 carries no `Retry-After`**, so client-side backoff is the
  only thing between a fan-out and a dead run. No API key raises it. The design response is to
  make per-trial fetching unnecessary rather than merely discouraged: `pageSize` reaches 1,000
  and `filter.ids` takes 300, so a 1,000-trial corpus is four requests.

- **`ctgov.py` inverts `pubmed.py`'s validation story on purpose.** That API answers a bad
  field, enum, area, sort or id with an HTTP 400 naming the token, so 4xx bodies are surfaced
  verbatim and there is **no local `check_field_tags()` analog — do not add one.** Three
  behaviours still return a wrong answer rather than an error and each has a guard: `pageSize`
  clamps silently at 1,000, unknown ids vanish from a `filter.ids` batch with no missing list,
  and `countTotal` is opt-in and first-page-only. A fourth is a footgun rather than a bug — an
  unfiltered `/studies` is legal and returns all ~600k studies, which is why `ctgov_search`
  refuses an empty query.

- **Registry reference types are not interchangeable.** `referencesModule` mixes RESULT
  (sponsor-designated, and sparse — 1 of 126 references across one measured phase 3 set),
  DERIVED (NLM's automatic back-link from PubMed's `[si]` field, where the coverage actually
  is) and BACKGROUND (prior literature the sponsor cited — *other people's papers*).
  `_study_to_record` splits them into `result_pmids`, `trial_pmids` and `background_pmids` so
  the model never has to filter on `type`; collapsing them back into one list would answer
  "what has this trial published" with a reading list.

- **`_http.py` holds the shared pacing/backoff mechanism, but each caller keeps its own
  `Throttle`.** NCBI and ClinicalTrials.gov meter independently, so a shared `_last_call`
  would make a PubMed search delay a registry fetch for nothing. `ctgov.py`'s `Throttle`
  serialises every request in the process, which is why that module needs no concurrency
  semaphore the way `pmc.py` does. Each module also keeps its own `_request` — the POST
  branch, 4xx handling and exception types genuinely differ. `pmc.py` uses none of `_http.py`;
  S3 caps concurrency with a semaphore instead.

- **`web.py` has no API and therefore no guards, cache or throttle — that is deliberate.**
  Both providers ship web search as a *server-side* tool that runs inside the model, so the
  module is a bound tool spec plus a digest parser. It keeps no cache because a label
  revision or approval date is exactly where a stale hit is a wrong answer, and no throttle
  because the metered requests are not ours; `_SEMAPHORE` caps concurrency for cost, not
  politeness. The reason it is a PTC tool rather than a search tool bound to the root model
  is measured: server-side results arrive as content blocks in the assistant message,
  outside `eval` and outside PTC, and one question cost 27.9k input tokens that way.

- **`cache_io.py` is the I/O floor: every blocking call goes through `asyncio.to_thread`**, and
  entries expire on the *idle* `paths.IDLE_TTL_SECONDS`, the same window that reaps a thread's
  sandbox. Note `MISSING` vs `None`: a resolved-to-nothing PMCID is cached as literal `null`,
  a real answer meaning "this article has no objects".
