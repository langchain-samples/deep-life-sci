**Concept addendum: full-text papers via PubMed Central**

Extends [`concept.md`](concept.md), which closes with *"To start with, we'll just do
abstract, since that's always just text. Papers, with their associated multimodal and
multi-file-type data will come later."* This is later.

Principle, unchanged: **this is a demo. Keep everything as simple as possible.** The
measured facts behind every number here are in [`pmc_api_notes/`](pmc_api_notes/).

---

## What actually changes

Three things, in descending order of how much they constrain the design. Only the third
is the one people expect.

**1. Token economics invert the fan-out.** An abstract is ~250 tokens. The median PMC
full text is **10,056 tokens** — 40× larger. The current design's signature move, 200
papers → 200 concurrent subagents in one `Promise.all`, costs ~50k tokens on abstracts and
**~850k tokens** on the 77 full texts we measured. Full text therefore cannot be the
default path; it is an *escalation on a shortlist*. This is the single most important
design consequence and it is not a multimodal problem at all.

**2. Availability is a cliff, and it fails silently.** Only **53% of PMIDs** end up with
retrievable full text (61% have a PMCID; 87% of those are actually available). Worse,
`efetch db=pmc` reports a closed article as **HTTP 200 with a complete `<front>` — title,
authors, abstract — and no `<body>`**, with no error anywhere in the response. That is the
identical failure class to the PubMed traps `pubmed.py` already guards
(`efetch` returning the wrong paper, `esearch` silently rewriting a query): the API hands
back something plausible rather than an error. "No full text" has to be a first-class
answer the agent reports, not a gap it papers over.

**3. Multimodal data is not in the E-utilities API at all.** JATS gives you
`<graphic xlink:href="fendo-09-00124-g001.jpg"/>` — a bare filename. Every documented way
to resolve it to a URL 404s, and the one service that did resolve it is **withdrawn on or
after 2026-08-24, seventeen days from now**.

## Recommendation 1: don't add a second E-utilities call — use the S3 bucket

The obvious move is `efetch db=pmc`. Don't. Use the **PMC Cloud Service** open-data bucket:

```
https://pmc-oa-opendata.s3.amazonaws.com/PMC{id}.{version}/
```

No auth, no API key, and **no NCBI rate limit** — 88 concurrent LIST requests completed in
**0.4 seconds**, against E-utilities' 3/sec. For an architecture whose whole premise is
fan-out, that is decisive. `concept.md` names the 3-req/sec limit as *"the constraint
driving the above"*; on this path the constraint mostly lifts.

Each per-article prefix holds exactly what we need and nothing we have to derive:

| object | size | why we want it |
|---|---|---|
| `.json` | **924 b** | `is_pmc_openaccess`, `is_manuscript`, `is_retracted`, `license_code`, `citation` |
| `.txt` | ~39 kb | PMC's own JATS→text rendering — **no extractor to write** |
| `.xml` | ~95 kb | JATS, when we need section structure |
| `*.jpg` | ~60 kb ea. | the figures, the only route that works |
| `*.xlsx` | varies | supplementary data tables |
| `.pdf` | 0.5–2 MB | ignore it; we have better |

And the LIST call **is** the availability check: 77 of 88 PMCIDs had objects, exactly the
77 `efetch` gave a body for. Zero objects means not available — one cheap request instead
of an 11.5 MB download.

Two traps to encode in the client, both verified:

- **Never hardcode the `.1` version suffix.** Observed versions: 1 (×73), 2 (×2), and
  **319** (×2). Read the version off the LIST response.
- **Never build against `oa.fcgi`, `oa_package/`, `oa_comm/`, or `deprecated/` paths.**
  The `deprecated/` prefixes still return 200 *today* and vanish on the 24th — code
  written against them passes testing this week and breaks in two.

`esearch`/`esummary`/`efetch db=pubmed` stay exactly as they are. And `pubmed_search`
gets the PMCID for **free** — `esummary`'s `articleids` already contains it, so
availability triage costs zero extra requests.

## Recommendation 2: three tools, and the middle one defaults to *not* the whole paper

Mirroring the existing two thin Python tools rather than growing a new subsystem:

**`pmc_locate(pmids)`** — the triage tool, and the one that makes the cost story work. One
LIST + one 924-byte sidecar per paper. Returns availability, `license_code`,
`is_retracted`, `is_manuscript`, **the section titles**, **the figure labels and caption
lengths**, and the body's character count — but **no body text**. This is what lets the
agent decide where to spend tokens before spending any.

**`fetch_full_text(pmcids, sections=None, include_captions=True)`** — the body, sliced.
`sections=["methods"]` costs 3.4k tokens instead of 10k. Slicing must degrade to the whole
body rather than returning nothing: **20 of 77 papers have no recognisable methods
section** (mostly reviews, which have none). Normalise headings for case and numeric
prefixes.

**`fetch_figures(pmcid, figure_ids)`** — image bytes. Where they *go* is the design
question; see below.

Supplementary `.xlsx` files are close to free given the existing `execute` surface — write
one into the sandbox and `pandas` reads it — so `fetch_supplementary` is a natural fourth,
but it belongs in a later phase.

## Recommendation 3: the escalation ladder is the prompt's job

The prompt already carries the real design work (the two-step probe-then-fetch search
discipline, the `eval`-vs-`execute` line). Full text needs one more segment, framed as a
**cost ladder** with measured numbers, because the model will otherwise reach straight for
the whole paper:

| rung | ≈ tokens/paper | use when |
|---|---|---|
| abstract | 250 | always — this is how you build the shortlist |
| captions + section titles | 1,400 | "which papers have a survival curve?" |
| one section | 3,400 | "what mouse strain did they use?" |
| whole body | 10,000 | genuinely needs the whole argument |
| figure image (vision) | a vision call | the answer is only in the picture |

The rule to state plainly: **abstracts triage the corpus, full text answers on ≤10–20
papers.** A 200-paper full-text fan-out is a bug, not thoroughness. And `pmc_locate`
before `fetch_full_text`, always.

## Recommendation 4: every tool feeds a fan-out, and delegation is the default

All three tools are designed to pipe into fanned-out subagents, exactly as
`fetch_abstracts` does today. The main agent keeps discretion to read something itself,
but the prompt should push hard toward delegating, and the reason is quantitative.

**Delegation is worth ~40× more on full text than on abstracts**, because the root
agent's context is *cumulative*. An abstract read directly costs 250 tokens once. A full
text read directly costs 10k tokens **on this turn and every subsequent turn of the
conversation**, because it stays in the message history. Fifteen papers read directly is
150k tokens of permanent context; the same fifteen delegated is 150k tokens spread across
fifteen throwaway Haiku contexts, with only ~200 tokens of structured answer each coming
back. Same work, and the root context stays small enough to keep reasoning well.

There is also a hard mechanical limit that makes this non-optional:
`max_result_chars=40_000` on the interpreter, against a **median full text of 40,227
chars**. Returning even one whole paper as an `eval` result gets it truncated. The
guardrail is already there — it just fails silently, so the prompt has to say why.

The per-tool split:

| tool | who consumes it | why |
|---|---|---|
| `pmc_locate` | **main agent** | it's the triage decision |
| `fetch_full_text` | **subagents** — `full-text-analyst` | 10k tokens/paper |
| `fetch_figures` | **subagents** — `figure-analyst` | image blocks |

> **Corrected during implementation.** I estimated `pmc_locate` at ~50 tokens/paper. Its
> actual return is **2,019 tokens/paper** — 155,496 across 77 papers — because it carries
> every figure caption in full (1,455 of those 2,019). The *decision* it supports is
> still tiny; the payload isn't. This doesn't change the architecture, since PTC keeps
> results in the JS heap and only returned values reach context, but it does mean the
> prompt must tell the agent to project the result down before returning it. The
> projected triage summary measures 40 tokens/paper.
>
> Same correction, sharper: `fetch_full_text` for a **single** paper is ~10,645 tokens
> against `max_result_chars=40,000` (≈10,000 tokens). It isn't that a corpus overflows
> one result — one paper does.

`full-text-analyst` is nearly free to build: it's `abstract-analyst` with a different
system prompt and a larger input. The existing fan-out snippet already works unchanged —
substitute the section text for `r.abstract`.

Two things worth adding to the fan-out that the abstract path doesn't currently use:

- **`responseSchema`** on `task()`, as `concept.md` intended. Confirmed supported from JS,
  with real limits to design within: **≤4,096 bytes serialised, ≤5 levels deep, ≤32
  properties total.** A flat `{pmid, answer, quote, section, confidence}` fits easily.
  Structured answers matter more here than for abstracts, because full-text answers need
  provenance — *which section, which sentence*.
- **Escalation inside one `eval`.** Because the interpreter is real code, the shortlist and
  the escalation can be one call: fan out over abstracts, filter on the structured answers,
  then fan out over full texts for only the survivors. No round trip through the
  orchestrator, and the abstracts never enter root context at all.

The one case where the main agent *should* read directly: a single paper the user named
explicitly ("what does the Liu 2021 paper say about off-target rates?"). One paper, one
question, no fan-out to amortise — delegating just adds a hop.

## Recommendation 5: figures — `read_file` already does this, no custom tool needed

I initially proposed a custom `ask_about_figure` Python vision tool here, on the
assumption that images couldn't reach a subagent. **That was wrong**, and the corrected
design is much simpler.

`task()` genuinely does take a string only — verified in
`langchain_quickjs/_subagent.py`, which passes `{"description": str, "subagent_type": ...}`
and nothing else. So an image block cannot go *through* `task()`. But it doesn't need to:
**deepagents' `read_file` returns multimodal content blocks for images natively**
(`middleware/filesystem.py`: *"Images (`.png`, `.jpg`, etc.), audio, video, and PDFs
return multimodal content blocks"* — it emits a synthetic `HumanMessage` carrying the
media, flagged `read_file_media_result`).

So the figure fan-out is the existing architecture with nothing new bolted on:

1. `fetch_figures` writes the JPEGs into the sandbox (bytes go host → sandbox, never
   through the interpreter or model context).
2. `task()` dispatches a `figure-analyst` per figure, with the **path**, the caption, and
   the question in the string description.
3. The subagent calls `read_file` on the path and *sees the figure*.

And `abstract-analyst` **already has exactly the middleware this needs** — `agent.py:111`
gives it `FilesystemMiddleware(backend=backend, tools=["read_file"])`, because
`read_file` is the floor that `FilesystemMiddleware` won't let you omit. The capability
has been sitting there unused.

Two useful properties of the fallback behaviour, both read off the source:

- Unsupported block types degrade to a **visible text placeholder** (`"[read_file: … was
  not attached because this model does not support image content]"`), not a silent drop.
- Model-profile gating **defaults to supported** when a profile field is missing — *"Only
  an explicit `False` rejects a block type."* So a gateway-proxied model with an unknown
  profile still gets the image rather than failing closed. Good default, but it means a
  genuinely non-vision model would be found out at runtime rather than at build time.

Caveat on honesty: this is verified from library source, not from a run. **Smoke-test
`read_file` on a real JPEG through the LangSmith sandbox as the first task of Phase 2** —
it's the one assumption the figure work rests on.

The three destinations for a figure still need to stay distinct:

**(a) Caption only — the default rung.** ~1,358 tokens for every caption in a paper, pure
text, answers most figure questions. No image fetch at all.

**(b) Image into a `figure-analyst` fan-out — the escalation.** As above. Haiku is
vision-capable, so the cheap-leaf cost story survives.

**(c) Image into `/workspace/out/` for the user.** Zero new machinery — a `.jpg` written
there already renders as a `chart` via `ArtifactMiddleware`. Note this is the one path
that deliberately breaks the prompt's *"never `readFile` anything in `out/`"* rule, and
the rule should be restated rather than dropped: the agent writes figures to `out/` for
the user and reads figures from `/workspace/figures/` for itself. Never `out/` for its own
consumption.

## Recommendation 6: license gating is a real feature, not a disclaimer

Full text introduces a compliance dimension abstracts never had. Measured `license_code`
across 77 papers: **CC BY 44, CC BY-NC-ND 19, TDM 11, CC BY-NC 3** — plus 12 NIH author
manuscripts, in PMC by funder mandate rather than an open licence.

`TDM` means *mine it, don't redistribute it*. So: **analysing a TDM/ND paper is fine;
copying its figure into the user's download folder is not.** That's a concrete gate on
destination (c) above, keyed off a field we already have in the sidecar. For a life
sciences client this is a strong demo beat and it costs one conditional.

## Phasing

All three phases are implemented in `pmc.py` and wired through `agent.py`. Each was
verified end to end against the live services with a real agent run.

**Phase 1 — text only. ✅** `pmc_locate` + `fetch_full_text` with section slicing and
captions, a `full-text-analyst` subagent, host-side cache, prompt ladder. No vision, no
new UI, no new middleware. This alone unlocks the thing abstracts genuinely cannot do:
*"read the methods of these eight papers and tell me the mouse strain and dose."*

*Verified:* a 12-paper base-editing query fanned out `full-text-analyst` subagents over
methods sections and returned per-paper delivery vehicles and doses (AAV9 at 1×10¹² vg,
LNP at 2.5 mg/kg, hydrodynamic injection at 30 µg) — none of which appear in an abstract —
while correctly reporting the 2 papers with no PMC full text.

**Phase 2 — figures. ✅** `fetch_figures`, a `figure-analyst` subagent reading images via
`read_file`, and license-gated export to `out/`.

*Verified:* the load-bearing question — does `read_file` return a usable image block to a
PTC-dispatched subagent — is **yes**. The analyst read GLUT1/GLUT4 box labels, node
colours and arrowhead shapes off a staged JPEG, none of which are in the caption. The
agent reached it by the intended ladder: `pmc_locate` → check `readable_in_sandbox` →
stage → delegate, without fetching body text it didn't need.

**Phase 3 — supplementary data. ✅** Route `.xlsx`/`.csv` into the sandbox and let the
existing `execute` surface do the work. Highest ratio of demo value to new code in the
whole plan, because it reuses everything.

*Verified:* a 117×12 supplementary spreadsheet loaded into pandas, summary statistics
computed, and a chart written to `out/` — with the licence checked first.

The licence gate was verified in the negative too, which is the case that matters: asked
to put a TDM-licensed figure into the download folder for a slide deck, the agent refused,
named the licence, and offered the publisher's permissions route instead.

## Caching

Extend the existing pattern — `data/abstracts/{pmid}.json` is host-side, invisible to the
agent, and exists to serve the rate limit — to `data/pmc/PMC{id}.{ver}/`. One caveat worth
stating: the 77 full texts are **11.5 MB of XML and 404 image files**, where 145 abstracts
are a rounding error. Cache the `.json`, `.txt`, `.xml` and fetched figures; never the
PDFs.

## Explicitly out of scope

`efetch db=pmc` (rate-limited, no images, no sidecar), the BioC API (nice, but a third API
for what S3's `.txt` already gives), Europe PMC (fully works — keep as fallback only),
OAI-PMH (built for corpus harvesting, not per-article lookup), and PDFs. Reasoning for
each in [`pmc_api_notes/`](pmc_api_notes/) §9.

## Risks

- **2026-08-24.** The retirement lands mid-build. The proposed path is the surviving one,
  but the new layout is young and thinly documented — I derived it by listing the bucket,
  not from docs. Worth re-probing right after the cutover.
- **Both beta surfaces still apply.** The interpreter and dynamic subagents were already
  flagged to clients. The new combination of them — a `read_file` image block inside a
  PTC-dispatched subagent — has since been verified end to end (see Phasing, Phase 2), so
  this is now the ordinary beta caveat rather than an open question.
- **53% coverage will surprise a biologist.** Better to make it a headline the agent
  reports ("full text available for 41 of 78 papers") than a caveat buried in a footnote.
