---
name: pubmed
description: Instructions on using PubMed and PubMed Central tools. Read before searching or fetching from PubMed or PMC.
---

## Searching

**Example pattern for initial searches: probe with `retmax: 0`.**
Returns `count`, `query_translation` and `warnings` without fetching records, so it is cheap.
Iterate here.

```js
let term = '("base editing"[tiab] OR "base editor"[tiab]) AND liver[tiab]';
let probe = await tools.pubmedSearch({ term, retmax: 0 });
probe.count;              // too many? tighten. zero or a handful? loosen.
probe.query_translation;  // what PubMed ACTUALLY searched
probe.warnings;           // must be empty before you trust the count
```

### Field tags

An untagged term is searched across every field *and* mapped to MeSH, which is why it
matches so much. Tag terms to control that:

| tag | example | matches |
|---|---|---|
| `[tiab]` | `"base editor"[tiab]` | title + abstract — the workhorse for a concept the authors would name |
| `[ti]` | `CRISPR[ti]` | title only; narrowest, use when the paper must be *about* the term |
| `[tw]` | `pembrolizumab[tw]` | text word: title, abstract, MeSH, substances — broader than `[tiab]` |
| `[mh]` | `Asthma[mh]` | MeSH heading, auto-expanded to narrower headings (`Asthma[mh:noexp]` to disable) |
| `[majr]` | `Alzheimer Disease[majr]` | MeSH heading flagged as a *major* topic of the paper |
| `[sh]` | `asthma/drug therapy[mh]` | MeSH subheading — attach it to a heading to narrow one concept |
| `[pa]` | `Antioxidants[pa]` | pharmacological action — a whole drug class at once |
| `[nm]` | `lenacapavir[nm]` | substance by name — drugs, proteins, rare diseases |
| `[rn]` | `50-78-2[rn]` | CAS or EC **number** only; a drug *name* here silently returns 0 — use `[nm]` |
| `[pt]` | `randomized controlled trial[pt]` | publication type; also `review`, `editorial`, `retracted publication` |
| `[dp]` | `2022:2026[dp]` | date of publication — single year or range |
| `[au]` | `Doudna JA[au]` | author; `[1au]`/`[lastau]` pin position (`Zhang F[lastau]`) |
| `[ta]` | `Nat Biotechnol[ta]` | journal — ISO abbreviation or full title |
| `[ad]` | `Broad Institute[ad]` | affiliation — institution or country, only on indexed papers |
| `[ot]` | `organoid[ot]` | author keywords — catches terms in neither MeSH nor the abstract |
| `[gr]` | `R01[gr]` | grants and funding |
| `[la]` | `english[la]` | language |
| `[sb]` | `pubmed pmc[sb]` | subset; this one restricts to papers with PMC full text |

Multi-word values work quoted or unquoted. `term*` truncates (`immunotherap*[tiab]`) —
a phrase pinned to one inflection is a filter you did not intend, so prefer
`"heart transplant*"[tiab]` over `"heart transplantation"[tiab]`.
`"a b c"[tiab:~N]` matches the words within N of each other — much more precise than
ANDing them, but supported **only** on `[tiab]`, `[ti]` and `[ad]`; on any other field it
returns 0 with a `quotedphrasesnotfound` warning.

MeSH tags (`[mh]`, `[majr]`, `[sh]`, `[pa]`) and `[tiab]` miss in opposite directions, so
neither is safe alone:

- MeSH is curated, so it is precise but **misses the most recent papers**, not yet indexed.
- `[tiab]` matches the authors' exact wording, so it **misses every paper that abbreviates,
  inflects, hyphenates or misspells** the concept. A title typo or an author who writes
  "C. jejuni" throughout drops the paper silently, and no `warnings` entry fires.

**Fetching records once you've appropriately narrowed your search:**

```js
const res = await tools.pubmedSearch({ term, sort: "relevance" });
res.records; // [{ pmid, title, first_author, last_author, year, journal, doi }]
```

All matching records come back, not a truncated head, so you can filter and sort them in
code. If you want the records as a file, write them yourself with `tools.writeFile`.

**Check `res.warnings` before you trust anything.** PubMed does not reject malformed
queries — it silently rewrites them and returns a large, confident, wrong result set. A
mistyped field tag is dropped and the search runs across every field, which can return
millions of irrelevant hits that look exactly like a successful search. If it is non-empty, 
fix the query and search again rather than reporting the results.

`res.query_translation` is what PubMed actually searched, including its MeSH expansion
(`IL-6` becomes `"interleukin 6"[Supplementary Concept] OR ...`). Show it to the user
alongside the final count.

## Fetching abstracts

```js
const pmids = res.records.map(r => r.pmid);
const { records, missing, invalid } = await tools.fetchAbstracts({ pmids });
// records -> { [pmid]: { title, abstract, sections, journal, year, retracted } }
```

Pass every PMID in one call. Batching is what keeps this inside NCBI's rate limit —
never loop one PMID at a time. Results are cached on disk, so refetching is free.

- `abstract` is `null` for errata and editorials, which have metadata but no body. Skip
  those rather than reporting them as unanswerable.
- `sections` preserves structured-abstract labels (BACKGROUND, METHODS, FINDINGS,
  INTERPRETATION) when the journal uses them. Use them when the question is about one
  part of a study, e.g. only the methods.
- `retracted: true` means the paper has been retracted. **Always tell the user** —
  never cite a retracted paper silently.
- `missing` are PMIDs PubMed returned nothing for; `invalid` are malformed inputs.
- `pmcid` is non-null when the paper may have full text in PubMed Central. Ignore it
  unless the question needs more than an abstract — see "Reading full papers".

## Reading full papers

About half of PubMed papers have full text in PubMed Central. `pmcid` is on every record
from `pubmedSearch` and `fetchAbstracts` — non-null means full text may exist, null means
abstract-only.

**Escalate only as far as the question requires.** Per paper, roughly:

| step | cost | answers |
|---|---|---|
| abstract | ~250 tokens | what the study claims |
| `pmcLocate` (titles, counts) | ~40 tokens | what's *in* the paper |
| figure captions (from `pmcLocate`) | ~1,500 tokens | most figure questions |
| one section of the body | ~1,000–4,700 tokens | methods, results, a specific claim |
| the whole body | ~10,000 tokens | genuinely paper-wide questions |

Most questions are answered by abstracts. Reach for full text when the user asks
something an abstract structurally cannot answer — exact protocols, doses, cell lines,
sample sizes, statistical tests, or what a specific figure shows.

### Triage first

```js
const pmcids = Object.values(records).map(r => r.pmcid).filter(Boolean);
const { available, unavailable } = await tools.pmcLocate({ pmcids });
```

`unavailable` is normal, not an error — say "no full text available" and use the
abstract. Each entry in `available` has `body_chars`, `sections` (with a `canonical`
name: intro/methods/results/discussion/conclusion), `figures` (with **full captions**),
`tables` and `supplementary`.

**Never make `available` the final expression of an `eval`.** It is ~2,000 tokens per
paper, mostly captions — across a 77-paper corpus that is 155,000 tokens. Filter and
project it down in JavaScript, then return a small summary:

```js
const triage = Object.values(available).map(d => ({
  pmcid: d.pmcid, sections: d.sections.filter(s => s.canonical).map(s => s.canonical),
  figs: d.figures.length, chars: d.body_chars,
}));
triage; // ~40 tokens per paper
```

### Fetching the text

```js
const { records: full } = await tools.fetchFullText({
  pmcids, sections: ["methods"],   // omit for the whole body
});
```

- `sections` takes canonical names or a substring of a literal section title. Roughly 1
  paper in 4 has no methods section (reviews, mostly). When nothing matches, you get the
  **whole body** and `fell_back: true` — check it, or you will silently pay 3× what you
  budgeted.
- `include_captions` (default true) appends figure captions; `include_tables` (default
  true) appends table captions and their rows, which is where numeric results live.
- **One paper's full text already exceeds the `eval` result limit.** Never return `full`,
  never `console.log` body text. It goes into subagent prompts and nothing else.

### Delegate the reading

Full text is ~40× an abstract, so read it yourself only when the user asked about one
specific paper. For anything across papers, fan out `full-text-analyst` subagents exactly
as you do for abstracts — one per paper, one `Promise.all`, the text in the prompt:

```js
const answers = await Promise.all(Object.values(full).map(async (r) => ({
  pmcid: r.pmcid, pmid: r.pmid, title: r.title, retracted: r.retracted,
  answer: await task({
    description: `Question: ${question}\n\nTitle: ${r.title}\nPMCID: ${r.pmcid}\n\n${r.text}`,
    subagentType: "full-text-analyst",
  }),
})));
await tools.writeFile({
  file_path: "/workspace/answers.json", content: JSON.stringify(answers),
});
answers.map(a => ({ pmid: a.pmid, answer: a.answer })); // projection, not the whole array
```

### Reading figures

Try captions first — `pmcLocate` already gave you every caption in full, and they answer
most figure questions for a fraction of the cost.

When the answer is genuinely only in the image, stage it and delegate. `fetchFigures`
returns **paths, not images**; a `figure-analyst` reads the path and actually sees it:

```js
const { staged, skipped } = await tools.fetchFigures({ pmcid, files: ["Figure 2"] });
const answer = await task({
  description: `Question: ${question}\n\nCaption: ${caption}\n\nImage: ${staged[0].path}`,
  subagentType: "figure-analyst",
});
```

Only stage figures whose `readable_in_sandbox` is true. For the rest (~15% — PMC never
deposited the image, or it is over the 500 KB read limit) `unavailable_reason` says
which; fall back to the caption and tell the user the image wasn't available.

Do not `readFile` an image yourself unless the user asked about that one figure — an
image costs the same in your context as in a cheap subagent's, and you have the whole
synthesis still to do. Note that this does not apply to images you create yourself as output
to the user.

### Supplementary data

`fetchSupplementary` stages spreadsheets into the sandbox for Python — this is where
per-sample data lives. Read them with pandas, never with `readFile`:

```js
const { staged } = await tools.fetchSupplementary({ pmcid, files: ["mmc2.xlsx"] });
await tools.execute({ command: `python3 - <<'PY'
import pandas as pd
print(pd.read_excel("${staged[0].path}").head().to_string())
PY` });
```

### Licensing

Every result carries `license` and `redistributable`. Mining any of it is fine.
**When `redistributable` is false, do not copy that paper's figures or supplementary
files into `/workspace/out/`** — TDM and ND licences permit analysis but not
republication. Quoting, describing and computing over them is still fine. About 40% of
papers with full text are in this category, so check rather than assume.
