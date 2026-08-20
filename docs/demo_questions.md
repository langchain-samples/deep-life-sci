# Demo questions

Ten questions this agent is well-suited to, chosen to be diverse along three axes: the
biology domain, the *kind* of question, and which of the agent's two code surfaces it
leans on (`eval`/fan-out for reading comprehension, `execute`/Python for counting and
statistics).

Each is written to be pasted verbatim as `DEMO_QUESTION` in `agent.py`.

---

## 1. Terminology drift — hepatology

> When did the term "MASLD" overtake "NAFLD" in the literature? Plot both by publication
> year, identify the crossover point, and tell me whether the combined volume grew or
> whether it was purely a renaming.

**Exercises:** two searches, then almost pure Python. Barely touches the fan-out — a good
check that the agent doesn't dispatch subagents for something that's just counting.

## 2. Methodological rigour over time — infectious disease / GI

> For clinical studies of fecal microbiota transplantation in *C. difficile* infection,
> what fraction were randomized and placebo-controlled, and has that fraction changed
> over time?

**Exercises:** fan-out classification (each abstract → a study-design label) feeding a
time series. The classification can't be done by string matching, and the trend can't be
done by reading.

## 3. Mapping disagreement — metabolism

> Do studies on intermittent fasting and insulin sensitivity in humans agree? Group them
> into supports / null / opposes, show the split, and tell me where the disagreement
> concentrates — by year, by journal, or by whether the study was randomized.

**Exercises:** the fan-out's real strength. "Supports vs null vs opposes" is a judgement
per abstract, and the honest answer includes how many abstracts don't address it at all.

## 4. Numeric extraction plus statistics — oncology / immunotherapy

> Pull the reported objective response rates from abstracts of CAR-T trials in solid
> tumours, then give me the median and interquartile range and plot the distribution.

**Exercises:** both surfaces at full stretch — subagents extract a number (or report that
there isn't one) from prose, Python does the descriptive statistics. Also a good test of
whether the agent reports its denominator, since many abstracts won't state an ORR.

## 5. Model-system inventory — neurodegeneration

> Across papers on tau propagation, which model systems were used — mouse, iPSC-derived
> neurons, organoids, human post-mortem tissue? Tally them and cross-tabulate against
> publication year.

**Exercises:** multi-label categorical fan-out (papers often use more than one system)
into a cross-tab. Tests whether the agent handles a paper belonging to several buckets
rather than forcing one.

## 6. Where a field publishes — structural biology

> Which journals publish the most cryo-EM work on membrane transporters, and has the
> field shifted venues over the past decade? Show the top journals by period.

**Exercises:** metadata only — no abstract reading needed at all. `pubmed_search` already
returns journal and year, so this should be one search and one Python block.

## 7. Publication bias — psychiatry / microbiome

> How often do abstracts on the gut microbiome and depression report a positive
> association versus a null or negative finding? Break it down by study type and by
> whether the work was in humans or rodents.

**Exercises:** fan-out with a deliberately awkward category (null results are often
phrased as absences, not statements) plus a two-way breakdown.

## 8. Integrity-aware reading list — stem cell biology

> Give me a reading list on cardiac regeneration via cardiomyocyte proliferation, and
> explicitly flag anything retracted or subject to a published correction. I want to know
> what I can safely cite.

**Exercises:** the `retracted` flag, which the tools surface and the prompt requires be
reported. A field with a genuine retraction history, so the flag should actually fire.

## 9. Two-subfield comparison — nephrology / methods

> Compare the adoption of single-cell RNA-seq versus spatial transcriptomics in kidney
> research: publication counts by year for each, and the typical sample sizes reported.

**Exercises:** two independent searches held side by side, then a fan-out for the sample
sizes (which only appear in prose) layered on metadata counts.

## 10. Mining structured abstracts — dermatology / immunology

> For phase III trials of biologics in atopic dermatitis, what primary endpoints do the
> abstracts report, and how consistent are they across trials?

**Exercises:** the `sections` field. Structured abstracts from clinical journals carry
METHODS/RESULTS labels, and the question is answerable from one section — a case where
using the labels beats reading the whole abstract.

---

## Poor fits, for contrast

Worth knowing where this agent will disappoint, so a demo doesn't wander into it:

- **Anything needing full text.** Abstracts only. No PMC, no figures, no supplementary
  tables. "What buffer did they use?" is unanswerable.
- **Anything needing data outside PubMed and ClinicalTrials.gov.** No GEO, no dbGaP, no
  citation counts, no web search. It cannot tell you what a paper's impact was.
- **Posted trial results.** The registry tells you a trial *has* a results section and
  links its publications; it will not read the posted outcome tables back to you. Ask for
  the paper instead.
- **Genomics file formats.** The sandbox has numpy/pandas/scipy/matplotlib, not PLINK or
  bcftools, and no way to fetch a VCF. It computes over what the abstracts yield.
- **Author disambiguation.** `first_author`/`last_author` are strings from PubMed. Two
  different people named J. Zhang are one string.
- **Exhaustive claims.** The agent targets ≤200 papers and shapes queries to get there,
  so "every paper ever published on X" is the wrong framing — ask for a characterised
  corpus and it will tell you what it searched and how many matched.
