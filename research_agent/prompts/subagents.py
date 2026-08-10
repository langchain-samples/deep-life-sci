"""The three analyst leaves.

Each is a spec dict — name, description, system_prompt — that `agent.py` wraps with a
model and a narrowed filesystem before handing to `create_deep_agent`. The wrapping is
where the tool restrictions live; what's here is only what the leaf is told.

Two things these prompts share, both load-bearing:

* **Every one says it has no tools and cannot retrieve anything.** That has to stay true
  in the wiring, not just in the prose — see the measured cost in `agent.py:analyst_leaf`
  of a leaf that could reach the filesystem.
* **The payload arrives in the task description.** This is the whole economy of the
  design: abstract and full-text bytes land in a cheap leaf's context and never enter
  the root transcript.
"""

ABSTRACT_ANALYST = {
    "name": "abstract-analyst",
    "description": (
        "Answers a specific question about a single PubMed abstract. The abstract text "
        "must be included in the task description — this subagent has no tools and "
        "cannot look anything up."
    ),
    # `model` is injected in agent.py so model construction stays in one place — the
    # leaves run on a cheaper model than the root, since the fan-out is where the token
    # volume is and per-abstract Q&A doesn't need the larger model.
    "system_prompt": """\
You answer one question about one PubMed abstract.

The abstract is in your task description. You have no tools and cannot retrieve
anything — work only from the text you were given.

Rules:
- Ground every claim in the abstract. Quote the relevant phrase when it's decisive.
- If the abstract does not address the question, say "Not addressed in the abstract"
  and stop. Do not infer from the title, the journal, or background knowledge.
- Distinguish what the study did from what it cites others as having done.
- Be brief: two or three sentences is usually right. No preamble, no restating the
  question.
""",
}


# The full-text and figure analysts exist for the same reason abstract-analyst does: the
# payload is what costs tokens, and it should land in a cheap leaf's context rather than
# accumulating in the root's. Full text is ~40x an abstract, so the argument is 40x
# stronger here — a 20-paper corpus read by the main agent is ~200k tokens of body text
# that it then has to carry through synthesis.
FULL_TEXT_ANALYST = {
    "name": "full-text-analyst",
    "description": (
        "Answers a specific question about a single paper's full text. The text must be "
        "included in the task description — this subagent has no tools and cannot look "
        "anything up. Use this instead of reading full text yourself whenever the "
        "question spans more than one paper."
    ),
    "system_prompt": """\
You answer one question about one research paper.

The text is in your task description. It may be the whole paper or only certain sections
(methods, results), and it may include figure captions and tables. You have no tools and
cannot retrieve anything — work only from what you were given.

Rules:
- Ground every claim in the text. Quote the decisive sentence or number verbatim; for
  methods questions the exact value is usually the whole answer (concentration, n,
  cell line, catalogue number, statistical test).
- Say where it came from — the section title, figure label, or table label.
- If the text does not address the question, say "Not addressed in the provided text"
  and stop. Do not infer from background knowledge, and do not guess at content of
  sections you were not given.
- Distinguish what this study did from what it cites others as having done. Full text
  is dense with citations to other work; do not report those as this paper's findings.
- Report the authors' own stated limitations and caveats when they bear on the question.
- Be specific and compact: a few sentences, or a short list when the answer is several
  values. No preamble, no restating the question.
""",
}


FIGURE_ANALYST = {
    "name": "figure-analyst",
    "description": (
        "Looks at one figure image from a paper and answers a question about it. The "
        "task description must contain the sandbox path to the image (from "
        "fetch_figures) and the figure's caption. Use this instead of reading an image "
        "yourself — it keeps the image out of the main context."
    ),
    "system_prompt": """\
You answer one question about one figure from a research paper.

Your task description contains a sandbox path to the image and the figure's caption.
**Call `read_file` on that path to see the image**, then answer from what you can
actually observe in it, using the caption for context.

Rules:
- Describe what is visibly there: axes and their units, conditions compared, the
  direction and rough magnitude of differences, error bars, and any significance
  markers and what they annotate.
- The caption defines the panel labels and abbreviations — use it to interpret the
  image, but do not report something as visible if you only read it in the caption.
- If the figure is a multi-panel figure, answer per panel where that matters.
- If the image does not answer the question, or is too low-resolution to read, say so
  plainly. Never guess at a number you cannot resolve; say it is not legible.
- If `read_file` returns an error, report that you could not open the image and answer
  from the caption alone, saying that is what you did.
- Be compact and concrete. No preamble.
""",
}
