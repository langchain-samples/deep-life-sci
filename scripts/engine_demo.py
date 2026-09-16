"""Generate LangSmith traces for the Engine demo: a healthy set and a broken set.

    uv run scripts/engine_demo.py --set healthy   # traces that show the agent working
    uv run scripts/engine_demo.py --set broken    # traces that show the planted failure
    uv run scripts/engine_demo.py --set both
    uv run scripts/engine_demo.py --probe-only    # no agent, just the tool

The demo mirrors `banking-concierge`: one deliberately planted bug, a set of prompts that
reach it reliably, and an isolated tracing project so the broken traces never mix with the
project the healthy agent writes to.

**The planted bug** lives in `deep_life_sci/models.py:WEB_SEARCH_SPECS`. The two providers'
server-side search specs have been "unified" behind a shared `_SEARCH_TOOL` fragment, so the
OpenAI path now carries Anthropic's `name` and `max_uses` keys. The Responses API rejects
them — `400 Unknown parameter: 'tools[0].name'` — on every single call.

**It does not propagate.** `sources/web.py:web_search` wraps the model call in a blanket
`except Exception` and returns `_failed(...)`: a normal digest with an empty `answer` and
the reason in `warnings`. That containment is deliberate upstream behaviour, not part of the
bug — a raise there would leave the QuickJS bridge and kill the whole run. So every web
search fails, the run continues, and the agent answers from PubMed, the registry and its own
memory with nothing from the live web.

**Isolation.** Two things are redirected before `deep_life_sci` is imported, because both
are read at import time:

* `LANGSMITH_PROJECT` gets the `-engine-demo` suffix, so Engine scans a project containing
  only demo traces. The healthy set is deliberately traced there too — Engine needs the
  working majority to cluster the failure against.
* `DEEP_LIFE_SCI_DATA_DIR` moves to `data-engine-demo/`, so demo runs never share cache
  state with the healthy agent.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Same capture-and-restore dance as `cli.py` and `evals/run.py`: override=True is what
# keeps an exported LANGSMITH_PROJECT from capturing these traces, but it is too blunt for
# the axes a demo run wants to vary from the command line.
from deep_life_sci.models import ENV_VARS  # noqa: E402

_OVERRIDES = {k: v for k in (*ENV_VARS, "DEEP_LIFE_SCI_CACHE_TTL")
              if (v := os.environ.get(k)) is not None}
load_dotenv(override=True)
os.environ.update(_OVERRIDES)

DEMO_SUFFIX = "-engine-demo"

_base_project = os.environ.get("LANGSMITH_PROJECT") or "deep-life-sci"
DEMO_PROJECT = (
    _base_project if _base_project.endswith(DEMO_SUFFIX) else _base_project + DEMO_SUFFIX
)
os.environ["LANGSMITH_PROJECT"] = DEMO_PROJECT
os.environ.setdefault("LANGSMITH_TRACING", "true")

# Must be set before `deep_life_sci.paths` is imported — it resolves DATA_DIR at import.
os.environ.setdefault("DEEP_LIFE_SCI_DATA_DIR", str(REPO_ROOT / "data-engine-demo"))
# A demo is re-run; an idle TTL expiring mid-rehearsal turns a warm corpus cold for
# reasons that have nothing to do with the agent.
os.environ.setdefault("DEEP_LIFE_SCI_CACHE_TTL", "off")

from langsmith import tracing_context  # noqa: E402

from deep_life_sci.models import check_gateway_config, describe  # noqa: E402
from deep_life_sci.runner import run_once  # noqa: E402
from deep_life_sci.sources import web  # noqa: E402

# --------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------

# Every one of these is squarely in `web_search`'s remit and outside everything else's:
# regulatory actions, guideline revisions, company announcements, software releases. The
# tool's own docstring lists exactly these as "what PubMed and ClinicalTrials.gov
# structurally cannot answer", so the agent reaches for it without being told to, and the
# question cannot be answered properly once it comes back empty.
BROKEN_PROMPTS: list[tuple[str, str]] = [
    (
        "glp1-regulatory-actions",
        "What regulatory actions have the FDA and EMA taken on GLP-1 receptor agonists "
        "during 2026? Cover new approvals, label changes and safety communications, and "
        "date each one.",
    ),
    (
        "sglt2-ckd-guidelines",
        "What do the current KDIGO and ADA guidelines recommend for SGLT2 inhibitors in "
        "chronic kidney disease, and when was each guideline last revised? Give the "
        "recommendation grades.",
    ),
    (
        "gene-therapy-clinical-holds",
        "Which gene therapy programs were placed on clinical hold or discontinued during "
        "2026? Give the company, the program, the stated reason and the date for each.",
    ),
    (
        "scrnaseq-qc-tooling",
        "What is the currently recommended quality-control pipeline for single-cell "
        "RNA-seq, and what changed in the most recent Scanpy and Seurat releases? Cite the "
        "release notes and give version numbers.",
    ),
]

# The contrast set. Deliberately answerable from PubMed and the registry alone, so the
# planted bug cannot fire: these are what you show first, and Engine needs the working
# majority in the project to cluster the failure against.
HEALTHY_PROMPTS: list[tuple[str, str]] = [
    (
        "semaglutide-phase3-landscape",
        "Summarize the phase 3 semaglutide trials in obesity registered on "
        "ClinicalTrials.gov: sponsor, enrollment, status and primary completion date for "
        "each, and tell me how many have posted results.",
    ),
    (
        "car-t-solid-tumor-literature",
        "Find recent PubMed papers on CAR-T therapy in solid tumors and tell me which ones "
        "report objective response rates. Plot the distribution of publication years.",
    ),
    (
        "alzheimers-completed-trials",
        "For the completed anti-amyloid trials in Alzheimer's disease registered on "
        "ClinicalTrials.gov, give me each trial's actual enrollment, eligible age window "
        "and lead sponsor, sorted by enrollment.",
    ),
    (
        "base-editing-liver-literature",
        "Using the published literature, what delivery approaches are used for base "
        "editing in the liver, and what off-target effects do those papers report? Base "
        "the answer on PubMed and PMC full text.",
    ),
]

SETS = {"broken": BROKEN_PROMPTS, "healthy": HEALTHY_PROMPTS}

# --------------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------------

PROBE_QUERY = "What did the FDA approve in August 2026?"

# What the Responses API says when Anthropic's extra spec keys reach it. Matched loosely —
# the provider names whichever of `name` / `max_uses` it validates first — but specifically
# enough that a real outage is not mistaken for the planted bug.
BUG_SIGNATURE = "unknown parameter"


async def probe() -> bool:
    """Call `web_search` directly and print what comes back. True if the bug fired."""
    result = await web.web_search.ainvoke({"query": PROBE_QUERY})
    warnings = result.get("warnings") or []
    answer = result.get("answer") or ""

    print(f"\n  query     : {result.get('query')}")
    print(f"  answer    : {len(answer)} chars")
    print(f"  sources   : {len(result.get('sources') or [])}")
    print(f"  searched  : {result.get('searched') or []}")
    for warning in warnings:
        print(f"  warning   : {warning[:300]}")

    contained = answer == "" and any("web search unavailable" in w for w in warnings)
    caused_by_spec = any(BUG_SIGNATURE in w.lower() for w in warnings)
    if contained and caused_by_spec:
        return True
    if contained:
        print("\n  !! the search failed, but not with the planted spec error — read the")
        print("     warning above before rehearsing; this may be a real outage.")
    return False


# --------------------------------------------------------------------------------
# Counting what the agent's web searches did
# --------------------------------------------------------------------------------

# Per-run tallies. A ContextVar rather than a module global because runs go out
# concurrently: `asyncio.gather` copies the context into each task, so a `set()` inside one
# run is invisible to its sibling, and the tasks it spawns inherit it.
_TALLY: contextvars.ContextVar[dict | None] = contextvars.ContextVar("tally", default=None)


def install_counters() -> None:
    """Count `web_search` attempts and containments without changing what either does.

    Patched onto the module rather than the tool object: `web_search` looks both names up
    in `web.py`'s globals at call time, so this catches every call however the tool was
    bound into the agent — including from inside a subagent or the QuickJS bridge.

    `web_search_model()` is called once per attempt and sits *outside* the tool's try block;
    `_failed()` is the containment path. Counting both is what separates "every search
    failed" from "the agent never searched", which are very different demo outcomes and
    look identical if you only count failures.
    """
    real_model = web.web_search_model
    real_failed = web._failed

    def counting_model(*args, **kwargs):
        if (tally := _TALLY.get()) is not None:
            tally["attempts"] += 1
        return real_model(*args, **kwargs)

    def counting_failed(query, reason):
        if (tally := _TALLY.get()) is not None:
            tally["failures"] += 1
            tally["reasons"].append(reason)
        return real_failed(query, reason)

    web.web_search_model = counting_model
    web._failed = counting_failed


def verdict(tally: dict) -> tuple[bool, str]:
    """Whether one run carries the failure, and why. `(fired, explanation)`.

    Measured at the tool boundary rather than read out of the answer text. That is the
    honest place for this bug: a contained failure leaves the final answer *plausible* —
    the agent routes around the dead tool and answers from PubMed and memory — so an
    answer-text check would mostly measure how candid the agent felt like being.
    """
    attempts, failures = tally["attempts"], tally["failures"]
    if attempts == 0:
        return False, "the agent never called web_search — this prompt does not reach the bug"
    spec_errors = sum(1 for r in tally["reasons"] if BUG_SIGNATURE in r.lower())
    if failures == attempts and spec_errors == failures:
        return True, f"BUG VISIBLE: {failures}/{attempts} web searches failed on the spec 400"
    if failures == attempts:
        return True, (
            f"all {failures}/{attempts} web searches failed, but "
            f"{failures - spec_errors} for some other reason — read the warnings"
        )
    return False, f"only {failures}/{attempts} web searches failed"


# --------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------

async def run_set(which: str, limit: int | None, concurrency: int) -> bool:
    """Run one prompt set. Returns True if every broken-set run carried the failure."""
    prompts = SETS[which][: limit or None]
    sem = asyncio.Semaphore(concurrency)
    results: list[tuple[str, float, dict]] = []

    # Answers are kept on disk because the trace is the real artifact: a rehearsal needs the
    # text to hand when a verdict looks wrong.
    answer_dir = Path(os.environ["DEEP_LIFE_SCI_DATA_DIR"]) / "answers" / which
    answer_dir.mkdir(parents=True, exist_ok=True)
    # Clear first, so a case that has been renamed or dropped from the set does not leave a
    # stale answer behind for the next rehearsal to read as current.
    for old_answer in answer_dir.glob("*.md"):
        old_answer.unlink()

    async def one(name: str, question: str) -> None:
        async with sem:
            tally: dict = {"attempts": 0, "failures": 0, "reasons": []}
            _TALLY.set(tally)
            print(f"[{which}] {name}: running")
            # The project comes from LANGSMITH_PROJECT set above; naming it here too makes
            # the destination explicit at the call site rather than a side effect of
            # module import order.
            with tracing_context(
                project_name=DEMO_PROJECT,
                tags=["engine-demo", f"set:{which}", f"case:{name}"],
                metadata={"demo_set": which, "demo_case": name},
            ):
                try:
                    result = await run_once(question)
                except Exception as exc:  # noqa: BLE001 — a failed run is still a trace
                    print(f"[{which}] {name}: ERROR {exc!r}")
                    return
            searches = f"{tally['failures']}/{tally['attempts']} web searches failed"
            (answer_dir / f"{name}.md").write_text(
                f"# {name}\n\n## Question\n\n{question}\n\n"
                f"## Web search\n\n{searches}\n\n## Answer\n\n{result.answer}\n"
            )
            results.append((name, result.duration_seconds, tally))
            print(f"[{which}] {name}: done in {result.duration_seconds:.0f}s, {searches}")

    await asyncio.gather(*(one(name, q) for name, q in prompts))

    print(f"\n=== {which} set ===")
    all_fired = True
    for name, seconds, tally in results:
        if which != "broken":
            hit = "" if tally["attempts"] == 0 else f"  ({tally['attempts']} web searches)"
            print(f"  {name:<32} {seconds:>5.0f}s  ok{hit}")
            continue
        fired, why = verdict(tally)
        all_fired = all_fired and fired
        print(f"  {name:<32} {seconds:>5.0f}s  {why}")
    print(f"\n[demo] answers written to {answer_dir}")
    return all_fired and len(results) == len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("broken", "healthy", "both"), default="broken")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Call web_search directly and exit. One cheap model call, no agent.",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N prompts only")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent agent runs. Each boots its own sandbox and fans out internally.",
    )
    args = parser.parse_args()

    print(f"[demo] tracing project : {DEMO_PROJECT}")
    print(f"[demo] cache directory : {os.environ['DEEP_LIFE_SCI_DATA_DIR']}")
    print(f"[demo] search role     : {describe('search')}")

    if args.probe_only:
        fired = asyncio.run(probe())
        print(
            "\n[demo] planted bug is ACTIVE" if fired
            else "\n[demo] planted bug did NOT fire — check models.py:WEB_SEARCH_SPECS"
        )
        raise SystemExit(0 if fired else 1)

    check_gateway_config()
    print(f"[demo] models          : {describe()}")
    install_counters()

    async def go() -> None:
        print("\n[demo] pre-flight probe of web_search:")
        if not await probe():
            print("\n[demo] planted bug did NOT fire — aborting before spending on runs")
            raise SystemExit(1)
        print("\n[demo] planted bug is ACTIVE\n")
        ok = True
        for which in (("healthy", "broken") if args.set == "both" else (args.set,)):
            ok = await run_set(which, args.limit, args.concurrency) and ok
        if args.set in ("broken", "both") and not ok:
            print(
                "\n[demo] at least one broken-set run did not carry the failure. "
                "Read the saved answers before rehearsing."
            )
            raise SystemExit(1)

    asyncio.run(go())


if __name__ == "__main__":
    main()
