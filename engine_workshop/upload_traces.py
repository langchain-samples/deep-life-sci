"""Replay the saved traces with fresh ids, recent timestamps and researcher ratings.

    uv run python -m engine_workshop.upload_traces
    uv run python -m engine_workshop.upload_traces --project my-project --days 0.5 --seed 42
    uv run python -m engine_workshop.upload_traces --if-missing   # what setup.py runs

**This is the way to generate traces for the workshop.** It calls no model, needs no
provider key, and takes about a minute, so everyone in the room gets the same batch and
Engine finds the same clusters. `scripts/engine_demo.py` drives the live agent instead,
which is worth watching once but produces a different batch every time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client, uuid7  # noqa: E402
from langsmith.utils import LangSmithNotFoundError  # noqa: E402

from engine_workshop import demo_project, query_runs  # noqa: E402

DEFAULT_INPUT = Path(__file__).resolve().parent / "traces.json"

# Ingest rejects any run whose start_time is more than 24 hours from now with a 422, on
# both the multipart and batch endpoints, and the runs are silently dropped. Stay under it
# so the earliest trace in the spread still clears the limit once the upload has run a while.
MAX_BACKDATE_DAYS = 0.95

# Ingested runs take a moment to become queryable, so the landing check retries rather than
# failing on the first empty read.
WAIT_ATTEMPTS = 10
WAIT_SECONDS = 3

RATING_KEY = "researcher_rating"

# What a researcher leaves on an answer, and the whole point of collecting it here: they
# rate what is in front of them. A fluent, well-organised answer reads as a win at the
# moment of rating even when the web search behind it returned nothing — the reader has no
# way to know a tool was dead. An answer that *says* it could not search is visibly
# unhelpful and gets the thumb down, even though it is the more honest of the two.
#
# So the ratings invert the truth of the run, which is exactly why user feedback alone does
# not find this bug and Engine has to. Most ratings carry no note, which is how people
# actually use thumbs.
GOOD_NOTES = (None, None, None, None, None, None,
              "clear, thanks", "exactly what I needed", "good summary", "useful")
BAD_NOTES = (None, None, None, None,
             "didn't actually answer the question", "said it couldn't look it up",
             "no sources, had to check myself")

# An answer that admits the tool failed. Matched on the phrasing the agent actually used
# across the captured runs rather than on the tool's own warning string, because this is
# about what reached the reader.
_DISCLOSURE = (
    "unavailable", "could not search", "couldn't search", "cannot search",
    "search service", "unable to search", "no live", "without live",
    "search failed", "could not be retrieved",
)


def parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO timestamp into a naive (tz-stripped) datetime."""
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def final_answer(trace_runs: list[dict]) -> str:
    """The text the researcher actually read, off the root run's outputs."""
    root = next((r for r in trace_runs if r["parent_run_id"] is None), None)
    return json.dumps((root or {}).get("outputs") or {}, default=str).lower()


def rate_trace(trace_runs: list[dict]) -> tuple[int, str | None]:
    """The (score, comment) a researcher would leave. 1 = good, 0 = bad.

    Derived from the answer, not from whether the tool failed — see the note on the rating
    constants. A silent failure is rated *well*.
    """
    answer = final_answer(trace_runs)
    if any(phrase in answer for phrase in _DISCLOSURE):
        return 0, random.choice(BAD_NOTES)
    return 1, random.choice(GOOD_NOTES)


def bootstrap_project(client: Client, name: str):
    """Materialise the project up front so downstream reads do not race propagation.

    On a fresh project `read_project` 404s until the first run has landed and been indexed,
    so the verification step at the end would crash even though the ingest succeeded.
    Creating it here also surfaces a real auth or tenant problem before hundreds of runs go
    out.
    """
    try:
        return client.read_project(project_name=name).id
    except LangSmithNotFoundError:
        return client.create_project(project_name=name).id


def batch_present(client: Client, project: str, n_traces: int) -> bool:
    """Whether `project` already holds at least one batch's worth of traces.

    What makes `--if-missing` idempotent, so `scripts/setup.py` can run this on every
    re-run. Counted on root runs rather than matched on ids, because every upload mints
    fresh ones: there is nothing stable to match on, and a whole batch's worth is the
    signal that one landed. A partial upload fails its own landing check below, so a
    short count means re-upload rather than accept.
    """
    try:
        project_id = client.read_project(project_name=project).id
    except LangSmithNotFoundError:
        return False
    return len(query_runs(client, project_id, is_root=True, selects=["ID"])) >= n_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default=None, help="Target project (default: the demo project)"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input file path")
    parser.add_argument(
        "--days",
        type=float,
        default=MAX_BACKDATE_DAYS,
        help=f"Spread traces randomly over this many days ending now (default "
             f"{MAX_BACKDATE_DAYS}; ingest rejects anything older than 24h)",
    )
    parser.add_argument("--seed", type=int, help="Seed for a reproducible upload")
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip the upload when the project already holds a full batch (for setup)",
    )
    args = parser.parse_args()

    if args.days > MAX_BACKDATE_DAYS:
        parser.error(
            f"--days {args.days} exceeds the ingest API's 24-hour backdating limit; runs "
            f"older than that are rejected with a 422 and never land. Use "
            f"--days {MAX_BACKDATE_DAYS} or less."
        )
    if args.seed is not None:
        random.seed(args.seed)

    project = args.project or demo_project()
    runs = json.loads(Path(args.input).read_text())
    print(f"Loaded {len(runs)} runs from {args.input}")
    if not runs:
        print("Nothing to upload.")
        return

    client = Client()
    n_traces = sum(1 for run in runs if run.get("parent_run_id") is None)
    if args.if_missing and batch_present(client, project, n_traces):
        print(f"'{project}' already holds the {n_traces}-trace batch; skipping the upload.")
        return

    # Fresh uuid7s (time-ordered). A root run's trace_id must equal its id, so map both to
    # the same new value.
    id_map: dict[str, str] = {}
    for run in runs:
        if run.get("parent_run_id") is None:
            new_id = str(uuid7())
            id_map[run["id"]] = new_id
            id_map[run["trace_id"]] = new_id
    for run in runs:
        for field in ("id", "parent_run_id"):
            old = run.get(field)
            if old and old not in id_map:
                id_map[old] = str(uuid7())

    traces: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        trace_id = id_map[run["trace_id"]]
        traces[trace_id].append(
            {
                "id": id_map[run["id"]],
                "trace_id": trace_id,
                "dotted_order": None,
                "parent_run_id": id_map.get(run.get("parent_run_id")),
                "name": run["name"],
                "run_type": run["run_type"],
                "inputs": run.get("inputs") or {},
                "outputs": run.get("outputs"),
                "error": run.get("error"),
                "extra": run.get("extra") or {},
                "tags": run.get("tags"),
                "start_time": parse_dt(run["start_time"]),
                "end_time": parse_dt(run["end_time"]) if run.get("end_time") else None,
                "feedback": run.get("feedback") or [],
            }
        )

    # Scatter each trace to a random point in the window, shifting all of its runs by the
    # same delta so the internal spacing and nesting stay intact. Traces land independently,
    # so the batch reads as a few days of ordinary use rather than one replayed burst.
    now = datetime.now(UTC).replace(tzinfo=None)
    window = timedelta(days=args.days)
    for trace_runs in traces.values():
        trace_start = min(r["start_time"] for r in trace_runs)
        trace_end = max([r["end_time"] for r in trace_runs if r["end_time"]] or [trace_start])
        latest_start = max(window - (trace_end - trace_start), timedelta(0))
        offset = timedelta(seconds=random.uniform(0, latest_start.total_seconds()))
        delta = (now - window + offset) - trace_start
        for run in trace_runs:
            run["start_time"] += delta
            if run["end_time"]:
                run["end_time"] += delta

    starts = [r["start_time"] for trace_runs in traces.values() for r in trace_runs]
    print(
        f"Spread {len(traces)} traces over {args.days} days: "
        f"{min(starts):%Y-%m-%d %H:%M} to {max(starts):%Y-%m-%d %H:%M}"
    )

    project_id = bootstrap_project(client, project)
    print(f"Uploading {len(traces)} traces to '{project}'...")

    for i, (_trace_id, trace_runs) in enumerate(traces.items(), 1):
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))
        by_id = {r["id"]: r for r in trace_runs}
        dotted: dict[str, str] = {}

        def dotted_order(run, by_id=by_id, dotted=dotted):
            """Walk the parent chain, so nesting is right whatever the run order."""
            rid = run["id"]
            if rid in dotted:
                return dotted[rid]
            segment = run["start_time"].strftime("%Y%m%dT%H%M%S%f") + "Z" + rid
            parent = by_id.get(run["parent_run_id"])
            order = segment if parent is None else f"{dotted_order(parent)}.{segment}"
            dotted[rid] = order
            run["dotted_order"] = order
            return order

        for run in trace_runs:
            dotted_order(run)
        for run in trace_runs:
            client.create_run(
                id=run["id"],
                trace_id=run["trace_id"],
                dotted_order=run["dotted_order"],
                parent_run_id=run["parent_run_id"],
                name=run["name"],
                run_type=run["run_type"],
                inputs=run["inputs"],
                outputs=run.get("outputs"),
                error=run.get("error"),
                extra=run.get("extra"),
                tags=run.get("tags"),
                start_time=run["start_time"],
                end_time=run["end_time"],
                project_name=project,
            )
        if i % 5 == 0:
            print(f"  Uploaded {i}/{len(traces)} traces")

    print("Flushing...")
    client.flush()

    # `create_run` only enqueues; the POST happens on a background thread and a rejected
    # batch is logged there, not raised here. Count what actually landed before claiming
    # success. Filter by the ids just uploaded rather than listing the project and
    # intersecting locally — the project holds every previous upload too.
    expected = {run["id"] for trace_runs in traces.values() for run in trace_runs}
    landed: set[str] = set()
    for _ in range(WAIT_ATTEMPTS):
        # By project_id, never by name: the name -> id index lags create_project by
        # seconds, so a name lookup would 404 on a fresh project.
        landed = {r.id for r in query_runs(client, project_id, ids=list(expected),
                                           selects=["ID"])}
        if len(landed) == len(expected):
            break
        time.sleep(WAIT_SECONDS)

    if missing := len(expected) - len(landed):
        print(
            f"ERROR: only {len(landed)}/{len(expected)} runs landed in '{project}' "
            f"({missing} missing). Check the ingest warnings above.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Verified {len(landed)}/{len(expected)} runs landed.")

    # Ratings go on after the runs exist, against the regenerated ids. A trace that already
    # carries its rating keeps it rather than being rated twice.
    rated = 0
    for trace_runs in traces.values():
        root = next(r for r in trace_runs if r["parent_run_id"] is None)
        if not any(f["key"] == RATING_KEY for f in root["feedback"]):
            score, comment = rate_trace(trace_runs)
            root["feedback"].append(
                {"key": RATING_KEY, "score": score, "value": None, "comment": comment}
            )
            rated += score == 0

    n_feedback = 0
    for trace_runs in traces.values():
        for run in trace_runs:
            for item in run["feedback"]:
                # trace_id puts each record on the batched tracing queue instead of a
                # blocking POST per record; the flush below waits for them.
                client.create_feedback(
                    run_id=run["id"],
                    trace_id=run["trace_id"],
                    key=item["key"],
                    score=item.get("score"),
                    value=item.get("value"),
                    comment=item.get("comment"),
                )
                n_feedback += 1
    if n_feedback:
        client.flush()

    print(
        f"Done! Uploaded {len(traces)} traces to '{project}' "
        f"({n_feedback} feedback records, {rated} rated thumbs-down)."
    )
    print(f"\nPoint LangSmith Engine at '{project}' and let it scan.")


if __name__ == "__main__":
    main()
