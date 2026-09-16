"""Capture a project's runs to `traces.json` so the workshop can replay them.

    uv run python -m engine_workshop.download_traces
    uv run python -m engine_workshop.download_traces --project my-project --tag engine-demo

Only needed when regenerating the committed reference batch. Attendees run
`upload_traces` instead.

**This prunes, and the reference workshop it is modelled on does not.** A GTM-agent trace
is a few dozen spans; traces from this agent run 46-2,661 spans and a mid-sized one is
8-13 MB on its own, because every middleware wrapper carries a copy of the QuickJS heap and
the deep-agent stack wraps each model call seven times. Captured raw, the workshop's batch
would be a few hundred megabytes, which is not something to commit.

So three reductions, all measured against the committed batch (7,858 spans -> 1,868 runs,
16 MB):

* **Keep the spans that carry the story** — the root, every `llm` and every `tool` run, plus
  any span that errored — and drop the middleware `chain` wrappers, re-parenting whatever
  survives onto its nearest surviving ancestor. That is 116-174 spans down to 22-34, and the
  result reads *better* in the Engine UI than the original: the tool calls and model turns
  are no longer buried under seven layers of `awrap_model_call`.
* **Cap every field** at `--cap` bytes. The QuickJS state and the fetched corpora are what
  make the payloads large, and neither is what Engine clusters on.
* **Drop `invocation_params`**, the tool-schema block repeated identically on every model
  span — 778 KB of the 980 KB of `extra` across two measured traces.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client  # noqa: E402

from engine_workshop import demo_project  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "engine_workshop" / "traces.json"

# Per-field ceiling. 16 KB keeps a full answer, a tool result and a model turn legible
# while cutting the heap snapshots and fetched corpora that dominate the raw size.
DEFAULT_CAP = 16_000

# Runtime tracebacks and sandbox paths bake in the absolute path of the local checkout,
# which leaks the author's home directory into a committed file. Rewrite any such path to a
# neutral `/app`. Two alternatives, longest first: the real repo root, then any POSIX
# home-style prefix ending in this directory's name, for traces captured elsewhere.
_LOCAL_PATH_RE = re.compile(
    re.escape(str(REPO_ROOT)) + r"|/(?:Users|home)/[^\s\"'\\]*?/" + re.escape(REPO_ROOT.name)
)

# `extra.runtime` records the machine the capture ran on. Library and SDK versions explain
# the shape of the runs and are worth keeping; the OS build string and Python patch level
# identify one laptop and say nothing about the agent.
_RUNTIME_FINGERPRINT_KEYS = ("platform", "runtime_version")

# `extra.invocation_params` is the full tool-schema block sent with the call, repeated
# identically on every model span — 778 KB of the 980 KB of `extra` across two measured
# traces, and not something Engine reads. Dropped whole rather than capped, because a
# half-truncated JSON schema is less useful than no schema.
_DROP_EXTRA_KEYS = ("invocation_params",)

# Feedback fetched by run id, in batches — the ids travel as repeated query params.
FEEDBACK_BATCH = 50

# Issue ids belong to the *source* project's Engine triage. Replaying them would staple
# someone else's issue onto a fresh upload.
SKIP_FEEDBACK_KEYS = {"langsmith_issue_id"}


def scrub(text: str) -> str:
    """Replace local absolute paths with a neutral `/app` prefix."""
    return _LOCAL_PATH_RE.sub("/app", text)


def scrub_metadata(extra: dict | None) -> dict | None:
    """Drop capture-environment identifiers from a run's `extra`.

    The tracing SDK copies the whole `LANGSMITH_*` environment into every run's metadata,
    which pins the capture to one workspace, project and endpoint. Those values are wrong
    the moment the traces are replayed somewhere else.
    """
    if not extra:
        return extra
    scrubbed = {k: v for k, v in extra.items() if k not in _DROP_EXTRA_KEYS}

    if metadata := extra.get("metadata"):
        cleaned = {k: v for k, v in metadata.items() if not k.startswith("LANGSMITH_")}
        # `revision_id` is the capture machine's `git describe`, so it picks up a `-dirty`
        # suffix whenever the checkout had uncommitted edits. Keep the commit, drop the
        # local working-tree state.
        revision = cleaned.get("revision_id")
        if isinstance(revision, str) and revision.endswith("-dirty"):
            cleaned["revision_id"] = revision[: -len("-dirty")]
        scrubbed["metadata"] = cleaned

    if runtime := extra.get("runtime"):
        scrubbed["runtime"] = {
            k: v for k, v in runtime.items() if k not in _RUNTIME_FINGERPRINT_KEYS
        }
    return scrubbed


def cap(value, limit: int):
    """`value` with every long string shortened, and the whole thing dropped if still huge.

    Truncation is marked rather than silent: a reader of the committed file — or of a
    replayed trace in the Engine UI — should be able to tell a short payload from a trimmed
    one, otherwise a capped tool result reads as a tool that returned nothing, which is the
    exact failure mode this repo guards against everywhere else.
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"... [+{len(value) - limit} chars]"
    if isinstance(value, dict):
        capped = {k: cap(v, limit) for k, v in value.items()}
    elif isinstance(value, list):
        capped = [cap(v, limit) for v in value]
    else:
        return value
    # A structure can stay under the per-string cap and still be enormous through sheer
    # breadth — a QuickJS heap arrives as thousands of short entries.
    if len(json.dumps(capped, default=str)) > limit * 4:
        return {
            "_dropped": f"{type(value).__name__} over {limit * 4} bytes, "
                        "trimmed for the workshop"
        }
    return capped


def keep_run(run) -> bool:
    """Is this span part of the story, or middleware scaffolding?

    `llm` and `tool` are where the agent's behaviour is; the root is the question and the
    answer. Anything that errored is kept whatever its type, since an error is the one thing
    a failure cluster must not lose.
    """
    return run.parent_run_id is None or run.run_type in ("llm", "tool") or bool(run.error)


def reparent(runs: list, kept_ids: set[str]) -> dict[str, str | None]:
    """Map each kept run to its nearest kept ancestor, so the tree stays connected."""
    by_id = {str(r.id): r for r in runs}
    new_parent: dict[str, str | None] = {}
    for run in runs:
        if str(run.id) not in kept_ids:
            continue
        parent = str(run.parent_run_id) if run.parent_run_id else None
        while parent is not None and parent not in kept_ids:
            ancestor = by_id.get(parent)
            parent = str(ancestor.parent_run_id) if ancestor and ancestor.parent_run_id else None
        new_parent[str(run.id)] = parent
    return new_parent


def fetch_feedback(client: Client, run_ids: list[str]) -> dict[str, list[dict]]:
    """Map run id -> its feedback records. Feedback is a separate resource from runs."""
    by_run = defaultdict(list)
    for i in range(0, len(run_ids), FEEDBACK_BATCH):
        for item in client.list_feedback(run_ids=run_ids[i:i + FEEDBACK_BATCH]):
            if item.key in SKIP_FEEDBACK_KEYS:
                continue
            by_run[str(item.run_id)].append(
                {"key": item.key, "score": item.score, "value": item.value,
                 "comment": item.comment}
            )
    return by_run


def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default=None, help="Source project (default: the demo project)"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output file path")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="Per-field byte ceiling")
    parser.add_argument(
        "--tag",
        default="engine-demo",
        help="Only capture traces whose root carries this tag. Pass '' for every trace.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Capture only the N most recent traces. A project accumulates every earlier "
             "capture and experiment, so this is how you take just the batch you meant to.",
    )
    args = parser.parse_args()

    project = args.project or demo_project()
    client = Client()
    project_id = client.read_project(project_name=project).id
    print(f"Fetching traces from '{project}'...")

    roots = [
        r for r in client.list_runs(project_id=project_id, is_root=True)
        if not args.tag or args.tag in (r.tags or [])
    ]
    # Newest first to apply --limit, then back to chronological order so the saved file
    # reads in the order the batch actually ran.
    roots.sort(key=lambda r: r.start_time or datetime.min, reverse=True)
    if args.limit:
        roots = roots[: args.limit]
    roots.reverse()
    print(f"  {len(roots)} root runs match")

    out: list[dict] = []
    for i, root in enumerate(roots, 1):
        # The generator pages internally; do NOT wrap it in a manual offset loop, which
        # restarts the listing every call and silently multiplies the run count.
        runs = list(client.list_runs(project_id=project_id, trace_id=str(root.trace_id)))
        kept_ids = {str(r.id) for r in runs if keep_run(r)}
        parents = reparent(runs, kept_ids)

        for run in runs:
            rid = str(run.id)
            if rid not in kept_ids:
                continue
            out.append(
                {
                    "id": rid,
                    "trace_id": str(run.trace_id),
                    "parent_run_id": parents[rid],
                    "name": run.name,
                    "run_type": run.run_type,
                    "inputs": cap(run.inputs, args.cap) or {},
                    "outputs": cap(run.outputs, args.cap),
                    "error": run.error,
                    "extra": scrub_metadata(run.extra),
                    "tags": run.tags,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                }
            )
        case = (root.metadata or {}).get("demo_case") or root.name
        print(f"  [{i}/{len(roots)}] {case:<34} {len(runs):>4} spans -> {len(kept_ids)}")

    feedback = fetch_feedback(client, [r["id"] for r in out])
    for run in out:
        run["feedback"] = feedback.get(run["id"], [])

    # Stable output: by trace, root first, then start time.
    out.sort(key=lambda r: (r["trace_id"], r["parent_run_id"] is not None, r["start_time"] or ""))

    payload = scrub(json.dumps(out, indent=2, default=serialize))
    Path(args.output).write_text(payload)

    n_traces = len({r["trace_id"] for r in out})
    print(
        f"\nSaved {len(out)} runs across {n_traces} traces "
        f"({len(payload) / 1e6:.2f} MB) to {args.output}"
    )


if __name__ == "__main__":
    main()
