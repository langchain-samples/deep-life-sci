"""Push the YAML seeds in `datasets/` up to LangSmith. Idempotent.

    uv run python -m evals.sync            # sync every *.yaml in datasets/
    uv run python -m evals.sync default    # just datasets/default.yaml

Why the seeds live in git rather than only in LangSmith: a dataset edited exclusively in
the web UI has no diff, no review, and no way to answer "what changed between the run
that scored 0.8 and the run that scored 0.6". The YAML is the source of truth and this
script makes LangSmith match it.

YAML rather than JSONL because the rubrics are the part that gets read and revised, and a
rubric is a paragraph and a table — one JSON line per example put those on one unwrappable
line each, which is where diffs stop being reviewable. Block scalars (`|-`) keep them
readable in the file and byte-exact in the payload.

Matching is by the `id` field, carried into example metadata as `seed_id`. That is what
makes re-running safe: an example whose question was reworded is *updated* rather than
duplicated, so its history in LangSmith stays attached to it.

Nothing here deletes. An example removed from the seed file is reported and left alone —
dropping it would silently orphan every experiment that scored it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import Client  # noqa: E402

DATASETS_DIR = Path(__file__).parent / "datasets"

# One LangSmith dataset per seed file, named for the file.
DATASET_PREFIX = "pubmed-agent"


def _load(path: Path) -> list[dict]:
    """Parse one YAML seed file, failing loudly on the example that's wrong.

    A partial sync is worse than no sync — it leaves LangSmith holding some of the edit —
    so every example is validated before the first one is pushed.
    """
    try:
        rows = yaml.safe_load(path.read_text()) or []
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path.name}: {exc}") from exc
    if not isinstance(rows, list):
        raise SystemExit(
            f"{path.name}: expected a top-level list of examples, got {type(rows).__name__}"
        )

    for n, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"{path.name}: example {n} is not a mapping")
        if "id" not in row or "question" not in row:
            raise SystemExit(f"{path.name}: example {n} needs an 'id' and a 'question'")

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise SystemExit(f"{path.name}: duplicate ids {sorted(dupes)}")
    return rows


def _split(row: dict) -> tuple[dict, dict, dict]:
    """Seed row -> (inputs, outputs, metadata).

    `question` is the only input; everything else is either a reference the evaluators
    compare against or provenance. Splitting here rather than at scoring time keeps the
    evaluators from having to know the seed file's shape.
    """
    inputs = {"question": row["question"]}
    outputs = {
        "expects_artifact": row.get("expects_artifact", []),
        "rubric": row.get("rubric", ""),
    }
    metadata = {
        "seed_id": row["id"],
        "domain": row.get("domain", ""),
        "surface": row.get("surface", ""),
        "notes": row.get("notes", ""),
    }
    return inputs, outputs, metadata


def sync_file(client: Client, path: Path) -> None:
    rows = _load(path)
    name = f"{DATASET_PREFIX}-{path.stem}"

    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            dataset_name=name,
            description=f"Seeded from {path.relative_to(path.parent.parent.parent)}",
        )
        print(f"[sync] created dataset {name}")

    existing = {
        ex.metadata.get("seed_id"): ex
        for ex in client.list_examples(dataset_id=dataset.id)
        if ex.metadata
    }

    created = updated = 0
    for row in rows:
        inputs, outputs, metadata = _split(row)
        if (found := existing.get(row["id"])) is not None:
            client.update_example(
                example_id=found.id, inputs=inputs, outputs=outputs, metadata=metadata
            )
            updated += 1
        else:
            client.create_examples(
                dataset_id=dataset.id, inputs=[inputs], outputs=[outputs], metadata=[metadata]
            )
            created += 1

    print(f"[sync] {name}: {created} created, {updated} updated, {len(rows)} total")

    if orphans := set(existing) - {r["id"] for r in rows}:
        # Reported, never deleted — see the module docstring.
        print(f"[sync] {name}: {len(orphans)} example(s) no longer in the seed file: "
              f"{sorted(o for o in orphans if o)}")


def main() -> None:
    wanted = sys.argv[1:]
    paths = sorted(DATASETS_DIR.glob("*.yaml"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted]
        if missing := set(wanted) - {p.stem for p in paths}:
            raise SystemExit(f"no such seed file(s): {sorted(missing)}")
    if not paths:
        raise SystemExit(f"no *.yaml under {DATASETS_DIR}")

    client = Client()
    for path in paths:
        sync_file(client, path)


if __name__ == "__main__":
    main()
