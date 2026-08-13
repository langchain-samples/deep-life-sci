"""Did the run actually hand back the artifact it was asked for?

A question that says "plot the distribution" is not answered by prose describing a plot.
The bytes never appear in the answer text — `ArtifactMiddleware` sweeps `/workspace/out`
and publishes them through the `ui` state key — so this is the only place a missing
deliverable is visible.

This also guards the failure mode the repo already documents as silent: when the `/ui/*`
proxy is misconfigured the component renders as an empty div with the run otherwise
intact. That's a frontend problem rather than an agent one, but an eval that only reads
`answer` would never notice the difference between "no chart produced" and "chart
produced, never displayed". Scoring the state key separates them.
"""

from __future__ import annotations

# Component names ArtifactMiddleware assigns, grouped by what the dataset asks for.
_KINDS = {
    "image": {"chart", "image"},
    "table": {"table"},
    "file": {"file", "download"},
}


def _published(outputs: dict) -> list[str]:
    return [n for n in (outputs.get("artifact_names") or []) if n]


def produced_expected_artifacts(run, example) -> dict:
    """True when every expected artifact kind is present, False when any is absent."""
    expected = (example.outputs or {}).get("expects_artifact") or []
    outputs = run.outputs or {}

    if not expected:
        # Most questions don't mandate a deliverable. Returning None keeps them out of
        # the aggregate instead of scoring a free 1.0 that inflates every experiment.
        return {
            "key": "produced_expected_artifacts",
            "score": None,
            "comment": "no artifact required by this example",
        }

    names = _published(outputs)
    lowered = {n.lower() for n in names}
    missing = [
        kind for kind in expected if not (_KINDS.get(kind, {kind}) & lowered)
    ]

    return {
        "key": "produced_expected_artifacts",
        "score": not missing,
        "comment": (
            f"expected {expected}, published {names or '[]'}"
            + (f"; missing {missing}" if missing else "")
        ),
    }
