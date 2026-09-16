"""Score the agent against a LangSmith dataset, for the workshop's Test stage.

    DATASET_NAME=my-dataset uv run python -m engine_workshop.eval

Deliberately thin, and deliberately separate from `evals/`. That harness is the repo's
real one: it carries its own seeds, three evaluators and a rubric judge, and it answers
"did this change make the agent worse". This one answers a single narrower question —
*does the fix on this branch beat the bug on `main`, on the examples Engine suggested* —
so it takes the dataset by name, runs the agent, and records the output for an evaluator
attached in the LangSmith UI to grade.

Reference outputs in that dataset are **assertions**, not expected strings: "the answer
cites at least one regulatory source URL", not a paragraph to diff against. Model wording
moves run to run; the assertion is about behaviour. The *Assertions* evaluator template in
LangSmith grades them and writes `assertions_passed`. It runs on the workspace's own model
credentials, not anything in your `.env`.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

# Same capture-and-restore dance as `cli.py` and `evals/run.py`: override=True stops an
# exported LANGSMITH_PROJECT from capturing these runs, but it is too blunt for the axes a
# run is meant to vary.
from deep_life_sci.models import ENV_VARS

_OVERRIDES = {k: v for k in ENV_VARS if (v := os.environ.get(k)) is not None}
load_dotenv(override=True)
os.environ.update(_OVERRIDES)

# Evals opt out of cache expiry for the same reason `evals/run.py` does: whether an example
# refetches from NCBI would otherwise depend on how long the run before it took.
os.environ.setdefault("DEEP_LIFE_SCI_CACHE_TTL", "off")

from langsmith import aevaluate  # noqa: E402

from deep_life_sci.runner import run_once  # noqa: E402

# One container per example, each fanning out internally, so the real parallelism is
# already inside a single example. Same reasoning and same value as `evals/run.py`.
MAX_CONCURRENCY = 1


async def target(inputs: dict) -> dict:
    """Run one dataset example and return what the evaluator grades.

    Accepts both the shape `evals/` uses (`question`) and the chat shape the Engine-suggested
    examples arrive in (`messages[0].content`), so a dataset assembled either way just works.
    """
    question = inputs.get("question")
    if not question:
        messages = inputs.get("messages") or []
        question = messages[0]["content"] if messages else ""
    if not question:
        raise ValueError(f"example has no question: {sorted(inputs)}")

    result = await run_once(question)
    return {
        "output": result.answer,
        # Carried so an assertion can be written about deliverables rather than prose —
        # "produced a chart" is not recoverable from the answer text.
        "artifact_names": [a.get("name") for a in result.artifacts],
        "tool_calls": result.tool_calls,
    }


async def run() -> None:
    # Hard-fails rather than computing a default: scoring the wrong dataset quietly is worse
    # than not running.
    dataset = os.environ["DATASET_NAME"]
    # EXPERIMENT_PREFIX lets the PR workflow name the two sides distinctly
    # (pr-<n>-main vs pr-<n>-fix); a local run is the baseline.
    prefix = os.environ.get("EXPERIMENT_PREFIX", "baseline")

    results = await aevaluate(
        target,
        data=dataset,
        experiment_prefix=prefix,
        max_concurrency=MAX_CONCURRENCY,
    )
    # Printed in a shape the workflow greps to build its PR comment.
    print(f"experiment_name={results.experiment_name}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
