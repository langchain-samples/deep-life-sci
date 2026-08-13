"""Score the agent against a LangSmith dataset.

    uv run python -m evals.run                          # full sweep, default profile
    uv run python -m evals.run --structural             # no judge model, no model cost
    uv run python -m evals.run --limit 3                # smoke test on three examples
    MODEL_PROFILE=mixed uv run python -m evals.run      # score a different model pair

Every run gets its own sandbox per example. That is the expensive choice and it is the
right one: `evals/` scores `artifact_names`, and a shared container would let one
example's leftover `/workspace/out` files be swept and attributed to the next.

Concurrency is capped low on purpose. Each example boots a container and fans out to a
few dozen subagents, so the real parallelism is already inside a single example — raising
this multiplies containers, not throughput, and NCBI's rate limit is shared across all of
them.
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith import aevaluate  # noqa: E402

from evals.evaluators import DEFAULT, STRUCTURAL  # noqa: E402
from research_agent.models import check_gateway_config, describe  # noqa: E402
from research_agent.runner import run_once  # noqa: E402

DATASET = "pubmed-agent-default"

# One container per example, each fanning out internally. One at a time is what a laptop's
# connection pool and NCBI's 10 req/sec stay comfortable with; raise it with --concurrency.
MAX_CONCURRENCY = 1


async def target(inputs: dict) -> dict:
    """The system under test, in the shape `aevaluate` expects.

    Agent errors are returned rather than raised. A single example that dies on a
    transient sandbox failure should score zero and let the sweep finish, not abort
    nineteen other results — which is the same trade `ResilientSandbox` makes one layer
    down.

    A malformed example is the other kind of failure and is raised, not returned: it
    means the dataset holds something this target cannot score, and swallowing it hides a
    bad example behind `error_rate: 0` and a silently missing score. `aevaluate` records
    the raise against that example and carries on with the rest.
    """
    if "question" not in inputs:
        raise KeyError(
            f"example has no 'question' input — got keys {sorted(inputs)}. "
            f"Is it from another dataset?"
        )
    try:
        result = await run_once(inputs["question"], quiet=True)
    except Exception as exc:  # noqa: BLE001 - one bad example must not kill the sweep
        return {"answer": "", "error": f"{type(exc).__name__}: {exc}"}
    return result.as_dict()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, help=f"LangSmith dataset (default: {DATASET})")
    parser.add_argument(
        "--structural",
        action="store_true",
        help="skip the LLM judge — deterministic evaluators only, no model cost",
    )
    parser.add_argument("--limit", type=int, help="score only the first N examples")
    parser.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENCY, help="examples in flight at once"
    )
    args = parser.parse_args()

    # Fail on a bad model config before booting the first container.
    check_gateway_config()
    profile = describe()
    print(f"[evals] {profile}")

    evaluators = STRUCTURAL if args.structural else DEFAULT
    print(f"[evals] {len(evaluators)} evaluators, dataset={args.dataset}")

    data = args.dataset
    if args.limit:
        from langsmith import Client

        client = Client()
        examples = list(client.list_examples(dataset_name=args.dataset, limit=args.limit))
        print(f"[evals] limited to {len(examples)} example(s)")
        data = examples

    results = await aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        # The prefix is what makes two experiments comparable in the LangSmith UI, so it
        # carries the model profile — the single most common thing being varied.
        experiment_prefix=f"pubmed-{profile.split(':')[0]}",
        max_concurrency=args.concurrency,
        metadata={"profile": profile, "judge": not args.structural},
    )
    print(f"\n[evals] {results}")


if __name__ == "__main__":
    asyncio.run(main())
