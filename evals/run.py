"""Score the agent against a LangSmith dataset.

    uv run python -m evals.run                          # full sweep, default pair
    uv run python -m evals.run --structural             # no judge model, no model cost
    uv run python -m evals.run --limit 3                # smoke test on three examples
    uv run python -m evals.run --seed-id tpd-publication-volume   # one example, by seed id
    ROOT_EFFORT=low uv run python -m evals.run          # score the root at low effort
    ROOT_MODEL=openai/gpt-5.6-terra uv run python -m evals.run   # score a different root

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
import os

from dotenv import load_dotenv

# Same capture-and-restore dance as cli.py, and for the same reason: override=True is what
# stops an exported LANGSMITH_PROJECT from capturing these traces, but it is too blunt for
# the settings a sweep exists to vary. Without this, the `ROOT_MODEL=...` in this module's
# own docstring is silently overwritten by the ROOT_MODEL in .env and the sweep scores the
# default pair twice while reporting two different names.
from research_agent.models import ENV_VARS

_ENV_OVERRIDES = {k: v for k in (*ENV_VARS, "RESEARCH_AGENT_CACHE_TTL",
                                 "RESEARCH_AGENT_NOTES")
                  if (v := os.environ.get(k))}
load_dotenv(override=True)
os.environ.update(_ENV_OVERRIDES)

# Evals opt out of cache expiry. The agent's cache is normally an idle TTL tied to the
# sandbox's, which is right for a demo and wrong here: whether a given example refetches
# from NCBI would then depend on how long the sweep before it took, so latency and
# `from_cache` counts would move between sweeps for reasons that have nothing to do with
# the agent. A sweep is also shared state across a concurrent run, and one example's
# expiry landing mid-fan-out of another is not a variable worth having.
#
# `setdefault`, so an explicit RESEARCH_AGENT_CACHE_TTL on the command line still wins
# (captured above, restored after .env) — the deliberate cold-cache run stays possible.
# For a fully cold corpus, point RESEARCH_AGENT_DATA_DIR at a fresh directory instead.
os.environ.setdefault("RESEARCH_AGENT_CACHE_TTL", "off")

from langsmith import aevaluate  # noqa: E402

from evals.evaluators import DEFAULT, STRUCTURAL  # noqa: E402
from research_agent.models import (  # noqa: E402
    check_gateway_config,
    describe,
    slug,
)
from research_agent.runner import run_once  # noqa: E402

DATASET = "deep-life-sci-default"

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
    parser.add_argument(
        "--dataset", default=DATASET, help=f"LangSmith dataset (default: {DATASET})"
    )
    parser.add_argument(
        "--structural",
        action="store_true",
        help="skip the LLM judge — deterministic evaluators only, no model cost",
    )
    parser.add_argument("--limit", type=int, help="score only the first N examples")
    parser.add_argument(
        "--seed-id",
        nargs="+",
        metavar="ID",
        help="score only these examples, by the `id` field in datasets/*.yaml",
    )
    parser.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENCY, help="examples in flight at once"
    )
    args = parser.parse_args()

    # Fail on a bad model config before booting the first container.
    check_gateway_config()
    pair = describe()
    print(f"[evals] {pair}")

    evaluators = STRUCTURAL if args.structural else DEFAULT
    print(f"[evals] {len(evaluators)} evaluators, dataset={args.dataset}")
    if not args.structural:
        # The effective judge, not the pinned default — JUDGE_MODEL in the environment
        # overrides it, and a sweep that printed the constant would hide that.
        print(f"[evals] {describe('judge')}")

    data = args.dataset
    if args.seed_id or args.limit:
        from langsmith import Client

        client = Client()
        examples = list(client.list_examples(dataset_name=args.dataset))
        if args.seed_id:
            wanted = set(args.seed_id)
            examples = [ex for ex in examples if (ex.metadata or {}).get("seed_id") in wanted]
            # A seed id matching nothing is a typo, not an empty result. Scoring whatever
            # else matched would report a green sweep for a question that never ran.
            if missing := wanted - {ex.metadata["seed_id"] for ex in examples}:
                raise SystemExit(f"[evals] no example with seed_id in {sorted(missing)}")
        if args.limit:
            examples = examples[: args.limit]
        print(f"[evals] limited to {len(examples)} example(s)")
        data = examples

    results = await aevaluate(
        target,
        data=data,
        evaluators=evaluators,
        # The prefix is what makes two experiments comparable in the LangSmith UI, so it
        # carries the root model and its effort — the things most commonly varied. The
        # full configuration goes in metadata, where the leaves and the judge survive too.
        experiment_prefix=f"pubmed-{slug()}",
        max_concurrency=args.concurrency,
        metadata={"models": describe("root", "subagent", "judge"), "judge": not args.structural},
    )
    # A verdict per seed, on stdout. Everything here is also in LangSmith, which is where
    # the answers, judge comments and trajectories are read from — this exists only so the
    # shape of a sweep is legible without leaving the terminal.
    rows = [
        (
            (row["example"].metadata or {}).get("seed_id"),
            {r.key: r.score for r in row["evaluation_results"]["results"]},
        )
        async for row in results
    ]

    print(f"\n[evals] {results}")
    for seed_id, scores in sorted(rows, key=lambda r: r[0] or ""):
        print(f"[evals] {seed_id:<34} " + " ".join(f"{k}={v}" for k, v in sorted(scores.items())))


if __name__ == "__main__":
    asyncio.run(main())
