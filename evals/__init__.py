"""Evaluation harness for the PubMed research assistant.

A package rather than loose scripts so `evals.evaluators` is importable by name. Run its
entry points with `-m` from the repo root, which is what puts the root on `sys.path`:

    uv run python -m evals.sync          # push datasets/*.yaml to LangSmith
    uv run python -m evals.run           # score the agent against them

Deliberately outside `deep_life_sci/`: this measures the agent, it isn't part of it, and
nothing the agent ships at deploy time should carry a test framework.
"""

import os

# One LangSmith dataset per seed file, named `{prefix}-{stem}`. The prefix lives here
# because `sync.py` writes the dataset and `run.py` reads it, and a hand-copied name in
# both is how a rename silently scores the old dataset (the repo's own rule: one axis,
# one place).
#
# `EVALS_DATASET_PREFIX` overrides it, which is what makes a workspace move a config
# change. Demo hygiene in a *shared* workspace asks for a `-<username>` suffix on
# anything another engineer could re-run and clobber; a personal workspace needs no
# suffix. Neither is a code edit.
#
# Resolved per call rather than captured at import: `run.py` deliberately loads `.env`
# *after* this package is imported, so a module-level constant here would read the
# environment one step too early and miss the file.
DEFAULT_DATASET_PREFIX = "deep-life-sci"


def dataset_prefix() -> str:
    return os.environ.get("EVALS_DATASET_PREFIX", "").strip() or DEFAULT_DATASET_PREFIX


def dataset_name(stem: str = "default") -> str:
    """The LangSmith dataset name for one seed file, by its stem."""
    return f"{dataset_prefix()}-{stem}"
