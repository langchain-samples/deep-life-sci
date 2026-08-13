"""Evaluation harness for the PubMed research assistant.

A package rather than loose scripts so `evals.evaluators` is importable by name. Run its
entry points with `-m` from the repo root, which is what puts the root on `sys.path`:

    uv run python -m evals.sync          # push datasets/*.yaml to LangSmith
    uv run python -m evals.run           # score the agent against them

Deliberately outside `research_agent/`: this measures the agent, it isn't part of it, and
nothing the agent ships at deploy time should carry a test framework.
"""
