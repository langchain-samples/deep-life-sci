"""Unit tests for `deep_life_sci` and `evals`.

A package, for the same reason `evals` is one: the modules here import shared helpers from
`tests.conftest` by name, and `conftest.py` is only importable as `tests.conftest` when the
directory is a package rooted at the repo root.

**What this suite is for, and what it deliberately is not.** `evals/` measures whether the
agent answers well, which needs models, containers and NCBI. This measures whether the code
around it behaves as its docstrings say, and makes no network call, boots no sandbox and
calls no model — so it runs on a laptop with an empty `.env` in a couple of seconds.

The tests are organised around the repo's own claims. Almost every guard in `sources/` was
written against a *measured* API behaviour that returns a wrong answer rather than an
error, and each of those is named in a docstring. A test here generally pins one of those
named failures, which is why so many of them assert something oddly specific: `PMC5379068`
is the cited paper's id that a `.//ArticleIdList` lookup really returned, and 5.7 million
is really what `cancer[nosuchfield]` matches.

`test_invariants.py` is the odd one out and the most valuable when it fails: it holds the
cross-file rules `CLAUDE.md` states, the ones no single module can enforce — a tool added
to the PTC allowlist but not to the prompt, an upload format registered in three of its
four places.
"""
