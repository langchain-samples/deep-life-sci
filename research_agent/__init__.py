"""A Deep Agents research assistant for life scientists.

Two entry points build the same agent via `agent.build_agent(backend)`:

* `cli.py` — one question, one sandbox, streams the answer and exits.
* `graph.py` — the LangGraph server, with the sandbox keyed to `thread_id` so a second
  turn sees the first turn's files.

`runner.py` is the third consumer: it runs a question to completion and returns the
result as data, which is what evaluators need.

Nothing is imported eagerly here. `graph.py` is loaded by the LangGraph server and
`cli.py` boots a sandbox, so a package-level import of either would make
`import research_agent` a side-effecting operation.
"""

__all__ = ["agent", "cli", "graph", "models", "paths", "runner", "sandbox"]
