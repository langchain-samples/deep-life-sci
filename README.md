# deepagents_testing

Minimal [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) scaffold
with traces exporting to LangSmith.

## Setup

1. Fill in `.env` (see `.env.example` for the shape):

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=deepagents_testing
   ```

2. Run it:

   ```bash
   uv run agent.py
   ```

Dependencies are already installed in `.venv` via `uv`.

## What's here

`agent.py` — a `create_deep_agent()` call with one stub tool (`get_weather`). Because it's a
deep agent, it also comes with the built-in harness tools (virtual filesystem, subagent
delegation via `task`) with no extra config.

Tracing is automatic: `load_dotenv()` runs before the agent is created, so the LangSmith
tracer is enabled by the `LANGSMITH_*` env vars. Runs show up under the
`LANGSMITH_PROJECT` project at https://smith.langchain.com.

`skills/` — on-demand skill folders loaded via `skills=["./skills/"]`, backed by a
`FilesystemBackend` (required — `skills=` with no matching backend silently no-ops). Each
skill is a directory with a `SKILL.md` (YAML frontmatter: `name`, `description`) plus any
supporting files. `skills/example-skill/` is a placeholder — fill in its `description` and
`Instructions` section, or replace it with a real skill, before relying on it.

## Next steps

- Task planning is opt-in as of `deepagents` 0.7 — pass `TodoListMiddleware()` via `middleware=`.
- Write real content into `skills/example-skill/SKILL.md` (or add sibling skill folders).
- [Customization](https://docs.langchain.com/oss/python/deepagents/customization) — subagents, backends, memory.
- [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production).
