# PubMed research assistant

A [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) demo for life
scientists. Ask a research question and the agent searches PubMed and retrieves abstracts and
PMC full text to answer questions and generate figures.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and a LangSmith account.

1. `cp .env.example .env` and fill it in. Note:
   **`OPENAI_API_KEY` is the LangSmith gateway service key (`lsv2_sk_...`), not an OpenAI
   key** — every model call goes through the LangSmith LLM gateway. `LANGSMITH_API_KEY` is
   for tracing and for provisioning the sandbox. `NCBI_API_KEY` is optional and raises
   NCBI's rate limit from 3 to 10 requests/sec.

2. Build a LangSmith Sandboxes snapshot:

   ```bash
   uv run scripts/build_snapshot.py     # ~100s, once
   ```

## Run

```bash
uv run agent                        # one-shot CLI
MODEL_PROFILE=mixed uv run agent    # a different model pair (profiles in research_agent/models.py)
./scripts/dev.sh                    # chat UI + graph server together, Ctrl-C stops both
```

`scripts/dev.sh` expects a clone of
[`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) at `../agent-chat-ui`
with the two local patches described in `CLAUDE.md`; `AGENT_CHAT_UI=<path>` points it
elsewhere.

`docs/` has demo questions worth starting with.

## Layout

```
research_agent/       the agent: assembly, entry points, tools, prompts, middleware
evals/                LangSmith datasets + evaluators (the closest thing to a test suite)
scripts/              dev.sh, build_snapshot.py
ui/                   artifact components rendered by the chat frontend
docs/                 design notes and demo questions
data/                 host-side abstract/PMC cache (gitignored)
```

## Limits

PubMed only, and full text only from PMC's open-access subset — paywalled papers stop at
the abstract. The sandbox has the standard scientific Python stack and nothing else: no
genomics binaries, no other data sources. The code interpreter and dynamic subagents are
both beta.
