# Bio/chem research assistant

A LangChain [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) assistant for 
life scientists and chemists. The agent searches PubMed, PMC full texts, and ClinicalTrials.gov 
to answer questions and generate figures. Note that full text journal articles are only available
if present in open-access subset. Only abstracts are available for paywalled papers.

## Quickstart

### 1. Setup

```bash
./setup_sci_agent                                # once per clone: keys, deps, sandbox, chat UI
```

`setup_sci_agent` installs [uv](https://docs.astral.sh/uv/) if you don't have it (uv brings
its own Python), writes `.env`, installs dependencies, builds the sandbox the agent's Python
runs in, and sets up the chat UI. Re-running it is cheap and skips whatever is already done.
`run_sci_agent` only runs the agent — it never installs.

You need a [LangSmith](https://smith.langchain.com) account. Setup prompts for two keys
from [Settings](https://smith.langchain.com/settings), which are **not** interchangeable:

- **`OPENAI_API_KEY`** is the LangSmith **gateway service key** (`lsv2_sk_...`), *not* an
  OpenAI key. Every model call goes through the LangSmith LLM gateway.
- **`LANGSMITH_API_KEY`** is for tracing, and for provisioning the sandbox.

`ROOT_MODEL=openai/gpt-5.6-terra ./run_sci_agent ...` swaps a model (see
`research_agent/models.py`). `--help` on either script covers the rest.

The UI is a clone of [`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) in
`.chat-ui/` (gitignored), pointed at the local server. It needs [Node](https://nodejs.org)
20+, the one thing setup won't install for you: everything else finishes without it, so
install Node and re-run to add the UI. `AGENT_CHAT_UI=<path>` uses your own checkout.

### 2. Running

```bash
./run_sci_agent                                    # opens the chat UI in your browser (recommended)

./run_sci_agent "which papers base-edit PCSK9?"    # runs headlessly in CLI
```

## Layout

```
setup_sci_agent       one-time setup; run_sci_agent starts the agent
research_agent/       the agent: assembly, entry points, tools, prompts, middleware
evals/                LangSmith datasets + evaluators (the closest thing to a test suite)
scripts/              dev.sh, build_snapshot.py
ui/                   artifact components rendered by the chat frontend
docs/                 design notes and demo questions
data/                 host-side abstract/PMC cache (gitignored)
```
