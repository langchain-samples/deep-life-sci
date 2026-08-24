# Deep Life Sci

A LangChain [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) assistant for 
life scientists and chemists. The agent searches PubMed, PMC full texts, and ClinicalTrials.gov 
to answer questions and generate figures. Note that full text journal articles are only available
if present in the open-access subset. Only abstracts are available for paywalled papers.

## Quickstart

### 1. Get the code

```bash
git clone https://github.com/mcunningham1440/pubmed_agent.git
cd pubmed_agent
```

Needs [git](https://git-scm.com/downloads), which setup also uses to fetch the chat UI.

### 2. Setup

You need [uv](https://docs.astral.sh/uv/) — it brings its own Python, so it is the only
prerequisite:

On macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows: 
```bash
irm https://astral.sh/uv/install.ps1 | iex
```

Then, from the repo:

```bash
uv run scripts/setup.py
```

You need a [LangSmith](https://smith.langchain.com) account. Setup prompts for two keys
from [Settings](https://smith.langchain.com/settings), which are **not** interchangeable:

- **`OPENAI_API_KEY`** is the LangSmith **gateway service key** (`lsv2_sk_...`), *not* an
  OpenAI key. Every model call goes through the LangSmith LLM gateway.
- **`LANGSMITH_API_KEY`** is for tracing, and for provisioning the sandbox.

`ROOT_MODEL=claude-sonnet-5` in front of either command below swaps a model (see
`research_agent/models.py`).

The UI is a clone of [`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) in
`.chat-ui/` (gitignored), pointed at the local server. It needs [Node](https://nodejs.org)
20+, the one thing setup won't install for you: everything else finishes without it, so
install Node and re-run to add the UI. `AGENT_CHAT_UI=<path>` uses your own checkout.

### 3. Running

```bash
uv run scripts/dev.py                        # opens the chat UI in your browser (recommended)

uv run agent "which papers base-edit PCSK9?" # runs headlessly in CLI
```

`./run_deep_life_sci` and `./run_deep_life_sci "question"` are shorthands for those two on macOS
and Linux. `NO_BROWSER=1` leaves the browser tab closed; Ctrl-C stops both servers.

## Layout

```
scripts/              setup.py (one-time setup), dev.py (chat stack), build_snapshot.py
setup_deep_life_sci   macOS/Linux shorthands for those two
research_agent/       the agent: assembly, entry points, tools, prompts, middleware
evals/                LangSmith datasets + evaluators (the closest thing to a test suite)
ui/                   artifact components rendered by the chat frontend
docs/                 design notes and demo questions
data/                 host-side abstract/PMC cache (gitignored)
```
