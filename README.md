# Deep Life Sci

A LangChain [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) assistant for 
life scientists and chemists. The agent searches PubMed, PMC full texts, and ClinicalTrials.gov 
to answer questions and generate figures.

## Quickstart

### 1. Get the code

In the terminal:

```bash
git clone https://github.com/mcunningham1440/deep-life-sci.git

cd deep-life-sci
```

Needs [git](https://git-scm.com/downloads), which setup also uses to fetch the chat UI.

### 2. Get uv

[uv](https://docs.astral.sh/uv/) is a widely-used package manager for Python that allows the setup script to install the necessary libraries.

_On macOS / Linux:_
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

_On Windows:_
```bash
irm https://astral.sh/uv/install.ps1 | iex
```

### 3. Run the setup script
```bash
uv run scripts/setup.py
```

### 4. Add your API keys

You need a [LangSmith](https://smith.langchain.com) account. Setup prompts for two keys
from [Settings](https://smith.langchain.com/settings), which are **not** interchangeable:

- **`LANGSMITH_GATEWAY_API_KEY`** is the **gateway service key** (`lsv2_sk_...`). Every
  model call goes through the LangSmith LLM gateway, whichever provider it names.
- **`LANGSMITH_API_KEY`** is for tracing, and for provisioning the sandbox.

`ROOT_MODEL=claude-sonnet-5` in front of either command below swaps a model (see
`research_agent/models.py`).

### 5. Run

```bash
uv run scripts/dev.py                        # opens the chat UI in your browser (recommended)

# or

uv run agent "which papers base-edit PCSK9?" # runs headlessly in CLI
```

Ctrl-C to stop the running server.

## Layout

```
scripts/              setup.py (one-time setup), dev.py (chat stack), build_snapshot.py
research_agent/       the agent: assembly, entry points, tools, prompts, middleware
evals/                LangSmith datasets + evaluators
ui/                   artifact components rendered by the chat frontend
docs/                 design notes and demo questions
```

## Notes

The UI is a clone of [`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) in
`.chat-ui/`. It needs [Node](https://nodejs.org)
20+, the one thing setup won't install for you: everything else finishes without it, so
install Node and re-run to add the UI.

Full text journal articles are only available if present in PMC's open-access subset. Only abstracts are available for paywalled papers.

**[MIT Licensed](https://opensource.org/license/MIT)**