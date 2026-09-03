# Deep Life Sci

A LangChain [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) assistant for biologists, bioinformaticians, and clinical researchers.

## Capabilities

* **Literature question-answering** - scan hundreds of papers and trial records at once to perform deep literature searches

* **Data analysis via code execution** - generate and execute code in a safely contained sandbox to perform almost any data analysis

* **File and figure generation** - create CSV and Excel files of data, Word docs such as clinical or lab protocols, and data visualizations and plots

## Data sources

* **PubMed** - over 40 million scientific abstracts

* **PMC full texts** - full text of over 8 million open-access papers

* **ClinicalTrials.gov** - records from over 600,000 trials

* **Web search** - agentic search over the entire open web

* **CSV/Excel upload** - upload your data files and let the agent do analyses on them

## Quickstart

### 1. Get LangSmith

You need a [LangSmith](https://smith.langchain.com) account. Setup will prompt you to add
your `LANGSMITH_API_KEY`.

Model calls go through the [LangSmith LLM gateway](https://docs.langchain.com/langsmith/llm-gateway), so your workspace also needs the
provider key behind them, added once under **Settings → Integrations → Provider Secrets**
as `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. Add whichever providers the models you run use
(the default is OpenAI).

[LangSmith Sandboxes](https://docs.langchain.com/langsmith/sandboxes) must also be enabled.
Click the Sandboxes tab on the right-hand side of your LangSmith console. If you are using a personal account on the free Dveloper tier, you will need to add a credit card to use sandboxes, but you get free 5 LangSmith Compute Units (LCUs) per month, enough for ~650 agent runs.

### 2. Get the code

In the terminal:

```bash
git clone https://github.com/langchain-samples/deep-life-sci.git

cd deep-life-sci
```

Needs [git](https://git-scm.com/downloads), which setup also uses to fetch the chat UI.

### 3. Get uv

[uv](https://docs.astral.sh/uv/) is a widely-used package manager for Python that allows the setup script to install the necessary libraries.

_On macOS / Linux:_
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

_On Windows:_
```bash
irm https://astral.sh/uv/install.ps1 | iex
```

### 4. Run the setup script
```bash
uv run scripts/setup.py
```

### 5. Run the agent

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

The UI is a modified clone of [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) in
`.chat-ui/`.

Full text journal articles are only available if present in PMC's open-access subset. Only abstracts are available for paywalled papers.

**[MIT Licensed](https://opensource.org/license/MIT)**