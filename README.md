<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/deep-helix-banner-dark.svg">
  <img src="assets/deep-helix-banner-light.svg" alt="Deep Life Sci" width="560">
</picture>

An open-source [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) assistant for biologists, bioinformaticians, and clinical researchers.

[![License](https://img.shields.io/github/license/langchain-samples/deep-life-sci?color=4f46e5)](LICENSE) [![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-4f46e5)](pyproject.toml) [![Built with deepagents](https://img.shields.io/badge/built%20with-deepagents-0d9488)](https://docs.langchain.com/oss/python/deepagents/overview) [![Traced with LangSmith](https://img.shields.io/badge/traced%20with-LangSmith-0d9488)](https://smith.langchain.com)

[Quickstart](#quickstart) · [Capabilities](#capabilities) · [Data sources](#data-sources)

</div>

## Capabilities

* **Literature question-answering** - scan hundreds of papers and trial records at once to perform deep literature searches

* **Data analysis via code execution** - generate and execute code in a safely contained [LangSmith Sandbox](https://docs.langchain.com/langsmith/sandboxes) to perform almost any data analysis

* **File and figure generation** - create CSV and Excel files of data, Word docs such as clinical or lab protocols, and data visualizations and plots

## Data sources

* **PubMed** - over 29 million scientific abstracts

* **PMC full texts** - full text of over 8 million open-access papers

* **ClinicalTrials.gov** - records from over 600,000 trials

* **Web search** - agentic search over the entire open web

* **File upload** - attach a spreadsheet, a reference-manager export (.nbib/.ris/.bib),
  a PDF, a compound set (.sdf/.smi), FASTA/GenBank, or a figure. Tables get analysed;
  a bibliography becomes a corpus the agent hydrates from PubMed and reads across

## Quickstart

### 1. Get and configure LangSmith

You need a [LangSmith](https://smith.langchain.com) account. Setup will prompt you to add
your `LANGSMITH_API_KEY`.

[LangSmith Sandboxes](https://docs.langchain.com/langsmith/sandboxes) must also be enabled from the Sandboxes tab in your LangSmith console. On a personal account on the free Developer tier you will need to add a credit card to use sandboxes, but you get free 5 LangSmith Compute Units (LCUs) per month, enough for ~650 agent runs.

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

The chat UI needs Node.js 20.9 or newer. If yours is missing or older, setup installs a
private copy inside the repo; nothing else on your machine changes.

### 5. Configure your models

Setup will ask you how the agent should reach its models:

- **LangSmith LLM Gateway (recommended).** Model calls go through the [LangSmith LLM gateway](https://docs.langchain.com/langsmith/llm-gateway), so your workspace also needs the
  provider key behind them, added once under **Settings → Integrations → Provider Secrets**
  as `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. Add whichever providers the models you run use.
- **Your own Anthropic, OpenAI or Amazon Bedrock credentials.** Model calls go to that provider directly.
  For Bedrock, setup asks for a Bedrock API key (or uses the AWS CLI profile you're signed in with) and a region.
  If `models.yaml` uses another provider's models, setup offers to switch them.
  LangSmith still handles tracing and sandboxes, so you still need its API key.

To change your answer later, set `MODEL_ACCESS` in `.env` to `gateway` or `direct` and run setup again.

Then, in the `models.yaml` file, configure the models you want to use for the agent. Defaults are OpenAI; recommended Anthropic alternatives are shown in comments.

### 6. Run the agent

```bash
uv run scripts/dev.py                        # opens the chat UI in your browser (recommended)

# or

uv run agent "which papers base-edit PCSK9?" # runs headlessly in CLI
```

Ctrl-C to stop the running server.

## Deploy to LangSmith

To run the agent server in the cloud with [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments):

```bash
uv run scripts/deploy.py
```

This creates a deployment named `deep-life-sci-cloud`, or updates it if it exists (use `--name` to pick
another). Models are whatever `models.yaml` says when you deploy; edit it and deploy
again to change them.

*Cost:* the default is a Dedicated Small deployment, always on, about $390 a month. For a cheaper
deployment to try things out, use `--type serverless`: about $62 a month at most, depending on usage,
and the LangSmith Plus plan includes one free. It scales to zero when idle, so the first request after
a quiet spell is slower. Model calls and sandboxes are billed separately. See
[pricing](https://www.langchain.com/pricing) for current rates.

*Sign-in:* set `OIDC_ISSUER` and `OIDC_CLIENT_ID` in `.env` (see `.env.example`) before
deploying, and the deployment serves the chat UI at `<deployment URL>/app/`, where people
sign in with your organization's identity provider: Entra ID, Okta, or any other OpenID
Connect provider. Each person sees only their own conversations.

Without them, the deployment accepts only LangSmith API keys from your workspace, and you chat with it 
from the local chat UI using the deployment URL shown on its LangSmith page:

```bash
uv run scripts/dev.py --remote https://<your-deployment>.langgraph.app
```

A deployment accepts requests only with a LangSmith API key from your workspace, so this is
for your own use.

*Retention:* a deployment deletes conversations after
90 days without use. To change, change `THREAD_TTL_MINUTES` in `deep_life_sci/paths.py`
and both `checkpointer.ttl` and `store.ttl` in `langgraph.json` together.

## Coming soon

* Additional scientific data sources

## Disclaimer

This is a demonstration project, intended for research and educational use. Its answers are
generated by language models from published literature and trial registries, and may be
incomplete, outdated, or wrong. **It is not medical advice, and must not be used for clinical
decision-making, diagnosis, or treatment.**

## Notes

*Models:* Deep Life Sci runs on GPT-5.6 Terra with High effort by default. To change the model or effort for the main agent, subagents or web search, edit [`models.yaml`](models.yaml); changes apply to your next message. To add a provider, including a custom OpenAI-compatible endpoint, configure it in LangSmith under **LLM Gateway**. The wrench under the chat box shows what each role is running.

The chat UI in `frontend/` began as [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui).

Full text journal articles are only available if present in PMC's open-access subset. Only abstracts are available for paywalled papers.

**[MIT Licensed](https://opensource.org/license/MIT)**