"""PubMed research assistant — a Deep Agent for biologists.

The agent doesn't call the PubMed tools directly. It writes JavaScript in the QuickJS
interpreter and reaches them through programmatic tool calling, which lets it search,
batch-fetch, and fan out subagents across many abstracts in a single step. See
`prompts.py` for the reference snippets that teach it the pattern.

Traces export to LangSmith automatically when LANGSMITH_TRACING=true and
LANGSMITH_API_KEY are set in .env.
"""

import asyncio

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_quickjs import CodeInterpreterMiddleware

# override=True so .env wins over ambient shell values. Without it a LANGSMITH_PROJECT
# already exported in the shell silently captures this project's traces.
load_dotenv(override=True)

from models import ROOT_MODEL, SUBAGENT_MODEL, gateway_model  # noqa: E402
from prompts import ABSTRACT_ANALYST, SYSTEM_PROMPT  # noqa: E402
from pubmed import fetch_abstracts, pubmed_search  # noqa: E402

agent = create_deep_agent(
    model=gateway_model(ROOT_MODEL),
    tools=[pubmed_search, fetch_abstracts],
    system_prompt=SYSTEM_PROMPT,
    subagents=[{**ABSTRACT_ANALYST, "model": gateway_model(SUBAGENT_MODEL)}],
    # Rooted at data/, so the agent sees the abstract cache as /abstracts/<pmid>.json.
    backend=FilesystemBackend(root_dir="data", virtual_mode=True),
    middleware=[
        CodeInterpreterMiddleware(
            # Tools reach JS camelCased: pubmed_search -> tools.pubmedSearch.
            ptc=[
                "pubmed_search",
                "fetch_abstracts",
                "read_file",
                "write_file",
                "ls",
                "glob",
            ],
            # The default is 5 seconds. A fan-out across a few dozen abstracts runs for
            # minutes, so leaving this at the default would kill every real query.
            timeout=900.0,
            # Enough room for the collected fan-out results to survive to synthesis.
            max_result_chars=40_000,
            max_ptc_calls=512,
        )
    ],
)


DEMO_QUESTION = (
    "Find recent papers on base editing in the liver and tell me which ones used "
    "in vivo mouse models."
)


async def main() -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": DEMO_QUESTION}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
