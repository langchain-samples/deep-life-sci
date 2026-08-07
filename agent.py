"""Minimal Deep Agents example.

Follows the basic pattern from
https://docs.langchain.com/oss/python/deepagents/overview

Traces export to LangSmith automatically when LANGSMITH_TRACING=true
and LANGSMITH_API_KEY are set in .env.
"""

from dotenv import load_dotenv

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Load .env before creating the agent so the LangSmith tracer picks up
# LANGSMITH_* vars at import/instantiation time.
load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    backend=FilesystemBackend(root_dir="data", virtual_mode=True),
    skills=["./skills/"],
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(result["messages"][-1].content)
