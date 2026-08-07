"""Model construction via the LangSmith LLM gateway.

Follows the intro-to-langsmith pattern: gateway compute is authenticated by
OPENAI_API_KEY (the `lsv2_sk_...` gateway service key, not an OpenAI key), while
LANGSMITH_API_KEY stays dedicated to tracing so runs land in a readable workspace.

We use the gateway's **Anthropic-native** path rather than its OpenAI-compatible one,
because prompt caching only survives on the native path. Verified against the live
gateway:

    /v1/chat/completions   (OpenAI-compatible)  -> cached_tokens: 0 on a repeated
                                                   14k-token prefix, with and without
                                                   an explicit cache_control block.
                                                   The gateway drops cache_control here.
    /anthropic/v1/messages (Anthropic-native)   -> cache_creation_input_tokens: 14413
                                                   then cache_read_input_tokens: 14413.

That matters a lot here: the root agent re-reads a growing transcript every turn, and
uncached it was ~96k input tokens per run — 82% of the bill.

Two quirks of the native path, both discovered the hard way:
  - The base URL must NOT include `/v1`; the Anthropic SDK appends it, and
    `/anthropic/v1/v1/messages` returns a 501 "path not allow-listed".
  - Model ids are bare here (`claude-sonnet-4-6`), not provider-prefixed. The
    `anthropic/`-prefixed form is only for the OpenAI-compatible path.
"""

import os

from langchain_anthropic import ChatAnthropic

DEFAULT_BASE_URL = "https://gateway.smith.langchain.com/anthropic"

ROOT_MODEL = "claude-sonnet-4-6"
SUBAGENT_MODEL = "claude-haiku-4-5-20251001"


def check_gateway_config() -> None:
    """Fail immediately and legibly when the gateway isn't configured.

    Without this the first model call dies deep inside the SDK with
    'Could not resolve authentication method', which is easy to mistake for a
    problem in the agent itself.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n\n"
            "This project calls models through the LangSmith LLM gateway, so this "
            "should be the gateway service key (starts with 'lsv2_sk_'), not an "
            "OpenAI key. Add it to .env — see .env.example."
        )


def gateway_model(model: str, **kwargs) -> ChatAnthropic:
    """Build a chat model backed by the LangSmith gateway's Anthropic path."""
    check_gateway_config()
    return ChatAnthropic(
        model=model,
        base_url=os.environ.get("LANGSMITH_GATEWAY_ANTHROPIC_URL", DEFAULT_BASE_URL),
        api_key=os.environ["OPENAI_API_KEY"],
        **kwargs,
    )
