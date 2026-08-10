"""Model construction via the LangSmith LLM gateway, with switchable providers.

Follows the intro-to-langsmith pattern: gateway compute is authenticated by
OPENAI_API_KEY (the `lsv2_sk_...` gateway service key, not an OpenAI key), while
LANGSMITH_API_KEY stays dedicated to tracing.

Pick a profile with MODEL_PROFILE in .env (or `MODEL_PROFILE=openai uv run agent.py`):

    anthropic  Sonnet 4.6 root  + Haiku 4.5 subagents   (default)
    openai     GPT-5.6 terra    + GPT-5.6 luna

The two profiles reach the gateway by different paths, and the difference is not
cosmetic:

    /anthropic/v1/messages  (native)            prompt caching WORKS
    /v1/chat/completions    (OpenAI-compatible) prompt caching for Anthropic models
                                                does NOT work — verified: cached_tokens
                                                stays 0 on a repeated 14k-token prefix
                                                even with an explicit cache_control
                                                block, which the gateway drops.

So Anthropic models must go native or they silently lose caching. OpenAI models only
have the OpenAI-compatible path, where caching is automatic and server-side.

Two quirks of the native path, both found the hard way:
  - The base URL must NOT include `/v1`; the Anthropic SDK appends it and
    `/anthropic/v1/v1/messages` returns 501 "path not allow-listed".
  - Model ids are bare there (`claude-sonnet-4-6`). The `anthropic/`-prefixed form is
    only for the OpenAI-compatible path.
"""

import os

ANTHROPIC_BASE_URL = "https://gateway.smith.langchain.com/anthropic"
OPENAI_BASE_URL = "https://gateway.smith.langchain.com/v1"

PROFILES = {
    "anthropic": {
        "root": "claude-sonnet-4-6",
        "subagent": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "root": "openai/gpt-5.6-terra",
        "subagent": "openai/gpt-5.6-luna",
    },
}
DEFAULT_PROFILE = "anthropic"

# Wall-clock ceiling on a single analyst request. Nothing else imposes one.
#
# langchain-anthropic treats an unset timeout as a *meaningful* None and forwards it
# (chat_models.py:1269), so the client ends up as `httpx.Timeout(timeout=None)` — no
# connect, read, write or pool deadline at any layer. The Anthropic SDK's own 10-minute
# DEFAULT_TIMEOUT never applies either: messages.py:1030 only computes a request timeout
# when `client.timeout == DEFAULT_TIMEOUT`, and ours is None. A socket that stalls is
# then waited on forever. In trace 019fe907-abed-70c0-b589-2cfcb3ef5d2b one
# abstract-analyst's ChatAnthropic call hung with 13 runs blocked behind it and was
# still pending 40+ minutes later; CodeInterpreterMiddleware(timeout=900) did not free
# it, because the eval was inside the same stuck await.
#
# 30s is chosen against the measured distribution, not by feel: warm fan-out waves run
# ~1.2-4.0s per call, and the first wave of a run pays a flat ~15s tax at the origin
# (the gateway reports it as `server-timing: x-originResponse;dur=16616`), with the
# slowest first-wave call observed at 18.86s. That leaves ~11s of headroom over the
# worst legitimate case. Anything slower is the pathology this is here to kill.
#
# max_retries stays at its default of 2, which is what makes this safe: a timeout is a
# retryable failure, so a single stalled socket costs one analyst ~30s and a retry
# rather than the whole run. Raise this if a profile's subagent legitimately runs longer.
SUBAGENT_TIMEOUT_SECONDS = 30.0


def active_profile() -> str:
    name = os.environ.get("MODEL_PROFILE", DEFAULT_PROFILE).strip().lower()
    if name not in PROFILES:
        raise SystemExit(
            f"MODEL_PROFILE={name!r} is not a known profile. "
            f"Choose one of: {', '.join(PROFILES)}"
        )
    return name


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


def _build(model: str, **kwargs):
    check_gateway_config()
    key = os.environ["OPENAI_API_KEY"]
    if active_profile() == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            base_url=os.environ.get("LANGSMITH_GATEWAY_ANTHROPIC_URL", ANTHROPIC_BASE_URL),
            api_key=key,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=os.environ.get("LANGSMITH_GATEWAY_BASE_URL", OPENAI_BASE_URL),
        api_key=key,
        **kwargs,
    )


def root_model(**kwargs):
    """The orchestrating agent's model."""
    return _build(PROFILES[active_profile()]["root"], **kwargs)


def subagent_model(**kwargs):
    """The per-abstract analyst's model — the cheaper one of the pair.

    Timed out by default; see SUBAGENT_TIMEOUT_SECONDS for why one is mandatory here.
    `timeout` is the constructor alias on both ChatAnthropic and ChatOpenAI, so this
    works on either profile, and an explicit `timeout=` from a caller still wins.
    """
    kwargs.setdefault("timeout", SUBAGENT_TIMEOUT_SECONDS)
    return _build(PROFILES[active_profile()]["subagent"], **kwargs)


def describe() -> str:
    p = active_profile()
    return f"{p}: root={PROFILES[p]['root']} subagent={PROFILES[p]['subagent']}"
