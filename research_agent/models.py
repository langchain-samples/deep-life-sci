"""Model construction via the LangSmith LLM gateway, with switchable providers.

Follows the intro-to-langsmith pattern: gateway compute is authenticated by
OPENAI_API_KEY (the `lsv2_sk_...` gateway service key, not an OpenAI key), while
LANGSMITH_API_KEY stays dedicated to tracing.

Pick a profile with MODEL_PROFILE in .env (or `MODEL_PROFILE=openai uv run agent`):

    anthropic  Sonnet 4.6 root  + Haiku 4.5 subagents   (default)
    mixed      GPT-5.6 terra    + Haiku 4.5 subagents
    openai     GPT-5.6 terra    + GPT-5.6 luna

A profile may mix providers, so the gateway path is chosen per *model* rather than per
profile — see `_provider_for`. The two paths are not cosmetically different:

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
    # terra root over Haiku leaves. The root swap is the measured win (README: 2 root
    # turns vs 6, and root occupancy is where the latency lives); the leaves stay on
    # Haiku because every sandbox-era measurement in this repo — the fan-out latency
    # distribution behind SUBAGENT_TIMEOUT_SECONDS especially — was taken against it.
    "mixed": {
        "root": "openai/gpt-5.6-terra",
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

# The judge is pinned rather than following the active profile. A sweep exists to compare
# profiles, so a judge that moved with the profile would shift the yardstick along with
# the thing being measured — a score change could then be the root model, the leaves, or
# the grader, with no way to tell which.
#
# Not Haiku 4.5, which was the obvious cheap pick: it has no effort scale at all, and the
# gateway answers `reasoning_effort` with 400 "This model does not support the effort
# parameter" (probed 2026-08-18). Its only thinking control is the deprecated
# budget_tokens form. luna takes `low` directly.
JUDGE_MODEL = "openai/gpt-5.6-luna"
JUDGE_EFFORT = "low"

# Deliberately longer than SUBAGENT_TIMEOUT_SECONDS. The judge is not on the fan-out
# critical path — one grading call per example, after the answer already exists — and a
# timeout here costs a missing score on an otherwise complete run, which is worse than
# waiting.
JUDGE_TIMEOUT_SECONDS = 60.0

# Effort levels the gateway accepts on some model. Validated here only to catch a typo
# before a sweep boots nine containers; whether a *given* model supports the level is the
# API's call (Haiku 4.5 rejects the parameter outright, Sonnet 4.6 has no `xhigh`).
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


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


def _provider_for(model: str) -> str:
    """Which gateway path a model id belongs to.

    The two paths take different id forms (see the module docstring), so the id itself
    says which one it is: bare ids like `claude-sonnet-4-6` are the Anthropic-native
    path, `provider/model` ids like `openai/gpt-5.6-terra` are the OpenAI-compatible
    one. Deciding here rather than from the profile name is what lets one profile mix
    providers, as `mixed` does.

    An id that matches neither form is rejected rather than guessed at: sending a bare
    OpenAI id down the native path returns a 501 from the gateway, which reads like an
    outage rather than a typo.
    """
    if "/" in model:
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    raise SystemExit(
        f"Cannot tell which gateway path {model!r} needs. Anthropic-native ids are "
        "bare ('claude-sonnet-4-6'); everything else must carry a provider prefix "
        "('openai/gpt-5.6-terra')."
    )


def _build(model: str, **kwargs):
    check_gateway_config()
    key = os.environ["OPENAI_API_KEY"]
    if _provider_for(model) == "anthropic":
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
    """The orchestrating agent's model.

    ROOT_EFFORT sets reasoning effort on the root half only, so one profile can be scored
    at two thinking levels without a second profile entry. It is an env var rather than a
    profile field because it is a sweep axis, not a property of the pair.

    On Anthropic ids this maps to `output_config.effort` and, because langchain-anthropic
    defaults `thinking` to adaptive whenever effort is set, it also turns thinking *on* —
    unset is not "effort=high", it is no thinking at all. That difference is the whole
    point of the axis, but it means summarized thinking lands in the root transcript, so
    expect root_context_chars to move with it.
    """
    if effort := os.environ.get("ROOT_EFFORT", "").strip().lower():
        if effort not in EFFORT_LEVELS:
            raise SystemExit(
                f"ROOT_EFFORT={effort!r} is not an effort level. "
                f"Choose one of: {', '.join(EFFORT_LEVELS)}"
            )
        kwargs.setdefault("reasoning_effort", effort)
    return _build(PROFILES[active_profile()]["root"], **kwargs)


def subagent_model(**kwargs):
    """The per-abstract analyst's model — the cheaper one of the pair.

    Timed out by default; see SUBAGENT_TIMEOUT_SECONDS for why one is mandatory here.
    `timeout` is the constructor alias on both ChatAnthropic and ChatOpenAI, so this
    works on either profile, and an explicit `timeout=` from a caller still wins.
    """
    kwargs.setdefault("timeout", SUBAGENT_TIMEOUT_SECONDS)
    return _build(PROFILES[active_profile()]["subagent"], **kwargs)


def judge_model(**kwargs):
    """The eval judge's model — configured independently of the profile under test.

    See JUDGE_MODEL for why it is pinned. JUDGE_MODEL in the environment overrides it,
    which is how you check whether a verdict is the answer's fault or the grader's.
    """
    kwargs.setdefault("timeout", JUDGE_TIMEOUT_SECONDS)
    kwargs.setdefault("reasoning_effort", os.environ.get("JUDGE_EFFORT", JUDGE_EFFORT))
    return _build(os.environ.get("JUDGE_MODEL", JUDGE_MODEL), **kwargs)


def describe() -> str:
    """One line naming the active pair and the path each half takes.

    The path is worth printing, not just the model: it is what decides whether prompt
    caching works, and a mixed profile sends its two halves different ways.
    """
    p = active_profile()
    parts = []
    for role in ("root", "subagent"):
        model = PROFILES[p][role]
        parts.append(f"{role}={model} ({_provider_for(model)})")
    if effort := os.environ.get("ROOT_EFFORT", "").strip().lower():
        parts.append(f"root_effort={effort}")
    return f"{p}: {' '.join(parts)}"
