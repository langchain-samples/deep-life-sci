"""Model construction via the LangSmith LLM gateway, with switchable providers.

Follows the intro-to-langsmith pattern: gateway compute is authenticated by
OPENAI_API_KEY (the `lsv2_sk_...` gateway service key, not an OpenAI key), while
LANGSMITH_API_KEY stays dedicated to tracing.

Each role is configured by three independent env vars, defaulting to the nine constants
below:

    ROOT_MODEL       SUBAGENT_MODEL       JUDGE_MODEL       gateway model id
    ROOT_PROVIDER    SUBAGENT_PROVIDER    JUDGE_PROVIDER    anthropic | openai
    ROOT_EFFORT      SUBAGENT_EFFORT      JUDGE_EFFORT      low | medium | high | xhigh | max

Three axes rather than one named profile because they vary independently, and a name that
covers combinations needs one entry per combination — a root swap, a leaf swap and a
thinking level are three experiments, and the profile enum could express only the first
two. `ROOT_EFFORT` already sat outside the naming for exactly that reason.

`{ROLE}_PROVIDER` names the gateway path, per role, which is how one run mixes providers
across roles. Swapping only `{ROLE}_MODEL` still works: a model with no path named
alongside it takes the path its id's *form* implies (see `_resolve`). The two paths are
not cosmetically different:

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

# --- The nine model settings: three roles x three axes -----------------------------
# Each is overridden by the identically named env var, so `ROOT_EFFORT=high uv run agent`
# needs no code change. `""` means unset, which is not the same thing on both paths: on an
# Anthropic model it is *no thinking at all* rather than a default level (see `_effort`),
# while an OpenAI model falls back to whatever the provider does by default. A provider
# belongs to the model beside it: swap the model in the environment without naming a path
# and the path comes from the new id's form instead (see `_resolve`).

ROOT_MODEL = "openai/gpt-5.6-terra"
ROOT_PROVIDER = "openai"
ROOT_EFFORT = ""

SUBAGENT_MODEL = "claude-haiku-4-5-20251001"
SUBAGENT_PROVIDER = "anthropic"
SUBAGENT_EFFORT = ""  # Haiku 4.5 has no effort scale; the gateway 400s on the parameter

JUDGE_MODEL = "openai/gpt-5.6-luna"
JUDGE_PROVIDER = "openai"
JUDGE_EFFORT = "low"

# Why these. terra and Sonnet 5 score the same as the root: over the eval dataset both hit
# 7/11 rubric and 8/9 citations, failing the same four rubric seeds as each other (probed
# 2026-08-23, both at ROOT_EFFORT=medium; terra's one regression was the missing deliverable
# on tpd-publication-volume). A tie on quality makes it a cost decision, and every
# head-to-head in docs/measurements.md puts terra far ahead per paper. Swapping back is one
# variable: `ROOT_MODEL=claude-sonnet-5` is the previous default, `claude-sonnet-4-6` the one
# before it, and both share these leaves, so either isolates the root — but watch root
# context when you do, because Sonnet 5 costs 1.9-2.6x Sonnet 4.6 there (fmt-cdiff 86k ->
# 214k chars) for fan-outs 22-62% faster (198s -> 76s on semaglutide-weightloss-boxplot),
# and that budget is what docs/measurements.md is about. The leaves stay on Haiku because
# every latency measurement in this repo was taken against it. The judge is pinned so that a
# score change is attributable to the pair under test rather than to the grader, and is
# luna rather than the cheaper Haiku 4.5 because Haiku has no effort scale at all (gateway
# 400 "This model does not support the effort parameter", probed 2026-08-18).

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

# Deliberately longer than SUBAGENT_TIMEOUT_SECONDS. The judge is not on the fan-out
# critical path — one grading call per example, after the answer already exists — and a
# timeout here costs a missing score on an otherwise complete run, which is worse than
# waiting.
JUDGE_TIMEOUT_SECONDS = 60.0

# Effort levels the gateway accepts on some model. Validated here only to catch a typo
# before a sweep boots nine containers; whether a *given* model supports the level is the
# API's call (Haiku 4.5 rejects the parameter outright, Sonnet 4.6 has no `xhigh`).
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The two gateway paths, which is all a provider selects here — see the module docstring.
PROVIDERS = ("anthropic", "openai")

# The nine settings above, indexed for lookup by role and axis.
DEFAULTS = {
    "root": {"model": ROOT_MODEL, "provider": ROOT_PROVIDER, "effort": ROOT_EFFORT},
    "subagent": {
        "model": SUBAGENT_MODEL,
        "provider": SUBAGENT_PROVIDER,
        "effort": SUBAGENT_EFFORT,
    },
    "judge": {"model": JUDGE_MODEL, "provider": JUDGE_PROVIDER, "effort": JUDGE_EFFORT},
}

# Every env var this module reads, so `cli.py` and `evals/run.py` can preserve them across
# their `load_dotenv(override=True)` without hand-maintaining a second copy of the list —
# a copy that drifts is how a `ROOT_MODEL=...` on the command line silently loses to .env.
ENV_VARS = tuple(
    f"{role.upper()}_{axis.upper()}"
    for role in DEFAULTS
    for axis in ("model", "provider", "effort")
)


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


def _infer_provider(model: str) -> str:
    """Which gateway path a model id looks like, or "" when it says nothing.

    The two paths take different id forms (see the module docstring), so the id itself
    usually says which one it is: bare ids like `claude-sonnet-4-6` are the
    Anthropic-native path, `provider/model` ids like `openai/gpt-5.6-terra` are the
    OpenAI-compatible one.
    """
    if "/" in model:
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return ""


def _provider_for(role: str, model: str, declared: str) -> str:
    """Validate one role's gateway path against the form of its model id.

    A named path wins where the form says nothing, which is the escape hatch: a model id in
    neither known form is usable by naming its path rather than by editing this module.
    Where the form *does* say something and the two disagree, that is an error rather than
    a preference — sending an id down the wrong path returns a 501 that reads like an
    outage, or silently drops prompt caching.
    """
    inferred = _infer_provider(model)
    if declared:
        if declared not in PROVIDERS:
            raise SystemExit(
                f"{role.upper()}_PROVIDER={declared!r} is not a gateway path. "
                f"Choose one of: {', '.join(PROVIDERS)}"
            )
        if inferred and inferred != declared:
            raise SystemExit(
                f"{role.upper()}_MODEL={model!r} is a {inferred!r} id but "
                f"{role.upper()}_PROVIDER says {declared!r}. The paths take different id "
                "forms: anthropic wants a bare id ('claude-sonnet-5'), openai a prefixed "
                "one ('openai/gpt-5.6-terra'). Fix one or the other, or unset the "
                "provider and let the form decide."
            )
        return declared
    if inferred:
        return inferred
    raise SystemExit(
        f"Cannot tell which gateway path {role.upper()}_MODEL={model!r} needs. "
        "Anthropic-native ids are bare ('claude-sonnet-5'); everything else must carry a "
        f"provider prefix ('openai/gpt-5.6-terra'). Or set {role.upper()}_PROVIDER "
        f"explicitly to one of: {', '.join(PROVIDERS)}"
    )


def _setting(role: str, axis: str) -> str:
    """One config value for one role: `{ROLE}_{AXIS}` in the environment, else the default.

    An env var set to whitespace reads as unset rather than as an empty model id.
    """
    return os.environ.get(f"{role.upper()}_{axis.upper()}", "").strip() or DEFAULTS[role][axis]


def _effort(role: str) -> str:
    """`{ROLE}_EFFORT`, validated locally.

    Validated here only to catch a typo before a sweep boots nine containers; whether a
    *given* model supports the level is the API's call (Haiku 4.5 rejects the parameter
    outright, Sonnet 4.6 has no `xhigh`).

    On Anthropic ids this maps to `output_config.effort` and, because langchain-anthropic
    defaults `thinking` to adaptive whenever effort is set, setting it also turns thinking
    *on* — unset is not "effort=high", it is no thinking at all. That difference is the
    whole point of the axis, but it means summarized thinking lands in the transcript, so
    expect root_context_chars to move with ROOT_EFFORT.
    """
    effort = _setting(role, "effort").lower()
    if effort and effort not in EFFORT_LEVELS:
        raise SystemExit(
            f"{role.upper()}_EFFORT={effort!r} is not an effort level. "
            f"Choose one of: {', '.join(EFFORT_LEVELS)}"
        )
    return effort


def _resolve(role: str) -> tuple[str, str, str]:
    """(model id, gateway path, effort) for one role, from the env or the defaults above."""
    model = _setting(role, "model")
    provider = os.environ.get(f"{role.upper()}_PROVIDER", "").strip().lower()
    if not provider:
        # A default provider describes the default model it sits beside, so it does not
        # survive that model being replaced: `ROOT_MODEL=openai/gpt-5.6-terra` alone would
        # otherwise contradict ROOT_PROVIDER and refuse to run.
        provider = (
            DEFAULTS[role]["provider"]
            if model == DEFAULTS[role]["model"]
            else _infer_provider(model)
        )
    return model, _provider_for(role, model, provider), _effort(role)


def _build(model: str, provider: str, **kwargs):
    check_gateway_config()
    key = os.environ["OPENAI_API_KEY"]
    if provider == "anthropic":
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


def _model_for(role: str, **kwargs):
    """One role's chat model, with its effort applied if it has one."""
    model, provider, effort = _resolve(role)
    if effort:
        kwargs.setdefault("reasoning_effort", effort)
    return _build(model, provider, **kwargs)


def root_model(**kwargs):
    """The orchestrating agent's model."""
    return _model_for("root", **kwargs)


def subagent_model(**kwargs):
    """The per-abstract analyst's model — the cheaper one of the pair.

    Timed out by default; see SUBAGENT_TIMEOUT_SECONDS for why one is mandatory here.
    `timeout` is the constructor alias on both ChatAnthropic and ChatOpenAI, so this works
    whichever path the leaves take, and an explicit `timeout=` from a caller still wins.
    """
    kwargs.setdefault("timeout", SUBAGENT_TIMEOUT_SECONDS)
    return _model_for("subagent", **kwargs)


def judge_model(**kwargs):
    """The eval judge's model — configured independently of the pair under test.

    See JUDGE_MODEL for why it is pinned. JUDGE_MODEL in the environment overrides it,
    which is how you check whether a verdict is the answer's fault or the grader's.
    """
    kwargs.setdefault("timeout", JUDGE_TIMEOUT_SECONDS)
    return _model_for("judge", **kwargs)


def describe(*roles: str) -> str:
    """One line saying which model is doing what, for startup logs and eval metadata.

    Names the model, its gateway path and its effort for each role asked for, defaulting to
    the pair that does the work:

        root=openai/gpt-5.6-terra (openai) subagent=claude-haiku-4-5-20251001 (anthropic)

    The path is printed and not just the model because it decides whether prompt caching
    works, and because nothing stops two roles taking different ones.
    """
    parts = []
    for role in roles or ("root", "subagent"):
        model, provider, effort = _resolve(role)
        parts.append(f"{role}={model} ({provider}" + (f", {effort})" if effort else ")"))
    return " ".join(parts)


def slug() -> str:
    """A short name for the current configuration, used as the eval experiment prefix.

    Just the root model and its effort — `gpt-5.6-terra`, or `claude-sonnet-5-medium` —
    because the root is what a sweep almost always varies. Two sweeps that differ only in
    their leaves therefore share a prefix and sort together in LangSmith, which is the
    comparison you wanted anyway; `describe()` goes into the experiment metadata, so the
    leaves and the judge are still recorded.
    """
    model, _, effort = _resolve("root")
    return f"{model.split('/')[-1]}" + (f"-{effort}" if effort else "")
