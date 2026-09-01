"""Model construction via the LangSmith LLM gateway, with switchable providers.

Model calls authenticate with a **LangSmith** key, never a provider key. The gateway
resolves the actual OpenAI/Anthropic credential from the workspace's Provider Secrets
(Settings -> Integrations), so a real `sk-...` is rejected with a 403 before it reaches
any provider — nothing here reads OPENAI_API_KEY, and one LangSmith key covers models,
tracing and sandboxes alike.

`LANGSMITH_GATEWAY_API_KEY` is therefore an override rather than a second required key:
it falls back to LANGSMITH_API_KEY, and differs only when model calls should bill under a
different workspace-scoped key (which must carry `gateway:invoke`) than the one doing the
tracing. `scripts/setup.py` prompts once and writes both, so setting them apart is a hand
edit that later setup runs leave alone. Either way the key is what the OpenAI SDK would
call an api_key, so it is passed explicitly rather than through the environment.

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

import httpx

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
ROOT_EFFORT = "low"

SUBAGENT_MODEL = "openai/gpt-5.6-luna"
SUBAGENT_PROVIDER = "openai"
SUBAGENT_EFFORT = "low"

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
# and that budget is what docs/measurements.md is about.
#
# The leaves moved from Haiku 4.5 to luna on 2026-08-24 on cost, with quality held flat.
# Three sweeps over the same dataset, notes off, judge pinned: terra-low/haiku-4.5 scored
# 7/11 rubric, 9/9 citations, $1.54; terra-low/luna-low scored the same *cell for cell* --
# every seed, all three evaluators -- for $0.88, 43% less on 21% fewer tokens, at +5s median
# latency (31.0s -> 36.2s). terra-medium/haiku-4.5 was the third and bought nothing: 7/11
# again, failing the same four seeds, and it lost a citation on
# psilocybin-depression-unpublished. `ROOT_EFFORT` is `low` for that reason -- medium cost
# 53% more latency per run and fixed exactly one cell (the tpd-publication-volume
# deliverable), which is not a trade worth making the default.
#
# Two caveats attach to that $0.88. It is one run of 11 examples with no repeats, so read
# 43% as a direction rather than a constant; and the leaves are now the same model and
# effort as the judge. That is not self-grading -- the judge scores the root's final answer,
# never leaf output -- but it is a confound worth retiring with a sweep on a different judge
# before leaning on the number.
#
# What did *not* move is the more useful finding: the same four rubric seeds fail in all
# three configurations. Root effort did not touch them and neither did swapping the leaf
# model across providers, so they are a prompt, tool or criteria problem rather than a
# model-selection one.
#
# The older latency measurements in docs/measurements.md were taken against Haiku leaves.
# `SUBAGENT_MODEL=claude-haiku-4-5-20251001` restores them in one variable, but note it also
# has to drop the effort (`SUBAGENT_EFFORT=`) -- Haiku 4.5 has no effort scale and the
# gateway answers the parameter with a 400.
#
# The judge is pinned so that a score change is attributable to the pair under test rather
# than to the grader. It was already luna, and stays there.

# Per-socket deadlines on the root model's streaming call. Components, not a scalar:
# the read timeout is the gap *between* chunks, not the whole request, so 10s is a
# "no token for 10s" watchdog rather than a ceiling on a turn. A long turn streams fine.
#
# That sentence is only true of a *streaming* request, which is why `root_model` also sets
# `streaming=True` — see its docstring. Do not attach this timeout to a model that might be
# invoked non-streaming: httpx then applies `read` to the whole response body and every turn
# longer than 10s dies after three attempts at ~31.5s.
#
# This is the same pathology SUBAGENT_TIMEOUT_SECONDS below was written for, on the one
# role that never got the fix. langchain-openai forwards an unset timeout as a meaningful
# None, so the SDK client ended up as `httpx.Timeout(timeout=None)` — no connect, read,
# write or pool deadline at any layer — and a gateway that stopped responding was waited
# on forever. Observed in thread 01a045c2-ff60-7b33-b5e5-bb62573052af: the run entered the
# model node, opened one socket to the gateway, and sat there at 0% CPU with the socket in
# CLOSE_WAIT (the peer had already sent FIN). The `model` node never returned and the
# thread stayed `busy`, blocking every later turn on it.
#
# 10s is chosen against the measured gap distribution on this gateway, not by feel. Root
# streaming, max inter-chunk gap: 0.39-0.64s over 10 short calls, 0.76-1.54s over 3 replays
# of that thread's own 27-message payload. Worst observed anywhere is 1.54s, so this is
# ~6.5x headroom. One 5.49s time-to-first-token outlier was seen in a separate batch and
# never recurred in 10 samples, which is why this is 10 and not 5.
#
# The ~15s (worst 18.86s) origin tax noted under SUBAGENT_TIMEOUT_SECONDS does not appear
# on the root path — root TTFT measured 0.32-1.54s — so it looks like a fan-out effect of
# 18 concurrent calls rather than something every first call pays. If this does start
# firing spuriously on the first call of a run, that is the reason: raise `read`, don't go
# back to a scalar.
#
# max_retries stays at its default of 2, which covers request establishment: a stall
# *before* the first token is retried transparently. A stall *after* the stream has opened
# surfaces as an error instead, because partial tokens have already reached the UI. That is
# a visible failure rather than a silent recovery — and still strictly better than a hang.
ROOT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)

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


def gateway_key() -> str:
    """The LangSmith key model calls authenticate with: the override, else the main one.

    See the module docstring for why these are one key by default. Whitespace reads as
    unset so a `LANGSMITH_GATEWAY_API_KEY=` left empty in .env falls through rather than
    authenticating as the empty string.
    """
    return (
        os.environ.get("LANGSMITH_GATEWAY_API_KEY", "").strip()
        or os.environ.get("LANGSMITH_API_KEY", "").strip()
    )


def check_gateway_config() -> None:
    """Fail immediately and legibly when the gateway isn't configured.

    Without this the first model call dies deep inside the SDK with
    'Could not resolve authentication method', which is easy to mistake for a
    problem in the agent itself.
    """
    if not gateway_key():
        raise SystemExit(
            "LANGSMITH_API_KEY is not set.\n\n"
            "Every model call goes through the LangSmith LLM gateway, which authenticates "
            "with your LangSmith key (starts with 'lsv2_') and resolves the provider "
            "credential from your workspace's Provider Secrets — a provider key of your "
            "own is not what this wants. Run `uv run scripts/setup.py`, or add it to .env "
            "by hand; see .env.example. (LANGSMITH_GATEWAY_API_KEY overrides it, for "
            "billing model calls under a different workspace-scoped key.)"
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
    key = gateway_key()
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
        # Chat Completions cannot carry an image, and `read_file` on a figure returns one:
        # a tool result is a `tool`-role message whose content is text, so the block goes
        # out verbatim and the gateway answers a non-retryable 400 that kills the thread.
        # The Responses API models a tool result as `function_call_output`, which does
        # accept `input_image`. Verified: 400 on chat/completions, 200 on responses.
        use_responses_api=True,
        **kwargs,
    )


def _model_for(role: str, **kwargs):
    """One role's chat model, with its effort applied if it has one."""
    model, provider, effort = _resolve(role)
    if effort:
        kwargs.setdefault("reasoning_effort", effort)
    return _build(model, provider, **kwargs)


def root_model(**kwargs):
    """The orchestrating agent's model.

    Timed out by default; see ROOT_TIMEOUT for why one is mandatory here and why it is a
    component timeout rather than a scalar. An explicit `timeout=` from a caller wins.

    `streaming=True` is what makes that timeout safe, and it is not optional. ROOT_TIMEOUT's
    `read=10.0` is a gap-between-chunks watchdog, which is only what it means on a streaming
    request; on a non-streaming one httpx applies it to the whole response body, turning a
    10s inter-chunk allowance into a 10s ceiling on an entire turn. The callers disagreed
    about this: `cli.py` and `graph.py` reach the model through `astream`, but
    `runner.py:run_once` — the seam `evals/` attaches to — uses `ainvoke`, which issues a
    plain request. So the eval sweep, and only the eval sweep, ran the root under a 10s
    per-turn ceiling; with `max_retries` at its default 2 that surfaced as three attempts
    and an APITimeoutError at ~31.5s, uniform to within 200ms across every failure. Five of
    eleven examples died that way on 2026-09-01 — the long fan-outs, since a root turn under
    10s never noticed. `agent.py` already documents the same arithmetic against the
    general-purpose subagent's inner (non-streaming) agent; the invariant is the general
    one, so it is fixed here rather than at each caller.

    Setting it here rather than asking every caller to stream keeps the two paths identical:
    `ainvoke` on a streaming model aggregates the stream itself, so the watchdog now measures
    what its calibration assumed no matter who calls.
    """
    kwargs.setdefault("timeout", ROOT_TIMEOUT)
    kwargs.setdefault("streaming", True)
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

        root=openai/gpt-5.6-terra (openai, low) subagent=openai/gpt-5.6-luna (openai, low)

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
