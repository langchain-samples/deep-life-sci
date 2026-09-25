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

Each role is configured by three independent env vars, defaulting to the role's entry in
`models.yaml` at the repository root (the reasons for those defaults are recorded below):

    ROOT_MODEL      SUBAGENT_MODEL      SEARCH_MODEL      JUDGE_MODEL      model id
    ROOT_PROVIDER   SUBAGENT_PROVIDER   SEARCH_PROVIDER   JUDGE_PROVIDER   anthropic|openai
    ROOT_EFFORT     SUBAGENT_EFFORT     SEARCH_EFFORT     JUDGE_EFFORT     see `_check_effort`

`search` is the role behind `sources/web.py`'s `web_search` tool: a model with the
provider's own server-side web search bound to it, called from inside the tool so its
output lands in the JS heap rather than in root context. It is a role of its own rather
than a reuse of `subagent` because web search is the one capability that is not uniform
across ids — the spec differs per gateway path (`WEB_SEARCH_SPECS`), a leaf swap must not
be able to take the tool out with it, and one search legitimately runs 4x longer than the
30s a leaf gets (`SEARCH_TIMEOUT_SECONDS`).

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
                                                block.

So Anthropic models must go native or they silently lose caching. OpenAI models only
have the OpenAI-compatible path, where caching is automatic and server-side.

Two things to know about the native path:
  - The base URL must NOT include `/v1`; the Anthropic SDK appends it and
    `/anthropic/v1/v1/messages` returns 501 "path not allow-listed".
  - Model ids are bare there (`claude-sonnet-4-6`). The `anthropic/`-prefixed form is
    only for the OpenAI-compatible path.

Amazon Bedrock is reached through the same gateway and key, with a `bedrock/` prefix and
the workspace's `AWS_BEARER_TOKEN_BEDROCK` Provider Secret; nothing here holds AWS
credentials. Claude on Bedrock takes the Anthropic Messages format on the gateway's
standard endpoint (`_messages_base_url`), so caching and effort work as they do natively;
every other Bedrock model takes the OpenAI-compatible path. Effort is checked against the
profile of the model behind the Bedrock id. Bedrock's web search serves only its GPT
models; any other search model leaves the agent running and each web search returning
that as its warning (`_web_search_spec`).
"""

import os
import re
import threading
from pathlib import Path
from typing import NamedTuple, get_args

import httpx
import yaml
from langchain_core.exceptions import ContextOverflowError

ANTHROPIC_BASE_URL = "https://gateway.smith.langchain.com/anthropic"
OPENAI_BASE_URL = "https://gateway.smith.langchain.com/v1"

# Per-socket deadlines on the root model's streaming call. Components, not a scalar:
# the read timeout is the gap *between* chunks, not the whole request, so `read` is a
# "no token for that long" watchdog rather than a ceiling on a turn. A long turn streams
# fine.
#
# That sentence is only true of a *streaming* request, which is why `root_model` also sets
# `streaming=True` — see its docstring. Do not attach this timeout to a model that might be
# invoked non-streaming: httpx then applies `read` to the whole response body and every turn
# longer than `read` dies after three attempts at ~3x it.
#
# This is the same pathology SUBAGENT_TIMEOUT_SECONDS below was written for, on the one
# role that never got the fix. An unset timeout is forwarded to the SDK client as a
# meaningful `None` — `httpx.Timeout(timeout=None)`, no connect, read, write or pool
# deadline at any layer — so a gateway that stops responding is waited on forever.
# Observed: the run entered the model node, opened one socket to the gateway, and sat
# there at 0% CPU with the socket in CLOSE_WAIT (the peer had already sent FIN). The
# `model` node never returned and the thread stayed `busy`, blocking every later turn.
#
# `read` is set against the measured inter-chunk gap on this path — worst observed ~1.5s,
# over replays of a 27-message payload — so 30s is ample headroom for a slow chunk while
# still bounding a socket that has gone silent. Raise it if a role needs more; do not go
# back to a scalar timeout.
#
# max_retries stays at its default of 2, which covers request establishment: a stall
# *before* the first token is retried transparently. A stall *after* the response object
# exists is not — the SDK does not retry mid-stream — which is why the failure above was
# a single silent wait rather than three attempts at it. That is a visible failure
# rather than a silent recovery, and still strictly better than a hang. Worst-case
# detection of a genuinely dead socket is now ~90s (3 attempts x 30s), up from ~30s, and
# still far inside CodeInterpreterMiddleware's 900s.
ROOT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

# Wall-clock ceiling on a single analyst request. Nothing else imposes one.
#
# An unset timeout is forwarded as a *meaningful* None (as of langchain-anthropic 1.5.x),
# so the client ends up as `httpx.Timeout(timeout=None)` — no connect, read, write or
# pool deadline at any layer. The Anthropic SDK's own 10-minute default never applies
# either: it computes a request timeout only when the client still carries that default,
# and ours is None. A socket that stalls is then waited on forever. Observed: one
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

# Wall-clock ceiling on one `web_search` call, which is one model call that does its own
# searching server-side before it answers. Deliberately 4x SUBAGENT_TIMEOUT_SECONDS: a
# leaf reads text already in its prompt, while this one issues real HTTP requests inside
# the provider and may issue several. Measured on a single-question digest through this
# gateway: 15.6s (Anthropic, 1 search), 12.0s (OpenAI, 3 searches + an open_page and a
# find_in_page). 120s is ~6x the worst of those, which is the same headroom ROOT_TIMEOUT
# takes, and it still sits well inside CodeInterpreterMiddleware's 900s — a `web_search`
# that outlives this has hung rather than searched hard.
SEARCH_TIMEOUT_SECONDS = 120.0

# The provider's own server-side web search, per gateway path. There is no portable form:
# each provider names its tool differently, and the search *runs inside the provider*
# rather than here, which is the whole reason this is a bound tool spec and not an HTTP
# client in `sources/`. Both were verified through this gateway on 2026-09-03.
#
# Do not bind either of these to the root model. They are server-side, so their results
# come back as content blocks in the assistant message and land in root context, outside
# `eval` and outside PTC — one such question measured 27.9k input tokens against a
# whole-run root budget the repo tunes in the low tens of thousands of *characters*.
# `sources/web.py` exists to spend those tokens in a throwaway call instead.
#
# A second trap, if anyone tries it anyway: passing a raw dict through
# `create_deep_agent(tools=[...])` crashes on the first model call with
# `AttributeError: 'dict' object has no attribute 'name'` — the PTC tool filter reads
# `.name` off every entry of `request.tools`, which LangChain types as
# `list[BaseTool | dict[str, Any]]`. Verified against langchain-quickjs 0.3.5.
WEB_SEARCH_SPECS = {
    # `max_uses` is a per-request cap on searches, and the only one either path offers.
    # Five is enough for a real question and bounds the cost of a runaway query.
    "anthropic": {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
    # Needs the Responses API, which `_build` already sets on this path — Chat
    # Completions has no server-side web search at all.
    "openai": {"type": "web_search"},
    # Bedrock's own search, for its GPT models on the Responses API (`_web_search_spec`).
    # Same tool type as OpenAI's. `external_web_access: False` keeps retrieval inside
    # Bedrock's index and cache: left at its default, every page fetch fails unless the
    # key's IAM identity also holds `bedrock-websearch:ExternalWebAccess`, which
    # AmazonBedrockFullAccess does not grant, and the answer degrades without saying so.
    "bedrock": {"type": "web_search", "external_web_access": False},
}

# The two gateway paths, which is all a provider selects here — see the module docstring.
# `anthropic` is the Anthropic Messages format, `openai` the OpenAI-compatible one; Bedrock
# is a vendor reached through either (see `_bedrock_parts`), not a path of its own.
PROVIDERS = ("anthropic", "openai")

# Bedrock models on its web search, by bare id prefix. Claude has none on Bedrock.
_BEDROCK_SEARCH_MODELS = ("gpt-5.4", "gpt-5.5", "gpt-5.6")

# Cross-region inference profiles prefix a Bedrock id with where it may run.
_BEDROCK_REGION_PREFIX = re.compile(r"^(?:us|eu|apac|jp|au|ca|global|us-gov)\.")

# Why models.yaml's defaults are what they are. Kept here rather than in the file a user
# edits, so that file stays short.
#
# 2026-09-16 sweep on the full 15-seed dataset, sol-low vs terra-high, judge and leaves
# held fixed: identical 10/15 rubric, same failure set (base-editing-t-cells-convergence,
# fmt-cdiff-placebo-trials, mrna-vaccines-lung-cancer-trials,
# psilocybin-depression-unpublished, semaglutide-weightloss-boxplot). terra-high found
# citations sol-low missed on 3 of 10 citation-checked seeds (egan-ulk1-ampk-sites,
# psilocybin-depression-unpublished, semaglutide-weightloss-boxplot), going 10/10 vs 7/10 —
# a citation-completeness win at equal rubric score, hence the default.
#
# Earlier baseline evaluations, before switching the root off Sonnet: terra and Sonnet 5
# hit 7/11 rubric and 8/9 citations, failing the same four rubric seeds as each other. A tie
# on quality makes it a cost decision, and every head-to-head so far puts terra far ahead
# per paper. `ROOT_MODEL=claude-sonnet-5` is the previous default's model,
# `claude-sonnet-4-6` the one before it, and both share these leaves, so either isolates
# the root — but watch root context when you do, because Sonnet 5 costs 1.9-2.6x Sonnet 4.6
# there (fmt-cdiff 86k -> 214k chars) for fan-outs 22-62% faster (198s -> 76s on
# semaglutide-weightloss-boxplot).
#
# The leaves are luna rather than Haiku 4.5 on cost, with quality held flat. Over the same
# dataset with the judge pinned, terra-low/luna-low scored the same *cell for cell*
# as terra-low/haiku-4.5 -- every seed, all three evaluators -- for ~40% less on 21% fewer
# tokens, at +5s median latency (31.0s -> 36.2s). Read that cost delta as a direction
# rather than a constant: it is one run of 11 examples with no repeats.
#
# What did *not* move across sol-low, terra-low and terra-high: the same core rubric seeds
# fail regardless of root model or effort. That points at a prompt, tool or criteria
# problem rather than a model-selection one.
#
# The older latency measurements above were taken against Haiku leaves.
# `SUBAGENT_MODEL=claude-haiku-4-5-20251001` restores them in one variable, but note it also
# has to drop the effort (`SUBAGENT_EFFORT=`) -- Haiku 4.5 has no effort scale and the
# gateway answers the parameter with a 400.
#
# The judge is pinned so that a score change is attributable to the pair under test rather
# than to the grader, and it is terra rather than luna because luna failed
# psilocybin-depression-unpublished on two claims that were both false about the answer in
# front of it. A grader that misreads the answer is a worse confound than a costlier one.

# The four roles, in the order `summary()` lists them. The first three run behind the chat
# UI; the judge only grades evals.
ROLES = ("root", "subagent", "search", "judge")
CHAT_ROLES = ROLES[:3]

_AXES = ("model", "provider", "effort")

# Every env var this module reads, so `cli.py` and `evals/run.py` can preserve them across
# their `load_dotenv(override=True)` without hand-maintaining a second copy of the list —
# a copy that drifts is how a `ROOT_MODEL=...` on the command line silently loses to .env.
# Built from names alone: both entry points import this before `.env` is loaded, so
# importing this module must not read models.yaml or import `paths` (see `refresh`).
ENV_VARS = tuple(f"{role.upper()}_{axis.upper()}" for role in ROLES for axis in _AXES)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated key rather than keeping the last one.

    A second `root:` block pasted under the first would otherwise replace it whole, and a
    repeated `effort:` would quietly win: the same silent wrong model that `_load` refuses
    unknown keys to prevent.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                repeated = key in seen
            except TypeError:  # unhashable; the base class reports that itself
                continue
            if repeated:
                raise yaml.constructor.ConstructorError(
                    "while reading a mapping", node.start_mark,
                    f"found `{key}` twice; keep one", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def _text(role: str, axis: str, value: object) -> str:
    """One scalar out of models.yaml, as the string the env var would have held.

    Empty or absent is `""`, which on the effort axis is a value (no effort), not an
    omission. A YAML boolean is refused rather than coerced: unquoted `off` or `no` parses
    as False, and silently reading that as "no effort" would hide a typo for `low`.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        empty = "" if axis == "model" else ", or leave it empty for none"
        raise SystemExit(
            f"models.yaml: {role}.{axis} is {value!r}; write it as text (quote it){empty}."
        )
    return value.strip().lower() if axis != "model" else value.strip()


def _load(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """(per-role defaults, display labels) from one models.yaml, refused legibly if malformed.

    Structure only. Whether a role's model, gateway path and effort work together is
    `_resolve`'s check, because env overrides take part in it. Unknown and repeated keys
    are errors rather than ignored, since a misspelt `subagents:` or a pasted second
    `root:` would otherwise run the wrong model silently.
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except FileNotFoundError:
        raise SystemExit(f"{path} is missing; it names the model each role runs.") from None
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise SystemExit(f"{path.name} must be a mapping of roles to models.")
    if unknown := set(raw) - {*ROLES, "labels"}:
        raise SystemExit(
            f"{path.name}: unknown key(s) {', '.join(sorted(map(str, unknown)))}. "
            f"Expected {', '.join(ROLES)} and labels."
        )

    defaults = {}
    for role in ROLES:
        entry = raw.get(role)
        if not isinstance(entry, dict) or not _text(role, "model", entry.get("model")):
            raise SystemExit(
                f"{path.name}: `{role}` needs a `model`, e.g. `model: openai/gpt-5.6-luna`."
            )
        if unknown := set(entry) - set(_AXES):
            raise SystemExit(
                f"{path.name}: {role} has unknown key(s) "
                f"{', '.join(sorted(map(str, unknown)))}. Expected {', '.join(_AXES)}."
            )
        defaults[role] = {axis: _text(role, axis, entry.get(axis)) for axis in _AXES}

    labels = raw.get("labels") or {}
    if not isinstance(labels, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in labels.items()
    ):
        raise SystemExit(f"{path.name}: `labels` must map model ids to display names.")
    return defaults, labels


class _Config(NamedTuple):
    stamp: tuple[int, int] | None  # (mtime_ns, size) of the file this was read from
    defaults: dict[str, dict[str, str]]
    labels: dict[str, str]


_config_state: _Config | None = None
_config_lock = threading.Lock()


def refresh() -> None:
    """Re-read models.yaml if it changed since the last read. Blocking file I/O.

    The server calls this before every graph build (`graph.make_graph`) and every
    `GET /models`, through `asyncio.to_thread`, so an edit applies to the next run and to
    the badge without a restart; the graph is rebuilt per run anyway. Env overrides need a
    restart only because the process environment does. Everything else reads the file on
    first use and keeps it (`_config`), so a CLI run or an eval sweep sees one
    configuration throughout.

    `paths` is imported here rather than at module level because it reads
    DEEP_LIFE_SCI_DATA_DIR at import, and `cli.py` imports this module before `.env`.
    """
    global _config_state
    from deep_life_sci import paths

    path = paths.MODELS_FILE
    with _config_lock:
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            stamp = None
        if _config_state is not None and stamp is not None and stamp == _config_state.stamp:
            return
        defaults, labels = _load(path)
        _config_state = _Config(stamp, defaults, labels)


def _config() -> _Config:
    if _config_state is None:
        refresh()
    return _config_state


def __getattr__(name: str):
    """`DEFAULTS` and `LABELS`, read from models.yaml on first use rather than at import."""
    if name == "DEFAULTS":
        return _config().defaults
    if name == "LABELS":
        return _config().labels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _bedrock_parts(model: str) -> tuple[str, str] | None:
    """(model maker, bare id) for a `bedrock/` id, or None for any other id.

    Bedrock ids name their maker and carry a version, and cross-region inference profiles
    add a region: `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` is
    ("anthropic", "claude-sonnet-4-5-20250929"), `bedrock/openai.gpt-5.6-terra` is
    ("openai", "gpt-5.6-terra"). The bare id is what the maker's langchain profile is
    keyed by, and the maker decides the gateway path.
    """
    vendor, _, rest = model.partition("/")
    if vendor != "bedrock" or not rest:
        return None
    maker, _, name = _BEDROCK_REGION_PREFIX.sub("", rest).partition(".")
    return (maker, re.sub(r"-v\d+(?::\d+)?$", "", name)) if name else ("", maker)


def _is_bedrock_claude(model: str) -> bool:
    return (_bedrock_parts(model) or ("",))[0] == "anthropic"


def _infer_provider(model: str) -> str:
    """Which gateway path a model id looks like, or "" when it says nothing.

    The two paths take different id forms (see the module docstring), so the id itself
    usually says which one it is: bare ids like `claude-sonnet-4-6` are the
    Anthropic-native path, `provider/model` ids like `openai/gpt-5.6-terra` are the
    OpenAI-compatible one.

    Claude on Bedrock (`bedrock/...anthropic.claude-...`) is the exception: it takes the
    Anthropic Messages format through the gateway's standard endpoint, which keeps what
    only `ChatAnthropic` does — prompt caching through `cache_control`, and effort mapped
    to `output_config` — where the OpenAI-compatible path would drop the first.
    """
    if "/" in model:
        return "anthropic" if _is_bedrock_claude(model) else "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return ""


def _provider_for(
    role: str,
    model: str,
    declared: str,
    *,
    model_source: str = "",
    provider_source: str = "",
) -> str:
    """Validate one role's gateway path against the form of its model id.

    A named path wins where the form says nothing, which is the escape hatch: a model id in
    neither known form is usable by naming its path rather than by editing this module.
    Where the form *does* say something and the two disagree, that is an error rather than
    a preference — sending an id down the wrong path returns a 501 that reads like an
    outage, or silently drops prompt caching.

    The sources name where each value came from (`models.yaml root.model`, or the env var),
    so the error points at the line to fix; they default to the env var names.
    """
    model_source = model_source or f"{role.upper()}_MODEL"
    provider_source = provider_source or f"{role.upper()}_PROVIDER"
    inferred = _infer_provider(model)
    if declared:
        if declared not in PROVIDERS:
            raise SystemExit(
                f"{provider_source}={declared!r} is not a gateway path. "
                f"Choose one of: {', '.join(PROVIDERS)}"
            )
        # Bedrock Claude also works on the OpenAI-compatible path, only without caching,
        # so naming that path is a choice rather than a contradiction.
        if inferred and inferred != declared and not _is_bedrock_claude(model):
            raise SystemExit(
                f"{model_source}={model!r} is a {inferred!r} id but {provider_source} "
                f"says {declared!r}. The paths take different id forms: anthropic wants a "
                "bare id ('claude-sonnet-5'), openai a prefixed one ('openai/gpt-5.6-terra'). "
                "Fix one or the other, or unset the provider and let the form decide."
            )
        return declared
    if inferred:
        return inferred
    raise SystemExit(
        f"Cannot tell which gateway path {model_source}={model!r} needs. "
        "Anthropic-native ids are bare ('claude-sonnet-5'); everything else must carry a "
        f"provider prefix ('openai/gpt-5.6-terra'). Or set {provider_source} explicitly to "
        f"one of: {', '.join(PROVIDERS)}"
    )


def _setting(role: str, axis: str) -> str:
    """One config value for one role: `{ROLE}_{AXIS}` in the environment, else models.yaml.

    An env var set to whitespace reads as unset rather than as an empty model id. That
    is why `_effort` does **not** go through here: on that axis an explicit empty value
    is a value, not an omission. Only `model` is left, so this reads narrower than it
    looks — do not fold the effort axis back into it.
    """
    value = os.environ.get(f"{role.upper()}_{axis.upper()}", "").strip()
    return value or _config().defaults[role][axis]


def _effort(role: str) -> str:
    """`{ROLE}_EFFORT`, else the role's models.yaml effort. `_check_effort` validates it.

    On Anthropic ids this maps to `output_config.effort` and, because langchain-anthropic
    defaults `thinking` to adaptive whenever effort is set, setting it also turns thinking
    *on* — unset is not "effort=high", it is no thinking at all. That difference is the
    whole point of the axis, but it means summarized thinking lands in the transcript, so
    expect root_context_chars to move with ROOT_EFFORT.

    **Read directly rather than through `_setting`, and that is the entire point of this
    line.** `_setting` treats empty as unset and falls back to the default, which is right
    for a model id — an empty one is unusable — and wrong here, because empty is a
    *value* on this axis: `SUBAGENT_EFFORT=` is the documented way to run a model that has
    no effort scale at all, and Haiku 4.5 answers the parameter with a 400. Through
    `_setting`, `SUBAGENT_MODEL=claude-haiku-4-5-20251001 SUBAGENT_EFFORT=` resolved to
    `low` and 400'd, with nothing anywhere saying the clear had been ignored.
    """
    raw = os.environ.get(f"{role.upper()}_EFFORT")
    return (_config().defaults[role]["effort"] if raw is None else raw).strip().lower()


# Every effort level any model here takes. A value outside it is a typo whatever the model.
_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _profile_levels(model: str, provider: str) -> tuple[str, ...] | None:
    """The effort levels langchain's model profile lists for this id, or None if unknown.

    Profiles are keyed by the vendor's bare id, so `openai/gpt-5.6-terra` is looked up as
    `gpt-5.6-terra` in langchain-openai's table and `claude-*` in langchain-anthropic's.
    Any other vendor has no table. A profile that lists no levels also counts as unknown
    rather than as "takes no effort": o3's lists none, and o3 takes low through high.
    """
    vendor, _, bare = model.rpartition("/")
    if not vendor and provider == "anthropic":
        vendor = "anthropic"
    if bedrock := _bedrock_parts(model):
        vendor, bare = bedrock
    try:
        if vendor == "openai":
            from langchain_openai.chat_models.base import _get_default_model_profile
        elif vendor == "anthropic":
            from langchain_anthropic.chat_models import _get_default_model_profile
        else:
            return None
        levels = (_get_default_model_profile(bare) or {}).get("reasoning_effort_levels")
    except Exception:  # noqa: BLE001 - a private helper; if it moves, there is no profile
        return None
    return tuple(levels) if levels else None


def _anthropic_efforts() -> tuple[str, ...]:
    """The levels `ChatAnthropic` accepts at all, read off its `reasoning_effort` Literal.

    Pydantic checks that Literal when the model is built, so anything else fails there
    with a bare ValidationError naming neither the role nor models.yaml.
    """
    from langchain_anthropic import ChatAnthropic

    annotation = ChatAnthropic.model_fields["reasoning_effort"].annotation
    return tuple(v for arg in get_args(annotation) for v in get_args(arg)) or _EFFORT_LEVELS


def _check_effort(model: str, provider: str, effort: str, source: str) -> None:
    """Refuse an effort this model cannot take, before anything is built or booted.

    Most specific first: the levels the id's langchain profile lists (Sonnet 4.6 has no
    `xhigh`; GPT-5.6 takes `none`); on the Anthropic-native path, what `ChatAnthropic`
    accepts at all; otherwise every known level, which still catches a typo before an
    eval sweep boots a container per example. What none of these knows — a model with no
    profile that takes no effort, like Haiku 4.5 — is left to the provider, and its 400
    reaches the user through `middleware/model_errors.py`.
    """
    if not effort:
        return
    levels, scope = _profile_levels(model, provider), f"for {model!r}"
    if levels is None and provider == "anthropic":
        levels, scope = _anthropic_efforts(), "on the Anthropic path"
    if levels is None:
        levels, scope = _EFFORT_LEVELS, "for any model"
    if effort not in levels:
        raise SystemExit(
            f"{source}={effort!r} is not an effort level {scope}. Choose one of: "
            f"{', '.join(levels)}; or leave it empty to omit it."
        )


def rejection_message(role: str, exc: BaseException) -> str | None:
    """A provider's HTTP error on one role's model call, as text the user will see.

    None for anything without a status code, so our own bugs stay as they are, and for a
    context overflow, which deepagents' summarization middleware catches as
    `ContextOverflowError` to compact the history and retry; a rewrapped one would end the
    run instead. Everything else is reported as the provider worded it, with the role's
    settings beside it and no guess at the cause: a 400 is as often a content filter or an
    unreadable image as a setting. Both SDKs' `APIStatusError` carry `status_code`, so no
    provider import is needed. Reads the settings without `_resolve`'s checks, so this
    cannot raise in place of the error it reports.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or isinstance(exc, ContextOverflowError):
        return None
    return (
        f"The LLM Gateway returned an error for the {role} model's request ({status}): "
        f"{exc}\n\n({role} role: model {_setting(role, 'model')!r}, effort "
        f"{_effort(role) or '(none)'!r}, from models.yaml or {role.upper()}_* env vars.)"
    )


def _resolve(role: str) -> tuple[str, str, str]:
    """(model id, gateway path, effort) for one role, from the env or models.yaml.

    Refuses any combination that cannot work, naming where each value came from.
    """
    upper, defaults = role.upper(), _config().defaults[role]
    model = _setting(role, "model")
    from_env = bool(os.environ.get(f"{upper}_MODEL", "").strip())
    model_source = f"{upper}_MODEL" if from_env else f"models.yaml {role}.model"
    provider = os.environ.get(f"{upper}_PROVIDER", "").strip().lower()
    provider_source = f"{upper}_PROVIDER"
    if not provider:
        # A default provider describes the default model it sits beside, so it does not
        # survive that model being replaced: `ROOT_MODEL=claude-sonnet-5` alone would
        # otherwise contradict a models.yaml `provider: openai` and refuse to run.
        provider = defaults["provider"] if model == defaults["model"] else _infer_provider(model)
        if not from_env:
            provider_source = f"models.yaml {role}.provider"
    provider = _provider_for(
        role, model, provider, model_source=model_source, provider_source=provider_source
    )
    effort = _effort(role)
    effort_source = (
        f"{upper}_EFFORT" if f"{upper}_EFFORT" in os.environ else f"models.yaml {role}.effort"
    )
    _check_effort(model, provider, effort, effort_source)
    return model, provider, effort


class WebSearchUnavailable(ValueError):
    """The search role's model has no server-side web search; the message says what does.

    Raised per search rather than at startup: a search model that cannot search takes out
    web search only, and each `web_search` call returns this as its warning for the root
    to relay, while the rest of the agent runs.
    """


def _web_search_spec(model: str, provider: str, source: str = "SEARCH_MODEL") -> dict:
    """The server-side web search tool for the search role's model, or why it has none.

    The spec follows the vendor, not only the path: Bedrock serves GPT models on the
    OpenAI-compatible path with a search of its own, and Claude on Bedrock has no
    server-side search at all, so the search role cannot run on it.
    """
    bedrock = _bedrock_parts(model)
    if bedrock is None:
        return WEB_SEARCH_SPECS[provider]
    maker, bare = bedrock
    if maker == "openai" and bare.startswith(_BEDROCK_SEARCH_MODELS):
        return WEB_SEARCH_SPECS["bedrock"]
    raise WebSearchUnavailable(
        f"{source}={model!r} cannot be the search model: Bedrock's web search runs only on "
        f"its OpenAI GPT models ({', '.join(m + '*' for m in _BEDROCK_SEARCH_MODELS)}, e.g. "
        "'bedrock/openai.gpt-5.6-luna', in us-east-1, us-east-2 or us-west-2), and Claude "
        "has no server-side search on Bedrock. Use one of those, or a model on OpenAI or "
        "Anthropic directly."
    )


def web_search_problem() -> str | None:
    """Why the search role cannot search, or None. For a warning at startup, not a refusal."""
    model, provider, _ = _resolve("search")
    try:
        _web_search_spec(model, provider, _source_of("search", "model"))
    except WebSearchUnavailable as exc:
        return str(exc)
    return None


def _source_of(role: str, axis: str) -> str:
    name = f"{role.upper()}_{axis.upper()}"
    return name if os.environ.get(name, "").strip() else f"models.yaml {role}.{axis}"


def validate(*roles: str) -> None:
    """Resolve each role now (all four by default), so a bad setting fails before a boot."""
    for role in roles or ROLES:
        _resolve(role)


def _messages_base_url(model: str) -> str:
    """Where an Anthropic Messages request for this id goes.

    A bare Claude id takes the direct `/anthropic` path. A prefixed one (Claude on Bedrock)
    takes the gateway's standard endpoint, the same host as the OpenAI-compatible `/v1`
    without it, which accepts `<provider>/<model>` in the Messages format too. The SDK
    appends `/v1/messages` either way.
    """
    if "/" not in model:
        return os.environ.get("LANGSMITH_GATEWAY_ANTHROPIC_URL", ANTHROPIC_BASE_URL)
    base = os.environ.get("LANGSMITH_GATEWAY_BASE_URL", OPENAI_BASE_URL).rstrip("/")
    return base.removesuffix("/v1")


def _build(model: str, provider: str, **kwargs):
    check_gateway_config()
    key = gateway_key()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic types `timeout` as a float and refuses an httpx.Timeout, which is
        # what ROOT_TIMEOUT is, so an Anthropic root could not be built at all. Its `read`
        # is the part that matters (the gap between streamed chunks); a float applies that
        # one value to every phase, so connect and pool wait 30s here rather than 5s.
        # ChatOpenAI takes the components as they are, so only this path collapses them.
        if isinstance(timeout := kwargs.get("timeout"), httpx.Timeout):
            kwargs["timeout"] = timeout.read
        return ChatAnthropic(
            model=model,
            base_url=_messages_base_url(model),
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
    `read` is a gap-between-chunks watchdog, which is only what it means on a streaming
    request; on a non-streaming one httpx applies it to the whole response body, turning an
    inter-chunk allowance into a ceiling on an entire turn. The callers disagreed
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


def web_search_model(**kwargs):
    """The `search` role, with the provider's server-side web search already bound.

    Returns a Runnable rather than a bare chat model, because which spec to bind is
    decided by the resolved gateway path (see `WEB_SEARCH_SPECS`) and that resolution
    lives here. `sources/web.py` therefore never has to know which provider it is on.

    Binding in this module is also what keeps a provider swap from silently sending the
    wrong spec: `SEARCH_PROVIDER` is an env axis like every other, so a hard-coded spec
    at the call site would answer `SEARCH_MODEL=claude-sonnet-5` with an Anthropic model
    holding an OpenAI tool definition, which the gateway rejects with a 400.

    Timed out at SEARCH_TIMEOUT_SECONDS rather than the leaves' 30s; see that constant.
    """
    model, provider, effort = _resolve("search")
    kwargs.setdefault("timeout", SEARCH_TIMEOUT_SECONDS)
    if effort:
        kwargs.setdefault("reasoning_effort", effort)
    spec = _web_search_spec(model, provider, _source_of("search", "model"))
    return _build(model, provider, **kwargs).bind_tools([spec])


def judge_model(**kwargs):
    """The eval judge's model — configured independently of the pair under test.

    See the defaults' rationale above for why it is pinned. JUDGE_MODEL in the environment
    overrides it, which is how you check whether a verdict is the answer's fault or the
    grader's.
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


def summary(*roles: str) -> list[dict[str, str]]:
    """Each role's resolved model, for the chat UI's model badge (`webapp.py`).

    Resolved rather than read from models.yaml, so the UI shows what this process actually
    runs, environment overrides included. `label` falls back to the model id.
    """
    rows = []
    for role in roles or ROLES:
        model, provider, effort = _resolve(role)
        rows.append({
            "role": role,
            "model": model,
            "label": _config().labels.get(model) or model,
            "provider": provider,
            "effort": effort,
        })
    return rows


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
