"""Who may use a deployed agent, and which threads each of them may see.

Loaded only by a deployment (`scripts/deploy.py` adds it to the deploy config); a local
server has no auth at all. Two kinds of caller reach a deployment:

* **People signed in with the institution's identity provider**, carrying a bearer token.
  Any OpenID Connect provider works — Entra ID, Okta, Google, Auth0, Keycloak, Cognito —
  configured by three settings rather than code:

  - `OIDC_ISSUER`: the provider's issuer URL, exactly as it appears in the tokens' `iss`.
  - `OIDC_CLIENT_ID`: the app registered with the provider. The chat UI signs in as it
    (webapp.py hands it the settings), and it is the audience of the ID tokens it sends.
  - `OIDC_AUDIENCE` (optional): comma-separated audiences to accept instead, for a UI set
    to send access tokens (`OIDC_TOKEN=access`) issued for an API, as Okta's custom
    authorization servers and Entra app registrations do. One of the two is required:
    without an audience, a token the provider issued for *any* app would be accepted.
  - `OIDC_ALLOWED_EMAIL_DOMAINS` (optional): comma-separated; a token must then carry a
    verified email in one of them.

  The token is verified against the provider's published signing keys (found through its
  discovery document), and its `sub` becomes the caller's identity. A token that is not a
  signed JWT — Google's access tokens are opaque — cannot be verified here; send the ID
  token instead.

* **Holders of a LangSmith API key for the workspace**: Studio, scripts, `dev.py --remote`.
  `allow_langsmith_api_keys` in the deploy config routes every request *without* an
  `Authorization` header to LangSmith's own auth, so this module never sees their key.

`DEEP_LIFE_SCI_AUTH` picks between them, and `deploy.py --auth` always sets it. `oidc` admits
both kinds of caller; anything else (or unset) is LangSmith-only: every bearer token is
refused, and only the second kind gets in. An explicit setting rather than "is OIDC_ISSUER
set", so a deployment switched back to `langsmith` does not stay open to a leftover issuer.

Signed-in users see and run only their own threads (an `owner` stamped in the thread's
metadata). What outlives a thread outside the server — its sandbox, its stored uploads — is
keyed by owner as well as thread id (ownership.py), since a deleted thread's id can be
created again by someone else. LangSmith key holders and Studio are the workspace's own
people and see everything. Anything not explicitly allowed is refused to signed-in users.
The store API is switched off in the deploy config rather than guarded here, which leaves
the graph's own store calls untouched.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import httpx
import jwt
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

# What `authenticate` grants a signed-in user, and what the resource handlers look for. A
# LangSmith-key caller arrives with `["authenticated"]` instead, and Studio as a StudioUser.
# Defined beside the code that scopes a thread's sandbox and uploads to its owner.
from deep_life_sci.ownership import USER

logger = logging.getLogger(__name__)

auth = Auth()

# Asymmetric only. An `HS*` token would be verified with a public key as its HMAC secret,
# and `none` with nothing at all; neither is something a provider's published keys can check.
ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384",
              "ES512", "EdDSA")

# The curve each ECDSA algorithm signs with. A token's header names its algorithm, so it is
# checked against the key before PyJWT sees the pair: handed an RSA key for an `ES256`
# header, PyJWT raises a TypeError rather than an InvalidTokenError.
_CURVES = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}

# Clock skew tolerated on `exp`, `nbf` and `iat`.
LEEWAY_SECONDS = 60

# Signing keys are cached per process. Past KEYS_TTL_SECONDS the next request refreshes them,
# and a token naming a key the cache lacks (a rotation, or a forgery) refreshes them sooner.
# Either way there is at most one attempt per REFETCH_SECONDS, failed or not, so neither an
# outage at the provider nor a stream of forged key ids becomes a stream of requests to it.
# When a refresh fails, the keys already held go on verifying for up to MAX_STALE_SECONDS:
# a provider that is down for a while should not sign out everyone already signed in.
KEYS_TTL_SECONDS = 3600
REFETCH_SECONDS = 30
MAX_STALE_SECONDS = 24 * 3600
FETCH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _unauthorized(detail: str) -> Auth.exceptions.HTTPException:
    return Auth.exceptions.HTTPException(status_code=401, detail=detail)


class _IssuerUnavailable(Exception):
    """The issuer's discovery document or key set could not be fetched or read."""


class _Key(NamedTuple):
    jwk: jwt.PyJWK
    # The JWK as published, for the `alg`, `kty` and `crv` a token's algorithm must fit.
    data: dict[str, Any]


def _signing_keys(published: Any) -> dict[str | None, _Key]:
    """The usable signing keys in a JWKS, by key id.

    Encryption keys, and keys PyJWT cannot load (a curve it does not support, a malformed
    entry), are skipped: one odd key in a provider's set must not refuse every token.
    """
    entries = published.get("keys") if isinstance(published, dict) else None
    keys: dict[str | None, _Key] = {}
    for data in entries if isinstance(entries, list) else ():
        if not isinstance(data, dict) or data.get("use", "sig") != "sig":
            continue
        kid = data.get("kid")
        if kid is not None and not isinstance(kid, str):
            continue
        try:
            keys[kid] = _Key(jwt.PyJWK(data), data)
        except (jwt.PyJWTError, ValueError, TypeError, KeyError):
            continue
    return keys


def _fits(algorithm: str, key: _Key) -> bool:
    """Whether a token claiming `algorithm` can be checked with this key at all.

    A key that names its algorithm accepts only that one; otherwise the algorithm's family
    must match the key's type, and for ECDSA its curve.
    """
    declared = key.data.get("alg")
    if declared:
        return declared == algorithm
    kty, crv = key.data.get("kty"), key.data.get("crv")
    if algorithm.startswith(("RS", "PS")):
        return kty == "RSA"
    if algorithm in _CURVES:
        return kty == "EC" and crv == _CURVES[algorithm]
    return algorithm == "EdDSA" and kty == "OKP" and crv in ("Ed25519", "Ed448")


class _SigningKeys:
    """The issuer's JWKS, found through its discovery document and cached.

    A request whose key is cached never waits on the provider while another request is
    refreshing, and goes on verifying with the cached keys if the refresh fails. A request
    whose key is not cached waits for a refresh, shared with any others waiting.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._issuer: str | None = None
        self._keys: dict[str | None, _Key] = {}
        self._jwks_uri: str | None = None
        self._fetched = float("-inf")  # the last refresh that succeeded
        self._tried = float("-inf")  # the last attempt, whatever its outcome
        self._tried_issuer: str | None = None
        self._error: str | None = None
        self._lock = asyncio.Lock()

    def _cached(self, issuer: str, kid: str | None) -> _Key | None:
        if issuer != self._issuer or self._clock() - self._fetched > MAX_STALE_SECONDS:
            return None
        if kid is None and len(self._keys) == 1:
            return next(iter(self._keys.values()))
        return self._keys.get(kid)

    def _fresh(self, issuer: str) -> bool:
        return issuer == self._issuer and self._clock() - self._fetched < KEYS_TTL_SECONDS

    async def get(self, issuer: str, kid: str | None) -> _Key:
        held = self._cached(issuer, kid)
        if held is not None and (self._fresh(issuer) or self._lock.locked()):
            return held
        async with self._lock:
            due = issuer != self._tried_issuer or self._clock() - self._tried >= REFETCH_SECONDS
            if due:
                try:
                    await self._refresh(issuer)
                except _IssuerUnavailable as exc:
                    if held is None:
                        raise
                    logger.warning("using cached signing keys; refreshing them failed: %s", exc)
        # A key set refreshed just now decides alone, so a key rotated out of it stops
        # verifying; otherwise the key held before is still the provider's own.
        key = self._cached(issuer, kid) if self._fresh(issuer) else held
        if key is not None:
            return key
        if self._error and not self._fresh(issuer):
            raise _IssuerUnavailable(self._error)
        raise _unauthorized("token signed with a key the issuer does not publish")

    async def _refresh(self, issuer: str) -> None:
        self._tried, self._tried_issuer = self._clock(), issuer
        try:
            keys, jwks_uri = await self._fetch(issuer)
        except _IssuerUnavailable as exc:
            self._error = str(exc)
            # Found again from the discovery document next time, in case it moved.
            self._jwks_uri = None
            raise
        self._issuer, self._keys, self._jwks_uri = issuer, keys, jwks_uri
        self._fetched, self._error = self._clock(), None

    async def _fetch(self, issuer: str) -> tuple[dict[str | None, _Key], str]:
        try:
            async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, transport=self._transport) as http:
                jwks_uri = self._jwks_uri if issuer == self._issuer else None
                if jwks_uri is None:
                    discovery = await http.get(
                        f"{issuer.rstrip('/')}/.well-known/openid-configuration"
                    )
                    discovery.raise_for_status()
                    document = discovery.json()
                    jwks_uri = document.get("jwks_uri") if isinstance(document, dict) else None
                    if not isinstance(jwks_uri, str) or not jwks_uri:
                        raise _IssuerUnavailable("its discovery document names no jwks_uri")
                response = await http.get(jwks_uri)
                response.raise_for_status()
                published = response.json()
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise _IssuerUnavailable(str(exc) or type(exc).__name__) from None
        except ValueError as exc:  # a body that is not JSON
            raise _IssuerUnavailable(f"it answered with something not JSON ({exc})") from None
        keys = _signing_keys(published)
        if not keys:
            raise _IssuerUnavailable("it publishes no signing key this server can use")
        return keys, jwks_uri


_keys = _SigningKeys()


async def verify(token: str) -> dict[str, Any]:
    """The token's claims, or an HTTP 401/403 saying what was wrong with it."""
    issuer = os.environ.get("OIDC_ISSUER", "").strip()
    audiences = _list("OIDC_AUDIENCE") or _list("OIDC_CLIENT_ID")
    if os.environ.get("DEEP_LIFE_SCI_AUTH", "").strip() != "oidc":
        raise _unauthorized("this deployment accepts LangSmith API keys only")
    if not issuer:
        raise _unauthorized("sign-in is not configured: OIDC_ISSUER is unset")
    if not audiences:
        # A deploy-time mistake, refused rather than half-honoured: see the module docstring.
        raise _unauthorized("sign-in is not configured: set OIDC_CLIENT_ID or OIDC_AUDIENCE")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise _unauthorized("not a signed JWT") from None
    algorithm = header.get("alg")
    if algorithm not in ALGORITHMS:
        raise _unauthorized(f"unsupported signing algorithm {algorithm!r}")
    try:
        # PyJWT has already refused a `kid` that is not a string.
        key = await _keys.get(issuer, header.get("kid"))
    except _IssuerUnavailable as exc:
        # The provider is down or the issuer is wrong: the caller cannot fix it, but a 401
        # is still the honest answer to "is this token good".
        raise _unauthorized(f"could not fetch the issuer's signing keys: {exc}") from None
    if not _fits(algorithm, key):
        raise _unauthorized(f"a {algorithm} token cannot be signed with the key it names")

    try:
        claims = jwt.decode(
            token,
            key=key.jwk.key,
            algorithms=[algorithm],
            audience=audiences,
            issuer=issuer,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "iss", "sub", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized(f"invalid token: {exc}") from None

    domains = {d.lower() for d in _list("OIDC_ALLOWED_EMAIL_DOMAINS")}
    if domains:
        email = str(claims.get("email", "")).lower()
        # Only an explicit `false` counts as unverified: several providers (Entra among them)
        # omit the claim for addresses they manage themselves.
        verified = claims.get("email_verified", True) not in (False, "false")
        if not (verified and email.rpartition("@")[2] in domains):
            raise Auth.exceptions.HTTPException(
                status_code=403, detail="this account's email domain is not allowed"
            )
    return claims


@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("expected an `Authorization: Bearer <token>` header")
    claims = await verify(token.strip())
    return {
        "identity": str(claims["sub"]),
        "display_name": str(claims.get("name") or claims.get("email") or claims["sub"]),
        "permissions": [USER],
    }


def _is_signed_in_user(ctx: Auth.types.AuthContext) -> bool:
    return not is_studio_user(ctx.user) and USER in ctx.permissions


def _owned(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, str]:
    """Stamp the caller as owner on what it writes, and filter what it reads to that."""
    owner = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.update(owner)
    return owner


@auth.on
async def deny_by_default(ctx: Auth.types.AuthContext, value: Any) -> None:
    if _is_signed_in_user(ctx):
        raise Auth.exceptions.HTTPException(status_code=403, detail="not permitted")


@auth.on.threads
async def own_threads(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict | None:
    return _owned(ctx, value) if _is_signed_in_user(ctx) else None


# Crons are left to `deny_by_default`: the chat UI never schedules one, and a scheduled run
# fires without the user's token being checked again, so a signed-in user's cron would go
# on running agents after their sign-in has expired or their account has been removed.


# The chat UI reads the graph's assistant to run it; changing assistants is the workspace's
# business, so signed-in users fall through to `deny_by_default` for the rest.
@auth.on.assistants.read
async def read_assistants(ctx: Auth.types.AuthContext, value: Any) -> None:
    return None


@auth.on.assistants.search
async def search_assistants(ctx: Auth.types.AuthContext, value: Any) -> None:
    return None
