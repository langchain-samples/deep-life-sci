"""Who may use a deployed agent, and which threads each of them may see.

Loaded only by a deployment (`scripts/deploy.py` adds it to the deploy config); a local
server has no auth at all. Two kinds of caller reach a deployment:

* **People signed in with the institution's identity provider**, carrying a bearer token.
  Any OpenID Connect provider works — Entra ID, Okta, Google, Auth0, Keycloak, Cognito —
  configured by three settings rather than code:

  - `OIDC_ISSUER`: the provider's issuer URL, exactly as it appears in the tokens' `iss`.
  - `OIDC_AUDIENCE`: comma-separated audiences a token may be for. An ID token's audience
    is the app's client ID; an access token's is whatever API it was issued for (Okta's
    custom authorization servers, Entra app registrations). Required: without it, any
    token the provider issued for *any* app would be accepted.
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
metadata, which sandboxes inherit because they are keyed by thread). LangSmith key holders
and Studio are the workspace's own people and see everything. Anything not explicitly
allowed is refused to signed-in users. The store API is switched off in the deploy config
rather than guarded here, which leaves the graph's own store calls untouched.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import jwt
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

auth = Auth()

# What `authenticate` grants a signed-in user, and what the resource handlers look for. A
# LangSmith-key caller arrives with `["authenticated"]` instead, and Studio as a StudioUser.
USER = "oidc-user"

# Asymmetric only. An `HS*` token would be verified with a public key as its HMAC secret,
# and `none` with nothing at all; neither is something a provider's published keys can check.
ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384",
              "ES512", "EdDSA")

# Clock skew tolerated on `exp`, `nbf` and `iat`.
LEEWAY_SECONDS = 60

# Signing keys are cached per process and fetched again after this long, or sooner when a
# token names a key we do not have (a rotation), but at most once per REFETCH_SECONDS so a
# stream of forged key ids cannot turn into a stream of requests to the provider.
KEYS_TTL_SECONDS = 3600
REFETCH_SECONDS = 30


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _unauthorized(detail: str) -> Auth.exceptions.HTTPException:
    return Auth.exceptions.HTTPException(status_code=401, detail=detail)


class _SigningKeys:
    """The issuer's JWKS, found through its discovery document and cached."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._issuer: str | None = None
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched = 0.0
        self._lock = asyncio.Lock()

    async def get(self, issuer: str, kid: str | None) -> jwt.PyJWK:
        async with self._lock:
            age = time.monotonic() - self._fetched
            stale = issuer != self._issuer or age > KEYS_TTL_SECONDS
            missing = kid not in self._keys
            if stale or (missing and age > REFETCH_SECONDS):
                await self._refresh(issuer)
        if kid is None and len(self._keys) == 1:
            return next(iter(self._keys.values()))
        if kid not in self._keys:
            raise _unauthorized("token signed with a key the issuer does not publish")
        return self._keys[kid]

    async def _refresh(self, issuer: str) -> None:
        async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as http:
            discovery = await http.get(f"{issuer.rstrip('/')}/.well-known/openid-configuration")
            discovery.raise_for_status()
            jwks = await http.get(discovery.json()["jwks_uri"])
            jwks.raise_for_status()
        keys = {}
        for data in jwks.json().get("keys", []):
            # Encryption keys and algorithms PyJWT cannot load are skipped, not fatal.
            if data.get("use", "sig") != "sig":
                continue
            try:
                key = jwt.PyJWK(data)
            except jwt.PyJWKError:
                continue
            keys[data.get("kid")] = key
        self._issuer, self._keys, self._fetched = issuer, keys, time.monotonic()


_keys = _SigningKeys()


async def verify(token: str) -> dict[str, Any]:
    """The token's claims, or an HTTP 401/403 saying what was wrong with it."""
    issuer = os.environ.get("OIDC_ISSUER", "").strip()
    audiences = _list("OIDC_AUDIENCE")
    if os.environ.get("DEEP_LIFE_SCI_AUTH", "").strip() != "oidc":
        raise _unauthorized("this deployment accepts LangSmith API keys only")
    if not issuer:
        raise _unauthorized("sign-in is not configured: OIDC_ISSUER is unset")
    if not audiences:
        # A deploy-time mistake, refused rather than half-honoured: see the module docstring.
        raise _unauthorized("sign-in is not configured: OIDC_AUDIENCE is unset")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise _unauthorized("not a signed JWT") from None
    algorithm = header.get("alg")
    if algorithm not in ALGORITHMS:
        raise _unauthorized(f"unsupported signing algorithm {algorithm!r}")

    try:
        key = await _keys.get(issuer, header.get("kid"))
    except httpx.HTTPError as exc:
        # The provider is down or the issuer is wrong: the caller cannot fix it, but a 401
        # is still the honest answer to "is this token good".
        raise _unauthorized(f"could not fetch the issuer's signing keys ({exc})") from None

    try:
        claims = jwt.decode(
            token,
            key=key.key,
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


@auth.on.crons
async def own_crons(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict | None:
    return _owned(ctx, value) if _is_signed_in_user(ctx) else None


# The chat UI reads the graph's assistant to run it; changing assistants is the workspace's
# business, so signed-in users fall through to `deny_by_default` for the rest.
@auth.on.assistants.read
async def read_assistants(ctx: Auth.types.AuthContext, value: Any) -> None:
    return None


@auth.on.assistants.search
async def search_assistants(ctx: Auth.types.AuthContext, value: Any) -> None:
    return None
