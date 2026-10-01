"""The deployment's sign-in and access rules, against a fake OIDC provider.

Tokens are signed with a key generated here and served from an `httpx.MockTransport`
discovery document, so nothing reaches a real provider. Handler resolution follows the Agent
Server's documented order, most specific first: (resource, action), (resource, "*"),
("*", action), then the global handler. The server's own resolver needs a running server's
settings to import, so it is not used here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser

from deep_life_sci import auth

ISSUER = "https://idp.example.edu"
CLIENT_ID = "deep-life-sci"


def _jwk(private: rsa.RSAPrivateKey, kid: str) -> dict[str, Any]:
    public = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict=True)
    return {**public, "kid": kid, "use": "sig", "alg": "RS256"}


@pytest.fixture
def idp(monkeypatch):
    """An issuer publishing one signing key, and a way to mint tokens from it."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    published = {"keys": [_jwk(key, "k1")]}
    fetches = []

    def handle(request: httpx.Request) -> httpx.Response:
        fetches.append(request.url.path)
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        return httpx.Response(200, json=published)

    monkeypatch.setattr(auth, "_keys", auth._SigningKeys(transport=httpx.MockTransport(handle)))
    monkeypatch.setenv("DEEP_LIFE_SCI_AUTH", "oidc")
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", CLIENT_ID)

    def mint(signing_key=key, kid="k1", algorithm="RS256", **claims: Any) -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "user-1", "iat": now,
                "exp": now + 300, **claims}
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(body, signing_key, algorithm=algorithm, headers={"kid": kid})

    return mint, fetches


async def _authenticate(token: str) -> dict:
    return await auth.authenticate(authorization=f"Bearer {token}")


async def _refused(token: str, status: int = 401) -> str:
    with pytest.raises(Auth.exceptions.HTTPException) as caught:
        await _authenticate(token)
    assert caught.value.status_code == status
    return caught.value.detail


class TestSignIn:
    async def test_a_valid_token_names_the_user_by_sub(self, idp):
        mint, _ = idp
        user = await _authenticate(mint(name="Ada Lovelace"))
        assert user == {"identity": "user-1", "display_name": "Ada Lovelace",
                        "permissions": [auth.USER]}

    @pytest.mark.parametrize(
        "claims",
        [{"iss": "https://someone-else.example"}, {"aud": "another-app"},
         {"exp": int(time.time()) - 3600}, {"sub": None}],
        ids=["issuer", "audience", "expired", "no-sub"],
    )
    async def test_a_token_for_someone_else_is_refused(self, idp, claims):
        mint, _ = idp
        await _refused(mint(**claims))

    async def test_a_token_signed_by_another_key_is_refused(self, idp):
        mint, _ = idp
        forger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        await _refused(mint(signing_key=forger))

    async def test_symmetric_and_unsigned_tokens_are_refused_before_any_key_lookup(self, idp):
        _, fetches = idp
        hs = jwt.encode({"iss": ISSUER, "aud": CLIENT_ID, "sub": "x"}, "s" * 32, "HS256")
        assert "unsupported signing algorithm" in await _refused(hs)
        assert "not a signed JWT" in await _refused("opaque-google-access-token")
        assert fetches == []

    async def test_an_unknown_key_id_refetches_at_most_once_per_window(self, idp):
        mint, fetches = idp
        await _authenticate(mint())
        for _ in range(3):
            await _refused(mint(kid="rotated-away"))
        assert fetches.count("/jwks") == 1

    async def test_langsmith_mode_refuses_every_bearer_token(self, idp, monkeypatch):
        mint, fetches = idp
        monkeypatch.setenv("DEEP_LIFE_SCI_AUTH", "langsmith")
        assert "LangSmith API keys only" in await _refused(mint())
        monkeypatch.delenv("DEEP_LIFE_SCI_AUTH")
        await _refused(mint())
        assert fetches == []

    async def test_an_unset_audience_refuses_rather_than_accepting_any_app(self, idp, monkeypatch):
        mint, _ = idp
        monkeypatch.delenv("OIDC_AUDIENCE")
        assert "OIDC_AUDIENCE" in await _refused(mint())

    async def test_the_client_id_is_the_audience_when_none_is_set(self, idp, monkeypatch):
        """The UI's ID tokens are for its own client ID; OIDC_AUDIENCE is for access tokens."""
        mint, _ = idp
        monkeypatch.delenv("OIDC_AUDIENCE")
        monkeypatch.setenv("OIDC_CLIENT_ID", CLIENT_ID)
        await _authenticate(mint())
        await _refused(mint(aud="another-app"))

    async def test_email_domains_restrict_when_set(self, idp, monkeypatch):
        mint, _ = idp
        monkeypatch.setenv("OIDC_ALLOWED_EMAIL_DOMAINS", "example.edu, lab.example.org")
        await _authenticate(mint(email="ada@Example.edu"))
        await _authenticate(mint(email="bo@lab.example.org"))
        await _refused(mint(email="eve@gmail.com"), 403)
        await _refused(mint(email="ada@example.edu", email_verified=False), 403)
        await _refused(mint(), 403)

    @pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer "])
    async def test_a_missing_or_non_bearer_header_is_refused(self, idp, header):
        with pytest.raises(Auth.exceptions.HTTPException) as caught:
            await auth.authenticate(authorization=header)
        assert caught.value.status_code == 401


def _ctx(user: Any, permissions: list[str], resource: str, action: str) -> Auth.types.AuthContext:
    return Auth.types.AuthContext(user=user, permissions=permissions, resource=resource,
                                  action=action)


class _User:
    """The user object the server builds from `authenticate`'s return value."""

    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.is_authenticated = True
        self.display_name = identity
        self.permissions = [auth.USER]


SIGNED_IN = ([auth.USER], _User("user-1"))
LANGSMITH_KEY = (["authenticated"], _User("langsmith-user-id"))
STUDIO = (["authenticated"], StudioUser("dev@example.org", is_authenticated=True))


async def _decide(caller, resource: str, action: str, value: dict | None = None):
    """What the most specific registered handler makes of this request."""
    permissions, user = caller
    ctx = _ctx(user, permissions, resource, action)
    handlers = auth.auth._handlers
    for key in ((resource, action), (resource, "*"), ("*", action)):
        if key in handlers:
            handler = handlers[key][0]
            break
    else:
        handler = auth.auth._global_handlers[0]
    value = {} if value is None else value
    return await handler(ctx=ctx, value=value), value


class TestAccess:
    async def test_signed_in_users_own_what_they_create_and_see_only_that(self):
        filters, value = await _decide(SIGNED_IN, "threads", "create", {"metadata": {"a": 1}})
        assert filters == {"owner": "user-1"}
        assert value["metadata"] == {"a": 1, "owner": "user-1"}
        for action in ("read", "search", "update", "delete", "create_run"):
            filters, _ = await _decide(SIGNED_IN, "threads", action)
            assert filters == {"owner": "user-1"}, action

    async def test_ownership_cannot_be_handed_to_someone_else(self):
        _, value = await _decide(SIGNED_IN, "threads", "update",
                                 {"metadata": {"owner": "user-2"}})
        assert value["metadata"]["owner"] == "user-1"

    async def test_signed_in_users_may_read_the_assistant_but_not_change_it(self):
        assert (await _decide(SIGNED_IN, "assistants", "read"))[0] is None
        assert (await _decide(SIGNED_IN, "assistants", "search"))[0] is None
        for action in ("create", "update", "delete"):
            with pytest.raises(Auth.exceptions.HTTPException) as caught:
                await _decide(SIGNED_IN, "assistants", action)
            assert caught.value.status_code == 403

    async def test_anything_not_allowed_is_refused_to_signed_in_users(self):
        with pytest.raises(Auth.exceptions.HTTPException):
            await _decide(SIGNED_IN, "store", "search")

    @pytest.mark.parametrize("caller", [LANGSMITH_KEY, STUDIO], ids=["langsmith-key", "studio"])
    @pytest.mark.parametrize(
        ("resource", "action"),
        [("threads", "search"), ("threads", "create_run"), ("assistants", "update"),
         ("crons", "create")],
    )
    async def test_the_workspace_sees_everything(self, caller, resource, action):
        filters, value = await _decide(caller, resource, action, {"metadata": {}})
        assert filters is None
        assert value == {"metadata": {}}


# --- the provider's keys: cache, outages, and keys or tokens that do not fit ---------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def provider(monkeypatch):
    """An issuer whose availability and published keys a test controls, cached on a clock
    the test moves. `calls` records every request that reaches it."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    state = SimpleNamespace(keys=[_jwk(key, "k1")], down=False, calls=[], discovery=None,
                            clock=_Clock(), key=key)

    def handle(request: httpx.Request) -> httpx.Response:
        state.calls.append(request.url.path)
        if state.down:
            return httpx.Response(503)
        if request.url.path == "/.well-known/openid-configuration":
            return state.discovery or httpx.Response(
                200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        return httpx.Response(200, json={"keys": state.keys})

    monkeypatch.setattr(auth, "_keys", auth._SigningKeys(
        transport=httpx.MockTransport(handle), clock=state.clock))
    monkeypatch.setenv("DEEP_LIFE_SCI_AUTH", "oidc")
    monkeypatch.setenv("OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("OIDC_AUDIENCE", CLIENT_ID)

    def mint(signing_key=key, kid="k1", algorithm="RS256") -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "user-1", "iat": now, "exp": now + 300}
        return jwt.encode(body, signing_key, algorithm=algorithm, headers={"kid": kid})

    state.mint = mint
    return state


def _forged(header: dict[str, Any]) -> str:
    """A token with this header and a signature nobody made."""
    def part(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    payload = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "x", "exp": int(time.time()) + 300}
    return ".".join([part(json.dumps(header).encode()), part(json.dumps(payload).encode()),
                     part(b"not a signature")])


class TestSigningKeys:
    async def test_an_outage_past_the_ttl_keeps_signed_in_users_signed_in(self, provider):
        await _authenticate(provider.mint())
        provider.clock.now += auth.KEYS_TTL_SECONDS + 1
        provider.down, provider.calls[:] = True, []
        users = await asyncio.gather(*(_authenticate(provider.mint()) for _ in range(10)))
        assert all(user["identity"] == "user-1" for user in users)
        # One attempt for all ten, not one each in turn behind a lock.
        assert len(provider.calls) == 1
        await _authenticate(provider.mint())
        assert len(provider.calls) == 1

    async def test_failed_refreshes_are_throttled_like_successful_ones(self, provider):
        provider.down = True
        for _ in range(3):
            assert "could not fetch" in await _refused(provider.mint())
        assert len(provider.calls) == 1
        provider.clock.now += auth.REFETCH_SECONDS
        provider.down = False
        await _authenticate(provider.mint())

    async def test_forged_key_ids_in_an_outage_cost_one_request_a_window(self, provider):
        await _authenticate(provider.mint())
        provider.clock.now += auth.REFETCH_SECONDS + 1
        provider.down, provider.calls[:] = True, []
        forged = [provider.mint(kid=f"forged-{i}") for i in range(10)]
        for outcome in await asyncio.gather(*(_authenticate(t) for t in forged),
                                            return_exceptions=True):
            assert isinstance(outcome, Auth.exceptions.HTTPException)
            assert outcome.status_code == 401
        assert len(provider.calls) == 1
        await _authenticate(provider.mint())  # the real key still verifies

    async def test_a_key_rotated_out_stops_verifying_once_a_refresh_succeeds(self, provider):
        await _authenticate(provider.mint())
        newer = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        provider.keys = [_jwk(newer, "k2")]
        provider.clock.now += auth.KEYS_TTL_SECONDS + 1
        assert "does not publish" in await _refused(provider.mint())
        await _authenticate(provider.mint(signing_key=newer, kid="k2"))

    async def test_cached_keys_give_out_after_a_day_of_outage(self, provider):
        await _authenticate(provider.mint())
        provider.clock.now += auth.MAX_STALE_SECONDS + 1
        provider.down = True
        assert "could not fetch" in await _refused(provider.mint())

    async def test_keys_pyjwt_cannot_load_are_skipped_not_fatal(self, provider):
        provider.keys = [
            {"kty": "OKP", "crv": "X25519", "kid": "ecdh", "x": "aGVsbG8"},
            {"kty": "OKP", "crv": "Ed448", "kid": "ed448-no-x"},
            {"kty": "RSA", "kid": "x5c-only", "x5c": ["MIIBIjANBg"]},
            {"kid": "no-kty"},
            {"kty": "EC", "crv": "P-256", "kid": "garbled", "x": "!!", "y": "!!"},
            "not even an object",
            *provider.keys,
        ]
        await _authenticate(provider.mint())

    @pytest.mark.parametrize("discovery", [
        httpx.Response(200, text="<html>Sign in</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"issuer": ISSUER}),
        httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": "not a url"}),
    ], ids=["html", "a-list", "no-jwks-uri", "bad-jwks-uri"])
    async def test_a_discovery_document_that_cannot_be_used_is_a_401(self, provider, discovery):
        provider.discovery = discovery
        assert "could not fetch" in await _refused(provider.mint())

    async def test_no_algorithm_and_key_pairing_escapes_as_an_error(self, provider):
        """Every algorithm against every kind of key, with and without the key naming its
        own algorithm: a 401 unless the token really is signed with that key."""
        from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519

        private = {
            "rsa": provider.key,
            "p256": ec.generate_private_key(ec.SECP256R1()),
            "p384": ec.generate_private_key(ec.SECP384R1()),
            "p521": ec.generate_private_key(ec.SECP521R1()),
            "ed25519": ed25519.Ed25519PrivateKey.generate(),
            "ed448": ed448.Ed448PrivateKey.generate(),
        }
        signs_with = {"rsa": "RS256", "p256": "ES256", "p384": "ES384", "p521": "ES512",
                      "ed25519": "EdDSA", "ed448": "EdDSA"}
        family = {"RS": jwt.algorithms.RSAAlgorithm, "ES": jwt.algorithms.ECAlgorithm,
                  "Ed": jwt.algorithms.OKPAlgorithm}
        published = []
        for name, key in private.items():
            jwk = family[signs_with[name][:2]].to_jwk(key.public_key(), as_dict=True)
            published.append({**jwk, "kid": name})
            published.append({**jwk, "kid": f"{name}-alg", "alg": signs_with[name]})
        provider.keys = published
        for kid in [entry["kid"] for entry in published]:
            for algorithm in auth.ALGORITHMS:
                await _refused(_forged({"alg": algorithm, "kid": kid}))
        for name, key in private.items():
            for kid in (name, f"{name}-alg"):
                token = provider.mint(signing_key=key, kid=kid, algorithm=signs_with[name])
                if kid == "ed448":
                    # PyJWT cannot tell an Ed448 key's algorithm unless the key names it, so
                    # that one is skipped as unloadable rather than failing the whole set.
                    assert "does not publish" in await _refused(token)
                else:
                    await _authenticate(token)

    @pytest.mark.parametrize("kid", [["k1"], {"k": 1}, 7])
    async def test_a_key_id_that_is_not_a_string_is_a_401(self, provider, kid):
        await _refused(_forged({"alg": "RS256", "kid": kid}))
        assert provider.calls == []
