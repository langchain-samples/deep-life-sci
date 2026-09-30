"""The agent server's own routes: the model badge, and the chat UI a signed-in deployment serves."""

from __future__ import annotations

import pytest
from langgraph_sdk import Auth
from starlette.testclient import TestClient

from deep_life_sci import auth, models, paths, webapp


@pytest.fixture
def client_for(monkeypatch, tmp_path):
    """A test client for the app as a deployment in the given auth mode would build it."""
    ui = tmp_path / "ui"
    monkeypatch.setattr(models, "refresh", lambda: None)
    monkeypatch.setattr(models, "summary", lambda *roles: [{"role": r} for r in roles])

    def make(mode: str | None, *, built: bool = True) -> TestClient:
        if mode:
            monkeypatch.setenv("DEEP_LIFE_SCI_AUTH", mode)
        if built:
            (ui / "auth" / "callback").mkdir(parents=True, exist_ok=True)
            (ui / "index.html").write_text("<p>chat</p>")
            (ui / "auth" / "callback" / "index.html").write_text("<p>callback</p>")
        monkeypatch.setattr(paths, "UI_DIR", ui)
        return TestClient(webapp.create_app(), follow_redirects=False)

    return make


def test_a_local_or_langsmith_only_server_serves_no_ui(client_for):
    for mode in (None, "langsmith"):
        client = client_for(mode)
        assert client.get("/app/").status_code == 404
        assert client.get("/app/config.json").status_code == 404
        assert client.get("/models").json() == {"roles": [{"role": r} for r in models.CHAT_ROLES]}


def test_a_signed_in_deployment_serves_the_ui_at_app(client_for, monkeypatch):
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example.edu")
    monkeypatch.setenv("OIDC_CLIENT_ID", "client-1")
    client = client_for("oidc")
    assert client.get("/").headers["location"] == "/app/"
    assert client.get("/app/").text == "<p>chat</p>"
    # What the identity provider redirects back to, without the trailing slash.
    assert client.get("/app/auth/callback?code=x&state=y", follow_redirects=True).text == (
        "<p>callback</p>"
    )
    config = client.get("/app/config.json")
    assert config.headers["cache-control"] == "no-store"
    assert config.json() == {"auth": "oidc", "issuer": "https://idp.example.edu",
                             "clientId": "client-1", "scope": "openid profile email",
                             "token": "id"}


def test_the_config_passes_the_access_token_choice_and_nothing_else(client_for, monkeypatch):
    monkeypatch.setenv("OIDC_TOKEN", "access")
    monkeypatch.setenv("OIDC_SCOPE", "openid email offline_access")
    body = client_for("oidc").get("/app/config.json").json()
    assert body["token"] == "access" and body["scope"] == "openid email offline_access"
    monkeypatch.setenv("OIDC_TOKEN", "something-else")
    assert client_for("oidc").get("/app/config.json").json()["token"] == "id"


def test_models_needs_a_signed_in_user_when_sign_in_is_on(client_for, monkeypatch):
    async def authenticate(authorization):
        if authorization != "Bearer good":
            raise Auth.exceptions.HTTPException(status_code=401, detail="no")
        return {"identity": "u"}

    monkeypatch.setattr(auth, "authenticate", authenticate)
    client = client_for("oidc")
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer good"}).status_code == 200


def test_an_image_without_the_ui_says_so(client_for):
    response = client_for("oidc", built=False).get("/app/index.html")
    assert response.status_code == 503 and "not built" in response.text
