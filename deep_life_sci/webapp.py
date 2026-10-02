"""Custom HTTP routes served beside the LangGraph API (`langgraph.json`'s `http.app`).

`GET /models` reports the model, gateway path and effort each role resolved to, for the chat
UI's model badge. It is read-only: models are chosen in `models.yaml` (or by env override),
so the server that runs them is the only honest source for what is running. It re-reads
models.yaml first, as each run does, so the badge and the next run agree after an edit.

A setting that cannot work answers 500 with `{"error": message}` for the badge to show.
It must not raise: the config checks raise SystemExit, which inside a request stops the
server's event loop under `langgraph dev`.

**A deployment with sign-in also serves the chat UI** (`DEEP_LIFE_SCI_AUTH=oidc`; see
auth.py). The deploy image builds it to `paths.UI_DIR`, and it is served under `/app` with
`/` redirecting there. One origin for the page, the API and the `/ui/*` artifact scripts is
what lets those scripts load at all, with no proxy and no CORS. `/app/config.json` hands the
page its sign-in settings at run time, so one build works with any provider.

Custom routes are outside the platform's auth (`enable_custom_route_auth` would lock the
sign-in page too), and `/models` asks for none in either mode. What it says — which models
the agent runs, at what effort — is what README.md says too, and a route checking only the
signed-in half of the platform's callers refused the other half: a LangSmith key, as
`dev.py --remote` sends one, cannot be verified here. Without sign-in the UI is not served:
a browser holds no LangSmith key to reach the API with.

Starlette is not declared by this package because nothing outside the API server imports
this module, and the server always ships it.
"""

import asyncio
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from deep_life_sci import models, paths

UI_PATH = "/app"


def _signing_in() -> bool:
    return os.environ.get("DEEP_LIFE_SCI_AUTH", "").strip() == "oidc"


async def models_route(_: Request) -> JSONResponse:
    try:
        await asyncio.to_thread(models.refresh)
        # The judge is left out: it grades evals and never runs behind the chat UI.
        return JSONResponse({"roles": models.summary(*models.CHAT_ROLES)})
    except SystemExit as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def ui_config(_: Request) -> JSONResponse:
    """What the page needs to sign in; nothing here is secret (see frontend/src/lib/auth.ts)."""
    return JSONResponse({
        "auth": "oidc",
        "issuer": os.environ.get("OIDC_ISSUER", "").strip(),
        "clientId": os.environ.get("OIDC_CLIENT_ID", "").strip(),
        "scope": os.environ.get("OIDC_SCOPE", "").strip() or "openid profile email",
        "token": "access" if os.environ.get("OIDC_TOKEN", "").strip() == "access" else "id",
    }, headers={"Cache-Control": "no-store"})


async def to_ui(_: Request) -> RedirectResponse:
    return RedirectResponse(f"{UI_PATH}/")


async def ui_missing(_: Request) -> PlainTextResponse:
    return PlainTextResponse(f"The chat UI was not built into this image ({paths.UI_DIR}).",
                             status_code=503)


def create_app() -> Starlette:
    routes = [Route("/models", models_route, methods=["GET"])]
    if _signing_in():
        routes.append(Route("/", to_ui, methods=["GET"]))
        routes.append(Route(f"{UI_PATH}/config.json", ui_config, methods=["GET"]))
        if paths.UI_DIR.is_dir():
            routes.append(Mount(UI_PATH, StaticFiles(directory=paths.UI_DIR, html=True)))
        else:
            routes.append(Route(f"{UI_PATH}/{{rest:path}}", ui_missing, methods=["GET"]))
    return Starlette(routes=routes)


app = create_app()
