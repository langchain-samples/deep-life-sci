"""Custom HTTP routes served beside the LangGraph API (`langgraph.json`'s `http.app`).

`GET /models` reports the model, gateway path and effort each role resolved to, for the chat
UI's model badge. It is read-only: models are chosen in `models.yaml` (or by env override)
and take effect on restart, so the server that runs them is the only honest source for what
is running. A value baked into the frontend at build time would go stale on the first edit.

Starlette is not declared by this package because nothing outside the API server imports
this module, and the server always ships it.
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from deep_life_sci.models import summary


async def models(_: Request) -> JSONResponse:
    # The judge is left out: it grades evals and never runs behind the chat UI.
    return JSONResponse({"roles": summary("root", "subagent", "search")})


app = Starlette(routes=[Route("/models", models, methods=["GET"])])
