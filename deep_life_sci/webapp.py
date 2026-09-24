"""Custom HTTP routes served beside the LangGraph API (`langgraph.json`'s `http.app`).

`GET /models` reports the model, gateway path and effort each role resolved to, for the chat
UI's model badge. It is read-only: models are chosen in `models.yaml` (or by env override),
so the server that runs them is the only honest source for what is running. It re-reads
models.yaml first, as each run does, so the badge and the next run agree after an edit.

A setting that cannot work answers 500 with `{"error": message}` for the badge to show.
It must not raise: the config checks raise SystemExit, which inside a request stops the
server's event loop under `langgraph dev`.

Starlette is not declared by this package because nothing outside the API server imports
this module, and the server always ships it.
"""

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from deep_life_sci import models


async def models_route(_: Request) -> JSONResponse:
    try:
        await asyncio.to_thread(models.refresh)
        # The judge is left out: it grades evals and never runs behind the chat UI.
        return JSONResponse({"roles": models.summary(*models.CHAT_ROLES)})
    except SystemExit as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


app = Starlette(routes=[Route("/models", models_route, methods=["GET"])])
