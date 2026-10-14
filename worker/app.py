"""HTTP transport for Runpod load-balancing endpoints (and anything else that speaks HTTP).

    POST /v1/decide     SystemOne request body -> SystemOne response body
    POST /v1/systemone  (alias) header `X-Decide-Debug: 1` adds the `debug` block (parity_http)
    GET  /ping          204 while the engine loads, 200 once ready, 500 if loading failed
                        (Runpod's health contract: 200 healthy, 204 initializing, else unhealthy)
    GET  /              engine info and state

The engine loads in a background thread so the port is open from the first second and
/ping can report "initializing" instead of refusing connections. Requests are handled in the
threadpool (sync route) and serialized by the engine's own lock. Set PORT and PORT_HEALTH to the
same value (default 8000) on the endpoint; see README.md.

    python -m worker.app            # uvicorn on 0.0.0.0:$PORT
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from worker.core import WorkerCore

MAX_BODY_BYTES = 25 * 1024 * 1024  # Runpod's payload limit is 30 MB; leave room for headers
TRUE_VALUES = {"1", "true", "yes"}


def default_loader() -> WorkerCore:
    from deciders.clef.engine import ClefEngine
    from worker.core import config_from_env

    return WorkerCore(ClefEngine(config_from_env()))


def create_app(
    loader: Callable[[], WorkerCore] = default_loader, core: WorkerCore | None = None
) -> FastAPI:
    """`core` given: ready immediately (tests). Otherwise `loader` runs in a thread at startup."""
    state: dict[str, Any] = {"core": core, "error": None}

    def load() -> None:
        try:
            state["core"] = loader()
        except Exception as e:  # noqa: BLE001 - reported through /ping, not raised in a thread
            state["error"] = f"{type(e).__name__}: {e}"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["core"] is None:
            threading.Thread(target=load, name="engine-load", daemon=True).start()
        yield

    app = FastAPI(title="worker-decide", lifespan=lifespan)
    app.state.worker = state

    @app.get("/ping")
    def ping() -> Response:
        if state["error"]:
            return JSONResponse({"status": "unhealthy", "error": state["error"]}, status_code=500)
        if state["core"] is None:
            return Response(status_code=204)
        return JSONResponse({"status": "healthy"})

    @app.get("/")
    def root() -> JSONResponse:
        c = state["core"]
        return JSONResponse(
            {
                "worker": "worker-decide",
                "state": "error" if state["error"] else ("ready" if c else "loading"),
                "error": state["error"],
                "engine": c.info if c else None,
                "routes": ["POST /v1/decide", "POST /v1/systemone", "GET /ping", "GET /"],
            }
        )

    @app.post("/v1/decide")
    @app.post("/v1/systemone")  # same handler; the README's wire-format name
    async def decide(request: Request) -> JSONResponse:
        c = state["core"]
        if c is None:
            return JSONResponse(
                {"error": state["error"] or "engine is still loading"},
                status_code=503,
                headers={"Retry-After": "5"},
            )
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"error": f"body exceeds {MAX_BODY_BYTES} bytes"}, status_code=413)
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse({"error": f"body exceeds {MAX_BODY_BYTES} bytes"}, status_code=413)
        try:
            import json

            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"error": "body is not valid JSON"}, status_code=400)
        debug = request.headers.get("x-decide-debug", "").lower() in TRUE_VALUES
        import anyio

        o = await anyio.to_thread.run_sync(c.decide, body, debug)
        return JSONResponse(o.body, status_code=o.status)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
