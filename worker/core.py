"""Transport-agnostic request handling: a SystemOne body in, a SystemOne body out.

Both transports call this — the queue handler (``worker/handler.py``, Runpod's `{"input": ...}`
jobs) and the HTTP app (``worker/app.py``, load-balancing endpoints) — so the two cannot drift.
Nothing here imports torch or the Runpod SDK; the engine is injected, which is also how the
tests run without a GPU.

Statuses are HTTP-shaped even on the queue path (the handler maps them): 200 answered,
400 the request is at fault (validation, undecodable image, encoder rejection such as a
schema longer than max_length), 500 the forward pass failed (OOM, kernel).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from deciders.clef.engine import DEFAULT_MODEL, EngineConfig, images_from_data_urls
from deciders.systemone import validate_request

PINNED_REVISION = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
PINNED_CODE_SHA = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
MAX_IMAGES = 8


def config_from_env(env: dict[str, str] | None = None) -> EngineConfig:
    """Engine settings from the endpoint's environment (README has the table)."""
    e = os.environ if env is None else env
    return EngineConfig(
        model=e.get("MODEL_NAME", DEFAULT_MODEL),
        revision=e.get("MODEL_REVISION", PINNED_REVISION) or None,
        dtype=e.get("DTYPE", "bfloat16"),
        device=e.get("DEVICE", "cuda"),
        max_length=int(e.get("MAX_LENGTH", "16384")),
        sdp=e.get("SDP", "efficient"),
        expect_code_sha=e.get("CODE_SHA", PINNED_CODE_SHA) or None,
        warmup=e.get("WARMUP", "0").strip().lower() in {"1", "true", "yes"},
    )


@dataclass
class Outcome:
    status: int
    body: dict[str, Any]


class WorkerCore:
    def __init__(self, engine: Any):
        self.engine = engine
        timeline = getattr(engine, "timeline", None)
        if timeline is not None:  # both transports build the core, then start serving
            timeline.mark("ready")
        m = engine.manifest()
        self.info = {
            k: m.get(k)
            for k in ("model", "revision", "dtype", "sdp", "gdn_fast_path", "versions", "gpu",
                      "worker_decide_commit", "load_s", "snapshot_source", "warmup_s",
                      "timeline")
        }  # fmt: skip
        self.info["code_sha"] = (m.get("files", {}).get("joint_schema_model.py") or {}).get(
            "sha256"
        )

    def decide(self, body: Any, debug: bool = False) -> Outcome:
        if not isinstance(body, dict):
            return Outcome(400, {"error": "request body must be a JSON object"})
        problems = validate_request(body)
        if problems:
            return Outcome(400, {"error": "invalid request", "problems": problems})
        images = None
        if "images" in body:
            urls = body["images"]
            if not isinstance(urls, list) or len(urls) > MAX_IMAGES:
                return Outcome(400, {"error": f"images must be a list of at most {MAX_IMAGES}"})
            try:
                images = images_from_data_urls(urls)
            except ValueError as e:
                return Outcome(400, {"error": str(e)})
        try:
            d = self.engine.decide(body, images, rid=uuid.uuid4().hex[:12])
        except ValueError as e:  # the encoder's own rejections
            return Outcome(400, {"error": str(e)})
        except RuntimeError as e:  # torch OOM is a RuntimeError subclass; kernel failures too
            return Outcome(500, {"error": "inference failed: " + str(e).split("\n")[0][:200]})
        out = dict(d.response)
        if debug:
            out["debug"] = {
                "request_sha": d.request_sha,
                "encoding": d.encoding,
                "questions": d.questions,
                "timing_ms": d.timing_ms,
                "engine": self.info,
            }
        return Outcome(200, out)

    # -- Runpod queue job ------------------------------------------------------------------

    def handle_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """`{"input": <SystemOne body, optional "debug": true>}` -> output dict.

        A dict with an `error` key marks the job FAILED in Runpod; anything else is COMPLETED.
        `debug` is popped before hashing so request_sha matches the fixtures.
        """
        inp = job.get("input")
        if not isinstance(inp, dict):
            return {"error": "input must be a JSON object (the SystemOne request body)"}
        body = dict(inp)
        debug = bool(body.pop("debug", False))
        o = self.decide(body, debug=debug)
        if o.status != 200:
            return {"error": o.body.get("error", "request failed"), **{
                k: v for k, v in o.body.items() if k != "error"
            }, "status": o.status}  # fmt: skip
        return o.body
