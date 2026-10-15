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
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from deciders.clef.engine import DEFAULT_MODEL, EngineConfig, images_from_data_urls
from deciders.systemone import validate_request
from worker.batcher import Batcher

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
        fused=e.get("FUSED", ""),
    )


@dataclass
class BatchSettings:
    """Dynamic batching (worker/batcher.py). max_batch 1 keeps the one-at-a-time path."""

    max_batch: int = 1
    max_padded_tokens: int = 32_768
    max_ratio: float = 2.0
    max_wait_ms: float = 0.0
    timeout_s: float = 600.0  # a request waiting longer than this for its batch gets a 500


def batch_settings_from_env(env: dict[str, str] | None = None) -> BatchSettings:
    e = os.environ if env is None else env
    return BatchSettings(
        max_batch=int(e.get("BATCH_MAX", "1")),
        max_padded_tokens=int(e.get("BATCH_MAX_TOKENS", "32768")),
        max_ratio=float(e.get("BATCH_MAX_RATIO", "2.0")),
        max_wait_ms=float(e.get("BATCH_MAX_WAIT_MS", "0")),
        timeout_s=float(e.get("BATCH_TIMEOUT_S", "600")),
    )


@dataclass
class Outcome:
    status: int
    body: dict[str, Any]


class WorkerCore:
    def __init__(self, engine: Any, batch: BatchSettings | None = None):
        self.engine = engine
        self.batch = batch or BatchSettings()
        self.batcher: Batcher | None = None
        # A dry-run engine has no model (model None): nothing to batch.
        has_model = getattr(engine, "model", True) is not None
        if self.batch.max_batch > 1 and hasattr(engine, "forward_batch") and has_model:
            self.batcher = Batcher(
                self._run_batch,
                max_batch=self.batch.max_batch,
                max_padded_tokens=self.batch.max_padded_tokens,
                max_ratio=self.batch.max_ratio,
                max_wait_ms=self.batch.max_wait_ms,
            )
        timeline = getattr(engine, "timeline", None)
        if timeline is not None:  # both transports build the core, then start serving
            timeline.mark("ready")
        m = engine.manifest()
        self.info = {
            k: m.get(k)
            for k in ("model", "revision", "dtype", "sdp", "fused", "gdn_fast_path", "gdn_kernels",
                      "versions", "gpu", "worker_decide_commit", "load_s", "snapshot_source",
                      "warmup_s", "timeline")
        }  # fmt: skip
        self.info["code_sha"] = (m.get("files", {}).get("joint_schema_model.py") or {}).get(
            "sha256"
        )
        self.info["batching"] = asdict(self.batch) if self.batcher else None

    @property
    def concurrency(self) -> int:
        """Jobs to run at once on the queue transport: enough to fill a batch."""
        return self.batch.max_batch if self.batcher else 1

    def _run_batch(self, prepared: list[Any]) -> list[Any]:
        t0 = time.perf_counter()
        outs = self.engine.forward_batch([p.encoded for p in prepared])
        ms = (time.perf_counter() - t0) * 1000
        info = {"size": len(prepared), "tokens": [p.tokens for p in prepared],
                "forward_ms": round(ms, 2)}  # fmt: skip
        return [(o, ms, info) for o in outs]

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
        rid = uuid.uuid4().hex[:12]
        batch_info = None
        try:
            if self.batcher is not None and not images:  # image requests run alone
                p = self.engine.prepare(body, None, rid=rid)
                fut = self.batcher.submit(p, p.tokens)
                try:
                    per_q, forward_ms, batch_info = fut.result(timeout=self.batch.timeout_s)
                except TimeoutError as e:  # concurrent.futures.TimeoutError since 3.11
                    raise RuntimeError(
                        f"no answer from the batcher in {self.batch.timeout_s} s"
                    ) from e
                d = self.engine.finish(p, per_q, forward_ms)
            else:
                d = self.engine.decide(body, images, rid=rid)
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
                "batch": batch_info,
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
