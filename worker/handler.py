"""Runpod queue-based entry point: loads the engine once, then serves jobs through WorkerCore.

    python -m worker.handler                     # on the endpoint
    python -m worker.handler --rp_serve_api      # local HTTP test server from the Runpod SDK
    python -m worker.handler --test_input '{"input": {...}}'

Job input is the SystemOne request body (plus optional `"debug": true`); output is the
SystemOne response body. Environment variables are listed in README.md.

With BATCH_MAX > 1 the handler is async, so the SDK runs up to BATCH_MAX jobs at once
(`concurrency_modifier`), each in a thread, and the batcher shares forward passes between
them. A job cancelled by a stop signal then stops being awaited while its thread finishes the
forward. With the default BATCH_MAX=1 the handler is the plain synchronous one, as before.
"""

from __future__ import annotations

import asyncio

import runpod

from deciders.clef.engine import ClefEngine
from worker.core import WorkerCore, batch_settings_from_env, config_from_env

core = WorkerCore(ClefEngine(config_from_env()), batch_settings_from_env())


async def batched_handler(job):
    return await asyncio.to_thread(core.handle_job, job)


def concurrency_modifier(current: int) -> int:
    return core.concurrency


if core.concurrency > 1:
    runpod.serverless.start(
        {"handler": batched_handler, "concurrency_modifier": concurrency_modifier}
    )
else:
    runpod.serverless.start({"handler": core.handle_job})
