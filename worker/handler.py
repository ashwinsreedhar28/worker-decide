"""Runpod queue-based entry point: loads the engine once, then serves jobs through WorkerCore.

    python -m worker.handler                     # on the endpoint
    python -m worker.handler --rp_serve_api      # local HTTP test server from the Runpod SDK
    python -m worker.handler --test_input '{"input": {...}}'

Job input is the SystemOne request body (plus optional `"debug": true`); output is the
SystemOne response body. Environment variables are listed in README.md.

The handler is async so the SDK can run several jobs at once (`concurrency_modifier`); each runs
WorkerCore in a thread. With BATCH_MAX > 1 the worker takes up to BATCH_MAX jobs at a time and
the batcher shares forward passes between them; with the default 1 it takes one job at a time,
as before.
"""

from __future__ import annotations

import asyncio

import runpod

from deciders.clef.engine import ClefEngine
from worker.core import WorkerCore, batch_settings_from_env, config_from_env

core = WorkerCore(ClefEngine(config_from_env()), batch_settings_from_env())


async def handler(job):
    return await asyncio.to_thread(core.handle_job, job)


def concurrency_modifier(current: int) -> int:
    return core.concurrency


runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})
