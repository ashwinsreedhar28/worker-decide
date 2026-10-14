"""Runpod queue-based entry point: loads the engine once, then serves jobs through WorkerCore.

    python -m worker.handler                     # on the endpoint
    python -m worker.handler --rp_serve_api      # local HTTP test server from the Runpod SDK
    python -m worker.handler --test_input '{"input": {...}}'

Job input is the SystemOne request body (plus optional `"debug": true`); output is the
SystemOne response body. Environment variables are listed in README.md.
"""

from __future__ import annotations

import runpod

from deciders.clef.engine import ClefEngine
from worker.core import WorkerCore, config_from_env

core = WorkerCore(ClefEngine(config_from_env()))


def handler(job):
    return core.handle_job(job)


runpod.serverless.start({"handler": handler})
