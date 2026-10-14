"""Runpod queue entry point at the repository root, where the Hub expects it.

Thin shim over ``worker/handler.py`` (which loads the engine and starts
``runpod.serverless`` on import), so the SDK's local conventions work from the root:

    python handler.py --test_input "$(cat test_input.json)"
    python handler.py --rp_serve_api

The image's default CMD (``python3 -m worker``) reaches the same handler.
"""

import worker.handler  # noqa: F401  (runpod.serverless.start runs on import)
