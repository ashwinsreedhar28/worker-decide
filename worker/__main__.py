"""`python -m worker`: start the transport named by TRANSPORT (queue | http)."""

from __future__ import annotations

import os
import sys


def main() -> None:
    transport = os.getenv("TRANSPORT", "queue").lower()
    if transport == "queue":
        import worker.handler  # noqa: F401  (runpod.serverless.start runs on import)
    elif transport == "http":
        import uvicorn

        uvicorn.run("worker.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    else:
        sys.exit(f"TRANSPORT must be queue or http, got {transport!r}")


if __name__ == "__main__":
    main()
