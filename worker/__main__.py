"""`python -m worker`: the image entry point.

Normal path: start the transport named by TRANSPORT (queue | http).

Pod path: with POD_SCRIPT set (e.g. scripts/pod_http_parity.sh), clone the repo to
/root/worker-decide and exec that script there, so a job pod needs only environment variables
(Runpod's template start-command field has twice not reached the container).

Either way, HF_HOME falls back from /runpod-volume (serverless mount) to /workspace (pod mount)
when the former does not exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO = "https://github.com/ashwinsreedhar28/worker-decide"
POD_CHECKOUT = Path("/root/worker-decide")


def fix_hf_home(env: dict[str, str] | os._Environ = os.environ) -> str | None:
    """Rewrite HF_HOME=/runpod-volume/... to /workspace/... when only /workspace is mounted."""
    hf = env.get("HF_HOME", "")
    if hf.startswith("/runpod-volume") and not Path("/runpod-volume").exists():
        if Path("/workspace").exists():
            env["HF_HOME"] = "/workspace" + hf[len("/runpod-volume") :]
            return env["HF_HOME"]
    return None


def run_pod_script(
    script: str, env: dict[str, str] | os._Environ = os.environ, tries: int = 20, wait: float = 6.0
) -> None:
    """Clone (retrying: a fresh pod's DNS can take a minute to come up) and exec the script."""
    repo = env.get("REPO_URL", DEFAULT_REPO)
    ref = env.get("REPO_REF", "main")
    cmd = ["git", "clone", "--branch", ref, "--depth", "1", repo, str(POD_CHECKOUT)]
    for attempt in range(1, tries + 1):
        if POD_CHECKOUT.exists():
            break
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"clone attempt {attempt}/{tries} failed: {r.stderr.strip()[-300:]}", flush=True)
        if attempt == tries:
            sys.exit("giving up on git clone")
        time.sleep(wait)
    os.chdir(POD_CHECKOUT)
    print(f"running {script} in {POD_CHECKOUT}", flush=True)
    os.execvp("bash", ["bash", script])


def main() -> None:
    moved = fix_hf_home()
    if moved:
        print(f"HF_HOME -> {moved} (/runpod-volume not mounted)", flush=True)
    script = os.getenv("POD_SCRIPT")
    if script:
        run_pod_script(script)
        return  # pragma: no cover - execvp does not return
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
