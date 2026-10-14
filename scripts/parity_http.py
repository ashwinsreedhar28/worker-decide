#!/usr/bin/env python3
"""Run the fixture requests through a worker over HTTP and write a parity run directory.

Posts every record in the request files to `POST <base>/v1/decide` with `X-Decide-Debug: 1`,
turns each reply into a record in the oracle schema (request_sha, encoding, per-question
logits/probs, response, error, timing) and writes it to RUN_DIR, then runs the parity gate
against a reference directory. Image records are sent with their PNGs inline as base64 data
URLs; `request_sha` is computed locally from the fixture body, so a worker that hashes
differently fails the request gate.

    python scripts/parity_http.py http://127.0.0.1:8000 runs/http-bf16 \\
        --reference fixtures/clef-flash/bfloat16 --max-dp 0.005 \\
        [--also fixtures/clef-flash/float32] [--requests fixtures/requests/*.jsonl] [--token T]

Exit code is the gate's (0 pass, 1 fail). `--also` prints an informational report against a
second reference (fp32 drift) that never affects the exit code.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from parity import compare, render  # noqa: E402

from deciders.clef.engine import request_sha  # noqa: E402
from deciders.clef.oracle import load_requests  # noqa: E402


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def wire_body(line: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """The fixture's request plus its images inline, as the worker expects them."""
    body = dict(line["request"])
    if line.get("images"):
        body["images"] = [data_url(root / p) for p in line["images"]]
    return body


def record_from_reply(line: dict[str, Any], status: int, reply: dict[str, Any]) -> dict[str, Any]:
    """Oracle-schema record from one HTTP reply (status, parsed JSON body)."""
    rec: dict[str, Any] = {
        "id": line["id"],
        "request_sha": request_sha(line["request"]),
        "encoding": None,
        "questions": [],
        "response": None,
        "error": None,
        "timing_ms": {},
    }
    if status != 200:
        rec["error"] = f"http {status}: {reply.get('error', '')}"[:200]
        return rec
    dbg = reply.get("debug") or {}
    rec["encoding"] = dbg.get("encoding")
    rec["questions"] = dbg.get("questions", [])
    rec["timing_ms"] = dbg.get("timing_ms", {})
    rec["response"] = {k: v for k, v in reply.items() if k != "debug"}
    if dbg.get("request_sha") and dbg["request_sha"] != rec["request_sha"]:
        # Keep the worker's hash as the record's so parity's request gate reports the drift.
        rec["request_sha_local"] = rec["request_sha"]
        rec["request_sha"] = dbg["request_sha"]
    return rec


def collect(lines: list[dict[str, Any]], run_dir: Path, poster) -> dict[str, int]:
    """Post every fixture line through `poster(body) -> (status, reply)` and write the records."""
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    for i, line in enumerate(lines, 1):
        t0 = time.perf_counter()
        status, reply = poster(wire_body(line))
        rec = record_from_reply(line, status, reply)
        rec["timing_ms"]["wall"] = round((time.perf_counter() - t0) * 1000, 1)
        (run_dir / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
        counts["error" if rec["error"] else "ok"] += 1
        tok = rec["encoding"]["input_tokens"] if rec["encoding"] else "-"
        err = f"  ERROR {rec['error'][:60]}" if rec["error"] else ""
        print(  # progress goes to stderr; stdout is the report
            f"[{i:>3}/{len(lines)}] {rec['id']:<24} tokens={tok!s:>6} "
            f"{rec['timing_ms']['wall']:>9.1f} ms{err}",
            file=sys.stderr,
            flush=True,
        )
    return counts


def post(base: str, body: dict[str, Any], token: str | None, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/decide",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Decide-Debug": "1",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {"error": str(e)}


def is_worker_unhealthy(status: int, body: bytes) -> bool:
    """Only our app's own failed-load answer is fatal. Behind Runpod's load balancer, 5xx
    responses also come from the gateway while no worker is up, and those must be waited out."""
    if status != 500:
        return False
    try:
        return json.loads(body).get("status") == "unhealthy"
    except (ValueError, AttributeError):
        return False


def wait_ready(base: str, token: str | None, timeout_s: float) -> None:
    t0 = time.time()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    while True:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(base.rstrip("/") + "/ping", headers=headers), timeout=10
            ) as r:
                if r.status == 200:
                    return
                state = f"status {r.status}"
        except urllib.error.HTTPError as e:
            body = e.read()[:500]
            if is_worker_unhealthy(e.code, body):
                raise SystemExit(f"worker reports unhealthy: {body!r}") from e
            state = f"status {e.code}"  # LB with no worker yet, gateway errors: keep waiting
        except (urllib.error.URLError, TimeoutError) as e:
            state = str(e)
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"worker not ready after {timeout_s:.0f}s ({state})")
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", help="worker base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=ROOT / "fixtures/clef-flash/bfloat16")
    ap.add_argument("--also", type=Path, action="append", default=[], help="informational refs")
    ap.add_argument("--requests", nargs="+", default=[str(ROOT / "fixtures/requests/*.jsonl")])
    ap.add_argument("--max-dp", type=float, default=0.005)
    ap.add_argument("--mean-dp", type=float, default=0.001)
    ap.add_argument("--token", default=None, help="Runpod API key for an endpoint URL")
    ap.add_argument("--timeout", type=float, default=330.0, help="per request, seconds")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--json", type=Path, help="write the gate report as JSON")
    a = ap.parse_args()

    wait_ready(a.base, a.token, a.ready_timeout)
    lines = load_requests(a.requests)
    counts = collect(lines, a.run_dir, lambda body: post(a.base, body, a.token, a.timeout))
    (a.run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "base": a.base,
                "reference": str(a.reference),
                "counts": counts,
                "requests": a.requests,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        )
    )

    report = compare(a.reference, a.run_dir, max_dp=a.max_dp, mean_dp=a.mean_dp)
    print(render(report))
    if a.json:
        a.json.write_text(json.dumps(report.__dict__, default=lambda o: o.__dict__, indent=1))
    for ref in a.also:
        print(f"\n--- informational: vs {ref} (not gated) ---\n")
        print(render(compare(ref, a.run_dir)))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
