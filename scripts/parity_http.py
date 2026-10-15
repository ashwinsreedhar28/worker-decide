#!/usr/bin/env python3
"""Run the fixture requests through a worker over HTTP and gate the answers.

Posts every record in the request files to `POST <base>/v1/decide` with `X-Decide-Debug: 1`,
turns each reply into a record in the oracle schema (request_sha, encoding, per-question
logits/probs, response, error, timing) and writes it to RUN_DIR. Image records are sent with
their PNGs inline as base64 data URLs; `request_sha` is computed locally from the fixture body,
so a worker that hashes differently fails the request gate.

Two gates, with different jobs (decided Oct 14, after an A40 failed a same-dtype gate against
the H100 bf16 run while being closer to fp32 than the H100 was):

- correctness, any GPU (default, always on): against the fp32 truth. Argmax must agree where
  the fp32 margin exceeds the near-tie window (2 x max-dp); max |dp| <= 0.025, mean of record
  means <= 0.0025. Set from measured bf16 noise on two architectures (H100 worst 0.0211, A40
  0.0163) and confirmed on a third (L40S 0.0225; one question, img_sign448/word, sets it).
- regression, same GPU as the reference (opt-in, --regression DIR): against a bf16 run made on
  the same architecture, at 0.005 / 0.001. It catches code changes, not hardware: two bf16 runs
  on different GPUs disagree by up to ~0.03 while each stays within the correctness bar.

    python scripts/parity_http.py http://127.0.0.1:8000 runs/http-bf16 \\
        [--regression fixtures/clef-flash/bfloat16] [--also DIR] [--token T]

Exit 0 when the correctness gate passes and, if requested, the regression gate passes too.
`--also` prints informational reports that never affect the exit code. The run's manifest
records the worker's engine info (GPU, dtype, versions) from the first debug block.
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


def collect(
    lines: list[dict[str, Any]],
    run_dir: Path,
    poster,
    meta: dict[str, Any] | None = None,
    to_record=None,
) -> dict[str, int]:
    """Post every fixture line through `poster(body) -> (status, reply)` and write the records.
    If `meta` is given, the first successful reply's engine info is stored in meta["engine"].
    `to_record(line, status, reply)` turns a reply into a record; the default reads this
    worker's debug block (scripts/parity_sglang.py passes one for SGLang's /v1/systemone)."""
    to_record = to_record or record_from_reply
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    for i, line in enumerate(lines, 1):
        t0 = time.perf_counter()
        status, reply = poster(wire_body(line))
        if meta is not None and status == 200 and "engine" not in meta:
            meta["engine"] = (reply.get("debug") or {}).get("engine")
        rec = to_record(line, status, reply)
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


def wait_ready(
    base: str, token: str | None, timeout_s: float, every_s: float = 30.0, poll_s: float = 3.0
) -> None:
    """Poll /ping until 200. Prints a progress line to stderr every `every_s` seconds: a cold
    worker can take minutes (download + load), and a silent terminal reads as a hang."""
    t0 = time.time()
    last_report = t0
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"[wait] polling {base.rstrip('/')}/ping until the worker is ready", file=sys.stderr)
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
        now = time.time()
        if now - t0 > timeout_s:
            raise SystemExit(f"worker not ready after {timeout_s:.0f}s ({state})")
        if now - last_report >= every_s:
            label = " (initializing)" if state == "status 204" else ""
            print(f"[wait] {now - t0:.0f}s: {state}{label}", file=sys.stderr, flush=True)
            last_report = now
        time.sleep(poll_s)


CORRECTNESS = {"max_dp": 0.025, "mean_dp": 0.0025}
REGRESSION = {"max_dp": 0.005, "mean_dp": 0.001}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", help="worker base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "fixtures/clef-flash/float32",
        help="correctness gate reference (default: the fp32 truth)",
    )
    ap.add_argument("--max-dp", type=float, default=CORRECTNESS["max_dp"])
    ap.add_argument("--mean-dp", type=float, default=CORRECTNESS["mean_dp"])
    ap.add_argument(
        "--regression",
        type=Path,
        default=None,
        help="same-GPU regression gate reference, e.g. fixtures/clef-flash/bfloat16 on Hopper",
    )
    ap.add_argument("--regression-max-dp", type=float, default=REGRESSION["max_dp"])
    ap.add_argument("--regression-mean-dp", type=float, default=REGRESSION["mean_dp"])
    ap.add_argument("--also", type=Path, action="append", default=[], help="informational refs")
    ap.add_argument("--requests", nargs="+", default=[str(ROOT / "fixtures/requests/*.jsonl")])
    ap.add_argument("--token", default=None, help="Runpod API key for an endpoint URL")
    ap.add_argument("--timeout", type=float, default=330.0, help="per request, seconds")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--json", type=Path, help="write the correctness report as JSON")
    return ap


def verdict(correct_passed: bool, regression_passed: bool | None) -> bool:
    """Correctness always gates; regression gates only when it was requested."""
    return correct_passed and regression_passed is not False


def main() -> None:
    a = build_parser().parse_args()
    wait_ready(a.base, a.token, a.ready_timeout)
    lines = load_requests(a.requests)
    meta: dict[str, Any] = {}
    counts = collect(lines, a.run_dir, lambda body: post(a.base, body, a.token, a.timeout), meta)
    (a.run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "base": a.base,
                "reference": str(a.reference),
                "regression": str(a.regression) if a.regression else None,
                "engine": meta.get("engine"),
                "counts": counts,
                "requests": a.requests,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        )
    )

    print("## Correctness gate (any GPU)\n")
    report = compare(a.reference, a.run_dir, max_dp=a.max_dp, mean_dp=a.mean_dp)
    print(render(report))
    if a.json:
        a.json.write_text(json.dumps(report.__dict__, default=lambda o: o.__dict__, indent=1))
    reg_passed = None
    if a.regression:
        print("\n## Regression gate (same GPU as the reference)\n")
        reg = compare(
            a.regression, a.run_dir, max_dp=a.regression_max_dp, mean_dp=a.regression_mean_dp
        )
        print(render(reg))
        reg_passed = reg.passed
    for ref in a.also:
        print(f"\n--- informational: vs {ref} (not gated) ---\n")
        print(render(compare(ref, a.run_dir)))
    sys.exit(0 if verdict(report.passed, reg_passed) else 1)


if __name__ == "__main__":
    main()
