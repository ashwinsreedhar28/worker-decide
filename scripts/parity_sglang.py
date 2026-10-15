#!/usr/bin/env python3
"""Run the fixture requests through an SGLang server's /v1/systemone and gate the answers.

Milestone 4: does SGLang compute Clef-flash? SGLang serves Clef's joint schema head from the
nightly that merged sgl-project/sglang#42721 (0.5.22.dev20261009+g37ae292e6f, image
lmsysorg/sglang:dev-clef); release 0.5.21 has the /v1/systemone route but scores generic models
through vocabulary label tokens, not Clef's head. This script does not assume which one it is
talking to: it records the server's version and lets the numbers decide.

What SGLang returns and how it is mapped to the oracle schema (verified against its source,
python/sglang/srt/entrypoints/systemone/{serving,joint_schema}.py at main 5b12ec1):

- choice: `probabilities` keyed by option id, full precision; the head scores options in
  sorted-id order, as encode_record does, so the record's option_ids are the sorted ids.
- noul: only `noul` = p(true); the head scores (true, false), so probs = [p, 1 - p].
- score: `probabilities` keyed "0".."n-1".
- `confidence` uses SGLang's System One adapter formulas, not the reference's max p, so it is
  not compared. `usage.input_tokens` is kept: there is no encoding block to run the span gate
  on, so token counts against the oracle's `encoding.input_tokens` stand in for it (equal
  counts on all 87 records are strong, not complete, evidence that the prompt is the same).

    python scripts/parity_sglang.py http://127.0.0.1:30000 runs/sglang-a40 \\
        [--also runs/http-a40] [--model Cloudflare/clef-flash]

Exit 0 when the correctness gate (vs the fp32 truth, 0.025 / 0.0025) passes.
"""

from __future__ import annotations

import argparse
import json
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
from parity_http import CORRECTNESS, collect  # noqa: E402

from deciders.clef.engine import request_sha  # noqa: E402
from deciders.clef.oracle import load_requests  # noqa: E402


def option_ids(question: dict[str, Any]) -> list[str]:
    """The order the joint schema head scores a question's options in (encode_record's)."""
    kind = question["type"]
    if kind == "noul":
        return ["true", "false"]
    if kind == "choice":
        return sorted(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    raise ValueError(f"unknown question type {kind!r}")


def probs_from_answer(question: dict[str, Any], answer: dict[str, Any]) -> list[float]:
    ids = option_ids(question)
    if question["type"] == "noul":
        p = float(answer["noul"])
        return [p, 1.0 - p]
    by_id = answer.get("probabilities") or {}
    if set(by_id) != set(ids):
        raise ValueError(f"answer options {sorted(by_id)} are not the request's {ids}")
    return [float(by_id[i]) for i in ids]


def record_from_systemone(line: dict[str, Any], status: int, reply: dict[str, Any]) -> dict:
    """Oracle-schema record from one /v1/systemone reply. No encoding block (span gate n/a)."""
    rec: dict[str, Any] = {
        "id": line["id"],
        "request_sha": request_sha(line["request"]),
        "encoding": None,
        "questions": [],
        "response": None,
        "usage_input_tokens": None,
        "error": None,
        "timing_ms": {},
    }
    if status != 200:
        detail = reply.get("error") or reply.get("detail") or reply.get("message") or reply
        rec["error"] = f"http {status}: {json.dumps(detail)[:200]}"
        return rec
    answers = reply.get("answers") or {}
    try:
        for qid, q in line["request"]["questions"].items():
            if qid not in answers:
                raise ValueError(f"no answer for question {qid!r}")
            rec["questions"].append(
                {
                    "id": qid,
                    "type": q["type"],
                    "option_ids": option_ids(q),
                    "probs": probs_from_answer(q, answers[qid]),
                }
            )
    except (ValueError, KeyError, TypeError) as e:
        rec["questions"], rec["error"] = [], f"unreadable reply: {e}"[:200]
        return rec
    rec["response"] = reply
    rec["usage_input_tokens"] = (reply.get("usage") or {}).get("input_tokens")
    return rec


def token_report(oracle_dir: Path, run_dir: Path) -> dict[str, Any]:
    """usage.input_tokens vs the oracle's encoding.input_tokens, per record that both answered."""
    equal, diffs, missing = 0, [], 0
    for p in sorted(oracle_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        o = json.loads(p.read_text())
        rp = run_dir / p.name
        if o.get("error") or not o.get("encoding"):
            continue
        r = json.loads(rp.read_text()) if rp.exists() else None
        if not r or r.get("error") or r.get("usage_input_tokens") is None:
            missing += 1
            continue
        want, got = o["encoding"]["input_tokens"], r["usage_input_tokens"]
        if want == got:
            equal += 1
        else:
            diffs.append({"id": o["id"], "oracle": want, "run": got})
    return {"equal": equal, "different": diffs, "missing": missing}


def render_tokens(t: dict[str, Any]) -> str:
    lines = [
        "| token counts vs oracle | value |",
        "|---|---|",
        f"| equal | {t['equal']} |",
        f"| different | {len(t['different'])} |",
        f"| no count (errored or absent) | {t['missing']} |",
    ]
    for d in t["different"][:20]:
        lines.append(f"| {d['id']} | oracle {d['oracle']} vs run {d['run']} |")
    return "\n".join(lines)


def post(base: str, body: dict[str, Any], timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw[:300].decode("utf-8", "replace")}


def get_json(base: str, path: str) -> dict | None:
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def wait_ready(base: str, timeout_s: float, every_s: float = 30.0, poll_s: float = 3.0) -> None:
    """SGLang's /health is 503 while starting and 200 once it serves."""
    t0 = last = time.time()
    while True:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=10) as r:
                if r.status == 200:
                    return
                state = f"status {r.status}"
        except urllib.error.HTTPError as e:
            state = f"status {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            state = str(e)
        now = time.time()
        if now - t0 > timeout_s:
            raise SystemExit(f"SGLang not ready after {timeout_s:.0f}s ({state})")
        if now - last >= every_s:
            print(f"[wait] {now - t0:.0f}s: {state}", file=sys.stderr, flush=True)
            last = now
        time.sleep(poll_s)


def with_model(model: str):
    def poster_body(body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "model": model}

    return poster_body


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", help="SGLang base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--reference", type=Path, default=ROOT / "fixtures/clef-flash/float32")
    ap.add_argument("--max-dp", type=float, default=CORRECTNESS["max_dp"])
    ap.add_argument("--mean-dp", type=float, default=CORRECTNESS["mean_dp"])
    ap.add_argument("--also", type=Path, action="append", default=[], help="informational refs")
    ap.add_argument("--requests", nargs="+", default=[str(ROOT / "fixtures/requests/*.jsonl")])
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--timeout", type=float, default=330.0)
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--json", type=Path, help="write the correctness report as JSON")
    a = ap.parse_args()

    wait_ready(a.base, a.ready_timeout)
    lines = load_requests(a.requests)
    body_for = with_model(a.model)
    counts = collect(
        lines,
        a.run_dir,
        lambda body: post(a.base, body_for(body), a.timeout),
        to_record=record_from_systemone,
    )
    tokens = token_report(a.reference, a.run_dir)
    (a.run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "base": a.base,
                "engine": "sglang",
                "server_info": get_json(a.base, "/get_server_info"),
                "model_info": get_json(a.base, "/get_model_info"),
                "reference": str(a.reference),
                "counts": counts,
                "tokens": tokens,
                "requests": a.requests,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
            default=str,
        )
    )
    print("## Correctness gate (vs fp32 truth)\n")
    report = compare(a.reference, a.run_dir, max_dp=a.max_dp, mean_dp=a.mean_dp)
    print(render(report))
    if a.json:
        a.json.write_text(json.dumps(report.__dict__, default=lambda o: o.__dict__, indent=1))
    print("\n## Prompt length (stand-in for the span gate)\n")
    print(render_tokens(tokens))
    for ref in a.also:
        print(f"\n--- informational: vs {ref} (not gated) ---\n")
        print(render(compare(ref, a.run_dir, max_dp=a.max_dp, mean_dp=a.mean_dp)))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
