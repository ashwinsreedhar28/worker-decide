#!/usr/bin/env python3
"""Closed-loop load test of a SystemOne server: latency percentiles and throughput.

Engine-agnostic: our worker (TRANSPORT=http, which aliases /v1/systemone to /v1/decide) and
SGLang both answer POST /v1/systemone with the same body, so the same requests go to both.
Each cell is one (workload, concurrency): `concurrency` threads each send a request, wait for
the answer, and send the next, until `requests` answers are in. Warm-up requests (default one
per thread) are sent first and not counted.

    python scripts/loadtest.py http://127.0.0.1:8000 runs/load-ours-a40 \\
        --workload ticket=card_trio --workload article=long_2k --concurrency 1 8 32

Workloads name fixture records (any id in fixtures/requests/*.jsonl). Defaults: ticket =
card_trio (3 questions, 300 tokens), pulse = the first Pulse record (1 question, ~850 tokens),
article = long_2k (2 questions, ~2k tokens).

Percentiles are nearest-rank over the counted requests, so p99 of 128 requests is the 127th
smallest. Throughput is answered requests / wall time of the counted phase. Writes load.json
and load.md to OUT_DIR.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from parity_http import wire_body  # noqa: E402

DEFAULT_WORKLOADS = {"ticket": "card_trio", "pulse": None, "article": "long_2k"}


def percentile(sorted_values: list[float], p: float) -> float:
    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def summarize(latencies_s: list[float], errors: int, wall_s: float) -> dict[str, Any]:
    v = sorted(latencies_s)
    if not v:
        return {"n": 0, "errors": errors, "wall_s": round(wall_s, 3), "req_per_s": 0.0}
    ms = [x * 1000 for x in v]
    return {
        "n": len(v),
        "errors": errors,
        "wall_s": round(wall_s, 3),
        "req_per_s": round(len(v) / wall_s, 3) if wall_s > 0 else None,
        "p50_ms": round(percentile(ms, 0.50), 1),
        "p90_ms": round(percentile(ms, 0.90), 1),
        "p99_ms": round(percentile(ms, 0.99), 1),
        "mean_ms": round(statistics.fmean(ms), 1),
        "min_ms": round(ms[0], 1),
        "max_ms": round(ms[-1], 1),
    }


def run_cell(poster, body: dict, concurrency: int, requests: int, warmup: int,
             clock=time.perf_counter) -> dict[str, Any]:  # fmt: skip
    """`poster(body) -> status` is called from `concurrency` threads."""
    lock = threading.Lock()
    state = {"issued": 0}
    latencies: list[float] = []
    errors = {"n": 0}

    def take(limit: int) -> bool:
        with lock:
            if state["issued"] >= limit:
                return False
            state["issued"] += 1
            return True

    def worker(limit: int, record: bool) -> None:
        while take(limit):
            t0 = clock()
            try:
                ok = poster(body) == 200
            except Exception:  # noqa: BLE001 — a dropped connection is an error, not a crash
                ok = False
            dt = clock() - t0
            with lock:
                if not record:
                    continue
                if ok:
                    latencies.append(dt)
                else:
                    errors["n"] += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda _: worker(warmup, False), range(concurrency)))
        state["issued"] = 0
        t0 = clock()
        list(pool.map(lambda _: worker(requests, True), range(concurrency)))
        wall = clock() - t0
    return summarize(latencies, errors["n"], wall)


def load_line(rid: str | None, requests_glob: str) -> dict:
    """A fixture line by id; None picks the first Pulse record."""
    for path in sorted(ROOT.glob(requests_glob)):
        for raw in path.read_text().splitlines():
            if not raw.strip():
                continue
            line = json.loads(raw)
            if rid is None and line["id"].startswith("pulse-") or line["id"] == rid:
                return line
    raise SystemExit(f"no fixture record {rid!r}")


def http_poster(base: str, path: str, model: str, timeout: float):
    url = base.rstrip("/") + path

    def poster(body: dict) -> int:
        req = urllib.request.Request(
            url,
            data=json.dumps({**body, "model": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read()
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    return poster


def render(label: str, cells: list[dict[str, Any]]) -> str:
    out = [
        f"## Load: {label}",
        "",
        "| workload | tokens | concurrency | n | errors | req/s | p50 ms | p90 ms | p99 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        s = c["stats"]
        out.append(
            f"| {c['workload']} | {c.get('input_tokens', '?')} | {c['concurrency']} | {s['n']} "
            f"| {s['errors']} | {s['req_per_s']} | {s.get('p50_ms', '-')} | "
            f"{s.get('p90_ms', '-')} | {s.get('p99_ms', '-')} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--path", default="/v1/systemone")
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--workload", action="append", default=[], help="name=fixture_id")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--requests", type=int, default=128, help="counted requests per cell")
    ap.add_argument("--warmup", type=int, default=None, help="default: one per thread")
    ap.add_argument("--timeout", type=float, default=330.0)
    ap.add_argument("--requests-glob", default="fixtures/requests/*.jsonl")
    a = ap.parse_args()

    workloads = dict(DEFAULT_WORKLOADS)
    if a.workload:
        workloads = dict(w.split("=", 1) for w in a.workload)
    poster = http_poster(a.base, a.path, a.model, a.timeout)
    oracle = ROOT / "fixtures/clef-flash/float32"
    cells = []
    for name, rid in workloads.items():
        line = load_line(rid, a.requests_glob)
        body = wire_body(line)
        o = oracle / f"{line['id']}.json"
        tokens = json.loads(o.read_text())["encoding"]["input_tokens"] if o.exists() else None
        for c in a.concurrency:
            print(f"[load] {name} ({line['id']}) c={c} ...", file=sys.stderr, flush=True)
            stats = run_cell(poster, body, c, a.requests, c if a.warmup is None else a.warmup)
            cells.append(
                {
                    "workload": name,
                    "record": line["id"],
                    "input_tokens": tokens,
                    "concurrency": c,
                    "stats": stats,
                }  # fmt: skip
            )
            print(f"[load]   {stats}", file=sys.stderr, flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    label = a.label or a.out_dir.name
    meta = {"label": label, "base": a.base, "path": a.path, "requests": a.requests,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    (a.out_dir / "load.json").write_text(json.dumps({**meta, "cells": cells}, indent=1))
    text = render(label, cells)
    (a.out_dir / "load.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
