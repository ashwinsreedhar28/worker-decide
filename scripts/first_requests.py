#!/usr/bin/env python3
"""First-request cost per input shape: what JIT kernels (Triton compile, autotune) add.

Sends each workload `repeat` times in sequence to a freshly started server (started with
WARMUP=0) and records every latency. first_minus_steady_ms is the first request minus the
median of the rest. With an empty kernel cache it holds compilation and autotuning for shapes
not seen yet, plus first-request costs no kernel cache removes (CUDA context and cuBLAS setup,
allocator growth). So the saving from persisting the kernel cache (TRITON_CACHE_DIR on a
volume, the worker-vllm #351 pattern) is first_minus_steady on an empty-cache run minus
first_minus_steady on a warm-cache run, not either number alone.

A request that fails at the connection level (refused, reset, timed out) is recorded as
status 0 with its elapsed time; main() exits non-zero if any status is not 200.

    python scripts/first_requests.py http://127.0.0.1:8000 runs/first-fla-h100 \\
        [--workload ticket=card_trio ...] [--repeat 3]
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from loadtest import DEFAULT_WORKLOADS, http_poster, load_line  # noqa: E402
from parity_http import wire_body  # noqa: E402


def measure(poster, bodies: dict[str, dict], repeat: int, clock=time.perf_counter) -> dict:
    """{workload: {"ms": [...], "first_minus_steady_ms": ..., "statuses": [...]}}, in order."""
    out: dict[str, Any] = {}
    for name, body in bodies.items():
        ms, statuses = [], []
        for _ in range(repeat):
            t0 = clock()
            try:
                status = poster(body)
            except (OSError, http.client.HTTPException):  # URLError, timeouts, resets
                status = 0
            statuses.append(status)
            ms.append(round((clock() - t0) * 1000, 1))
        steady = statistics.median(ms[1:]) if len(ms) > 1 else None
        out[name] = {
            "ms": ms,
            "statuses": statuses,
            "first_minus_steady_ms": None if steady is None else round(ms[0] - steady, 1),
        }
    return out


def render(label: str, result: dict) -> str:
    lines = [f"## First requests: {label}", "", "| workload | requests (ms) | first - steady |",
             "|---|---|---|"]  # fmt: skip
    for name, r in result.items():
        diff = r["first_minus_steady_ms"]
        lines.append(f"| {name} | {', '.join(str(x) for x in r['ms'])} | "
                     f"{'-' if diff is None else f'{diff} ms'} |")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workload", action="append", default=[], help="name=fixture_id")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--path", default="/v1/systemone")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    workloads = dict(w.split("=", 1) for w in a.workload) if a.workload else DEFAULT_WORKLOADS
    bodies = {n: wire_body(load_line(rid, "fixtures/requests/*.jsonl")) for n, rid in
              workloads.items()}  # fmt: skip
    result = measure(http_poster(a.base, a.path, a.model, 600.0), bodies, a.repeat)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "first_requests.json").write_text(json.dumps(result, indent=1))
    text = render(a.label or a.out_dir.name, result)
    (a.out_dir / "first_requests.md").write_text(text + "\n")
    print(text)
    if any(s != 200 for r in result.values() for s in r["statuses"]):
        raise SystemExit("some requests were not answered 200")


if __name__ == "__main__":
    main()
