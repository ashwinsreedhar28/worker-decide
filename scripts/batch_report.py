#!/usr/bin/env python3
"""Did batching actually happen? Batch sizes behind a parity run, and the batcher's own stats.

A correctness gate run against a batching worker passes just as well if every batch failed and
each request was quietly retried alone (worker/batcher.py's failure isolation). This reads the
`batch` block parity_http keeps per record and the batcher stats from the worker's GET /, and
fails unless some records were answered in batches of two or more and no batch was retried.

    python scripts/batch_report.py runs/http-batched-a40 [--root runs/worker-tier1-root.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def sizes(run_dir: Path) -> Counter:
    """{batch size: records answered in a batch that size}; None = answered without a batch
    block (image requests, errors, or a worker that is not batching)."""
    c: Counter = Counter()
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        rec = json.loads(f.read_text())
        if "id" not in rec:
            continue
        c[(rec.get("batch") or {}).get("size")] += 1
    return c


def verdict(counts: Counter, stats: dict[str, Any] | None) -> tuple[bool, list[str]]:
    reasons = []
    batched = sum(n for s, n in counts.items() if s is not None and s > 1)
    if batched == 0:
        reasons.append("no record was answered in a batch of 2 or more")
    if stats:
        if stats.get("retried_alone"):
            reasons.append(
                f"{stats['retried_alone']} requests were retried alone after a failed batch"
            )
        if stats.get("stopped"):
            reasons.append(f"the batcher stopped: {stats['stopped']}")
    return not reasons, reasons


def render(counts: Counter, stats: dict[str, Any] | None, ok: bool, reasons: list[str]) -> str:
    total = sum(counts.values())
    batched = sum(n for s, n in counts.items() if s is not None and s > 1)
    rows = ["| batch size | records |", "|---|---|"]
    for s in sorted(counts, key=lambda x: (x is None, x or 0)):
        rows.append(f"| {'none' if s is None else s} | {counts[s]} |")
    lines = [f"## Batching: {batched} of {total} records answered in batches of 2+", "", *rows]
    if stats:
        lines += ["", "Batcher stats: " + ", ".join(f"{k}={v}" for k, v in stats.items())]
    lines += ["", "result: " + ("PASS" if ok else "FAIL (" + "; ".join(reasons) + ")")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--root", type=Path, help="the worker's GET / saved as JSON")
    a = ap.parse_args(argv)
    counts = sizes(a.run_dir)
    stats = None
    if a.root and a.root.exists():
        stats = json.loads(a.root.read_text()).get("batcher")
    ok, reasons = verdict(counts, stats)
    print(render(counts, stats, ok, reasons))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
