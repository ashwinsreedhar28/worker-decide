#!/usr/bin/env python3
"""Sanity checks on a dry-run directory: token-length targets the fixtures were built to hit.

WARN, not fail: the targets are design intent, and the numbers get recorded either way.

    python scripts/check_dry_run.py runs/dry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOSTED_PULSE_TOKENS = 843  # Workers AI's usage.input_tokens for pulse-11714037 on Oct 13


def main(run_dir: Path) -> int:
    recs = {
        p.stem: json.loads(p.read_text())
        for p in run_dir.glob("*.json")
        if p.name != "manifest.json"
    }

    def tokens(rid: str) -> int | None:
        enc = recs.get(rid, {}).get("encoding")
        return enc["input_tokens"] if enc else None

    checks = [
        (
            "long_2k tokens in [1500, 3000]",
            1500 <= (tokens("long_2k") or 0) <= 3000,
            tokens("long_2k"),
        ),
        ("trunc_16k tokens == 16384", tokens("trunc_16k") == 16384, tokens("trunc_16k")),
        (
            "schema_overflow errored",
            bool(recs.get("schema_overflow", {}).get("error")),
            (recs.get("schema_overflow", {}).get("error") or "")[:60],
        ),
        (
            "img_board2048 tokens < 4000",
            (tokens("img_board2048") or 10**9) < 4000,
            tokens("img_board2048"),
        ),
        (
            f"pulse-11714037 tokens == hosted {HOSTED_PULSE_TOKENS}",
            tokens("pulse-11714037") == HOSTED_PULSE_TOKENS,
            tokens("pulse-11714037"),
        ),
        (
            "no unexpected errors",
            all(not r.get("error") for k, r in recs.items() if k != "schema_overflow"),
            None,
        ),
    ]
    warns = 0
    for name, ok, got in checks:
        print(("PASS " if ok else "WARN ") + name + ("" if ok else f"  (got {got})"))
        warns += not ok
    print(f"{len(recs)} records, {warns} warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
