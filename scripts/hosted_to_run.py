#!/usr/bin/env python3
"""Convert hosted Workers AI responses (fixtures/hosted/*.jsonl) into a parity run directory.

Hosted lines carry probabilities (4 dp) for one question per record and no token ids, so the
resulting run has no `encoding` block: parity.py reports the span gate as n/a and judges the
numbers only. Option ids are taken from the oracle record so the vectors line up: a list is
accepted as-is when its length matches (score questions: index = score), a dict (choice and
noul responses) is re-ordered by the oracle's option ids. Anything else is skipped with a
message. `request_sha` is copied from the oracle because hosted rows do not carry one, so
parity's request gate is vacuous for these runs.

    python scripts/hosted_to_run.py HOSTED.jsonl RUN_DIR ORACLE_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _align(probabilities, option_ids: list[str]) -> list[float] | None:
    if isinstance(probabilities, dict):
        if set(probabilities) != set(option_ids):
            return None
        return [float(probabilities[o]) for o in option_ids]
    if isinstance(probabilities, list) and len(probabilities) == len(option_ids):
        return [float(x) for x in probabilities]
    return None


def convert(hosted: Path, run_dir: Path, oracle_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in hosted.read_text().splitlines():
        if not line.strip():
            continue
        h = json.loads(line)
        oracle_path = oracle_dir / f"{h['id']}.json"
        if not oracle_path.exists():
            print(f"skip {h['id']}: no oracle record", file=sys.stderr)
            continue
        o = json.loads(oracle_path.read_text())
        oq = next((q for q in o["questions"] if q["id"] == h["question"]), None)
        if oq is None:
            print(
                f"skip {h['id']}: oracle has no answer for {h['question']}"
                f"{' (' + o['error'][:50] + ')' if o.get('error') else ''}",
                file=sys.stderr,
            )
            continue
        probs = _align(h["probabilities"], oq["option_ids"])
        if probs is None:
            print(f"skip {h['id']}: probabilities do not match oracle option ids", file=sys.stderr)
            continue
        rec = {
            "id": h["id"],
            "request_sha": o["request_sha"],
            "questions": [
                {
                    "id": h["question"],
                    "type": oq["type"],
                    "option_ids": oq["option_ids"],
                    "probs": probs,
                }
            ],
            "response": {
                "hosted": {k: h[k] for k in ("confidence", "expected", "prompt_tokens", "wall_ms")}
            },
            "error": None,
            "source": {
                "model": h.get("model"),
                "batch_id": h.get("batch_id"),
                "ts_utc": h.get("ts_utc"),
            },
        }
        (run_dir / f"{h['id']}.json").write_text(json.dumps(rec))
        n += 1
    return n


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    n = convert(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"converted {n} hosted responses → {sys.argv[2]}")


if __name__ == "__main__":
    main()
