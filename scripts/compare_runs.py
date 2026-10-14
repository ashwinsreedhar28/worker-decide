#!/usr/bin/env python3
"""Compare two runs against a truth reference, and against each other, question by question.

The parity gate answers "is this run within tolerance of a reference". This answers the
question behind it when two runs disagree: is one of them wrong, or are both independent noisy
estimates of the truth that err in different directions? Built for cross-GPU bf16 runs (Oct 14:
an A40 failed a 0.005 gate against the H100 bf16 run while being closer to fp32 than the H100).

    python scripts/compare_runs.py fixtures/clef-flash/float32 \\
        fixtures/clef-flash/bfloat16 runs/http-a40 --labels H100-bf16,A40-bf16

Per question the metric is max |Δp| over options. "Clear" argmax flips are those where the
truth's top-2 margin exceeds --near-tie; flips inside it are expected noise. Records that
errored in any of the three, or questions missing from any, are left out and counted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def load(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """record id -> question id -> probs, for records without an error."""
    out: dict[str, dict[str, list[float]]] = {}
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        rec = json.loads(p.read_text())
        if rec.get("error"):
            continue
        out[rec["id"]] = {q["id"]: q["probs"] for q in rec.get("questions", [])}
    return out


def _argmax(p: list[float]) -> int:
    return max(range(len(p)), key=lambda i: p[i])


def _margin(p: list[float]) -> float:
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def _maxdp(x: list[float], y: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(x, y, strict=True))


@dataclass
class Row:
    record: str
    question: str
    a_truth: float
    b_truth: float
    a_b: float
    truth_margin: float
    flip_a: bool  # a's argmax differs from truth's
    flip_b: bool
    flip_ab: bool


@dataclass
class Comparison:
    rows: list[Row]
    skipped: int
    near_tie: float
    labels: tuple[str, str]
    worst: list[Row] = field(default_factory=list)

    def stats(self, attr: str) -> dict[str, float]:
        v = sorted(getattr(r, attr) for r in self.rows)
        n = len(v)
        return {
            "max": v[-1],
            "p99": v[min(n - 1, int(n * 0.99))],
            "p90": v[min(n - 1, int(n * 0.9))],
            "median": statistics.median(v),
            "mean": statistics.fmean(v),
        }

    def flips(self, attr: str) -> tuple[int, list[Row]]:
        fl = [r for r in self.rows if getattr(r, attr)]
        return len(fl), [r for r in fl if r.truth_margin > self.near_tie]

    @property
    def a_closer(self) -> int:
        return sum(r.a_truth < r.b_truth for r in self.rows)

    @property
    def b_closer(self) -> int:
        return sum(r.b_truth < r.a_truth for r in self.rows)


def compare(
    truth: Path,
    a: Path,
    b: Path,
    near_tie: float = 0.04,
    labels: tuple[str, str] = ("A", "B"),
    worst_n: int = 6,
) -> Comparison:
    T, A, B = load(truth), load(a), load(b)
    rows: list[Row] = []
    skipped = 0
    for rid, tq in T.items():
        for qid, tp in tq.items():
            ap, bp = A.get(rid, {}).get(qid), B.get(rid, {}).get(qid)
            if ap is None or bp is None or not (len(ap) == len(bp) == len(tp)):
                skipped += 1
                continue
            ti = _argmax(tp)
            rows.append(
                Row(
                    record=rid,
                    question=qid,
                    a_truth=_maxdp(ap, tp),
                    b_truth=_maxdp(bp, tp),
                    a_b=_maxdp(ap, bp),
                    truth_margin=_margin(tp),
                    flip_a=_argmax(ap) != ti,
                    flip_b=_argmax(bp) != ti,
                    flip_ab=_argmax(ap) != _argmax(bp),
                )
            )
    c = Comparison(rows=rows, skipped=skipped, near_tie=near_tie, labels=labels)
    c.worst = sorted(rows, key=lambda r: -r.a_b)[:worst_n]
    return c


def render(c: Comparison) -> str:
    la, lb = c.labels
    lines = [
        f"{len(c.rows)} questions compared ({c.skipped} skipped: errored or missing in a run)",
        "",
        "| per-question max \\|Δp\\| | max | p99 | p90 | median | mean |",
        "|---|---|---|---|---|---|",
    ]
    for attr, lab in (("a_truth", f"{la} vs truth"), ("b_truth", f"{lb} vs truth"),
                      ("a_b", f"{la} vs {lb}")):  # fmt: skip
        s = c.stats(attr)
        lines.append(
            f"| {lab} | {s['max']:.4f} | {s['p99']:.4f} | {s['p90']:.4f} | "
            f"{s['median']:.5f} | {s['mean']:.5f} |"
        )
    lines += ["", f"Argmax flips (clear = truth top-2 margin > {c.near_tie}):"]
    for attr, lab in (("flip_a", f"{la} vs truth"), ("flip_b", f"{lb} vs truth"),
                      ("flip_ab", f"{la} vs {lb}")):  # fmt: skip
        n, clear = c.flips(attr)
        lines.append(f"- {lab}: {n} total, {len(clear)} clear")
        lines += [
            f"  - CLEAR: {r.record}/{r.question} (margin {r.truth_margin:.3f})" for r in clear
        ]
    lines += [
        "",
        f"{la} closer to truth on {c.a_closer} questions, {lb} closer on {c.b_closer}.",
        "",
        f"Largest {la}-vs-{lb} gaps:",
    ]
    lines += [
        f"- {r.record}/{r.question}: {la}-{lb} {r.a_b:.4f}, {la}-truth {r.a_truth:.4f}, "
        f"{lb}-truth {r.b_truth:.4f}, truth margin {r.truth_margin:.3f}"
        for r in c.worst
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("truth", type=Path)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--near-tie", type=float, default=0.04)
    ap.add_argument("--labels", default="A,B", help="two comma-separated names")
    a = ap.parse_args()
    labels = tuple(a.labels.split(",", 1)) if "," in a.labels else ("A", "B")
    print(render(compare(a.truth, a.run_a, a.run_b, a.near_tie, labels)))


if __name__ == "__main__":
    main()
