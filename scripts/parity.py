#!/usr/bin/env python3
"""Parity gate: compare a run directory against the oracle directory, record by record.

Both directories hold one JSON per record id (see deciders/clef/oracle.py for the schema).
Two gates, reported in this order:

1. span gate     the run encoded the request to the same token ids and the same question /
                 option spans as the oracle (sha256 of each). Encoding drift is the silent
                 failure mode, so it is reported before any numbers. A run file without an
                 `encoding` block (e.g. converted hosted responses) skips this gate: "n/a".
2. numeric gate  per question: argmax equal where the oracle's top-2 margin > NEAR_TIE
                 (near-ties are listed, not failed); max |Δp| ≤ MAX_DP. Over the whole run:
                 mean over records of each record's mean |Δp| ≤ MEAN_DP.

Error records (oracle has `error`, e.g. schema_overflow) pass iff the run also errored.

Usage:
    python scripts/parity.py ORACLE_DIR RUN_DIR [--max-dp 0.02] [--mean-dp 0.002] [--near-tie 0.02]
                                                [--json report.json]
Exit 0 on pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_DP = 0.02
MEAN_DP = 0.002
NEAR_TIE = 0.02


@dataclass
class QuestionResult:
    record: str
    question: str
    argmax_ok: bool | None  # None = near-tie, not judged
    max_dp: float
    near_tie: bool
    oracle_top: str
    run_top: str


@dataclass
class RecordResult:
    record: str
    status: str  # PASS | FAIL | SKIP
    reasons: list[str] = field(default_factory=list)
    span: str = "n/a"  # ok | mismatch | n/a
    mean_dp: float | None = None
    questions: list[QuestionResult] = field(default_factory=list)


@dataclass
class Report:
    oracle_dir: str
    run_dir: str
    max_dp: float
    mean_dp: float
    near_tie: float
    records: list[RecordResult]
    overall_mean_dp: float | None
    passed: bool

    def summary_lines(self) -> list[str]:
        n = len(self.records)
        fails = [r for r in self.records if r.status == "FAIL"]
        spans = sum(1 for r in self.records if r.span == "mismatch")
        ties = sum(1 for r in self.records for q in r.questions if q.near_tie)
        worst = max((q.max_dp for r in self.records for q in r.questions), default=0.0)
        lines = [
            f"| records | {n} |",
            f"| failed | {len(fails)} |",
            f"| span mismatches | {spans} |",
            f"| max \\|Δp\\| (worst question) | {worst:.4f} (bar {self.max_dp}) |",
            f"| mean of record means | "
            f"{'n/a' if self.overall_mean_dp is None else f'{self.overall_mean_dp:.5f}'} "
            f"(bar {self.mean_dp}) |",
            f"| near-ties (not judged) | {ties} |",
            f"| result | {'PASS' if self.passed else 'FAIL'} |",
        ]
        return ["| metric | value |", "|---|---|", *lines]


def _load_dir(path: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(path.glob("*.json")):
        if p.name == "manifest.json":
            continue
        rec = json.loads(p.read_text())
        out[rec["id"]] = rec
    return out


def _argmax(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def _margin(xs: list[float]) -> float:
    s = sorted(xs, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def compare_record(o: dict, r: dict | None, max_dp: float, near_tie: float) -> RecordResult:
    res = RecordResult(record=o["id"], status="PASS")
    if r is None:
        res.status = "FAIL"
        res.reasons.append("missing in run")
        return res

    if o.get("error"):
        if r.get("error"):
            res.reasons.append("both errored")
        else:
            res.status = "FAIL"
            res.reasons.append(f"oracle errored ({o['error'][:60]}), run did not")
        return res
    if r.get("error"):
        res.status = "FAIL"
        res.reasons.append(f"run errored: {r['error'][:80]}")
        return res

    oe, re_ = o.get("encoding"), r.get("encoding")
    if oe and re_:
        if oe["ids_sha"] == re_["ids_sha"] and oe["spans_sha"] == re_["spans_sha"]:
            res.span = "ok"
        else:
            res.span = "mismatch"
            res.status = "FAIL"
            which = []
            if oe["ids_sha"] != re_["ids_sha"]:
                which.append(f"ids ({oe['input_tokens']} vs {re_.get('input_tokens')} tokens)")
            if oe["spans_sha"] != re_["spans_sha"]:
                which.append("spans")
            res.reasons.append("span gate: " + ", ".join(which))

    oq = {q["id"]: q for q in o["questions"]}
    rq = {q["id"]: q for q in r.get("questions", [])}
    dps: list[float] = []
    for qid, q in oq.items():
        if qid not in rq:
            res.status = "FAIL"
            res.reasons.append(f"{qid}: missing in run")
            continue
        op, rp = q["probs"], rq[qid]["probs"]
        if len(op) != len(rp):
            res.status = "FAIL"
            res.reasons.append(f"{qid}: {len(op)} vs {len(rp)} options")
            continue
        d = [abs(a - b) for a, b in zip(op, rp, strict=True)]
        dps.extend(d)
        mdp = max(d)
        tie = _margin(op) <= near_tie
        oi, ri = _argmax(op), _argmax(rp)
        qr = QuestionResult(
            record=o["id"],
            question=qid,
            argmax_ok=None if tie else oi == ri,
            max_dp=mdp,
            near_tie=tie,
            oracle_top=q["option_ids"][oi],
            run_top=q["option_ids"][ri],
        )
        res.questions.append(qr)
        if qr.argmax_ok is False:
            res.status = "FAIL"
            res.reasons.append(
                f"{qid}: argmax {qr.oracle_top} → {qr.run_top} (margin {_margin(op):.3f})"
            )
        if mdp > max_dp:
            res.status = "FAIL"
            res.reasons.append(f"{qid}: max |Δp| {mdp:.4f} > {max_dp}")
    res.mean_dp = sum(dps) / len(dps) if dps else None
    return res


def compare(
    oracle_dir: Path,
    run_dir: Path,
    max_dp: float = MAX_DP,
    mean_dp: float = MEAN_DP,
    near_tie: float = NEAR_TIE,
) -> Report:
    oracle, run = _load_dir(oracle_dir), _load_dir(run_dir)
    records = [compare_record(o, run.get(rid), max_dp, near_tie) for rid, o in oracle.items()]
    means = [r.mean_dp for r in records if r.mean_dp is not None]
    overall = sum(means) / len(means) if means else None
    passed = all(r.status != "FAIL" for r in records) and (overall is None or overall <= mean_dp)
    return Report(
        oracle_dir=str(oracle_dir),
        run_dir=str(run_dir),
        max_dp=max_dp,
        mean_dp=mean_dp,
        near_tie=near_tie,
        records=records,
        overall_mean_dp=overall,
        passed=passed,
    )


def render(report: Report) -> str:
    lines = [f"# Parity: {report.run_dir} vs {report.oracle_dir}", "", *report.summary_lines(), ""]
    fails = [r for r in report.records if r.status == "FAIL"]
    if fails:
        lines += ["## Failures", ""]
        for r in fails:
            lines.append(f"- **{r.record}** (span {r.span}): " + "; ".join(r.reasons))
        lines.append("")
    ties = [q for r in report.records for q in r.questions if q.near_tie]
    if ties:
        lines += ["## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)", ""]
        for q in ties:
            lines.append(
                f"- {q.record}/{q.question}: oracle {q.oracle_top}, run {q.run_top}, "
                f"max |Δp| {q.max_dp:.4f}"
            )
        lines.append("")
    if report.overall_mean_dp is not None and report.overall_mean_dp > report.mean_dp:
        lines.append(
            f"**Mean-of-means {report.overall_mean_dp:.5f} exceeds bar {report.mean_dp}.**"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-dp", type=float, default=MAX_DP)
    ap.add_argument("--mean-dp", type=float, default=MEAN_DP)
    ap.add_argument("--near-tie", type=float, default=NEAR_TIE)
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    a = ap.parse_args()
    report = compare(a.oracle_dir, a.run_dir, a.max_dp, a.mean_dp, a.near_tie)
    print(render(report))
    if a.json:
        a.json.write_text(json.dumps(asdict(report), indent=1))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
