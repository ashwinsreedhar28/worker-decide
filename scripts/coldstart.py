#!/usr/bin/env python3
"""Measure cold starts on a Runpod queue endpoint, split into where the time goes.

Each iteration waits until the endpoint has no running or initializing worker and no job in
flight, submits one job (cold), then one more right away (warm, same worker), and records:

- Runpod's own numbers from /status: delayTime (submission -> execution start, cold start
  included) and executionTime (the handler's run), plus the workerId;
- the worker's startup timeline from the debug block (seconds since its process started:
  snapshot_resolved, model_code_imported, weights_loaded, warmup_done, ready) and how the
  snapshot arrived (model_store | hf_cache | hf_download, bytes, seconds);
- a /health snapshot taken just before the cold submit.

"Scaled to zero" is not "no workers": Runpod's Idle state is "scaled down, waiting for
requests" and not billed, but an idle slot has typically already pulled the image onto a host,
so a start onto it skips the pull. The idle count recorded before each submit labels which kind
of cold start a sample was. A first start on a fresh host (image pull included) cannot be forced
from here; measure it on a new endpoint and label it as such.

    python scripts/coldstart.py ENDPOINT_ID runs/coldstart-disk-a40 --runs 10 --label disk-a40

Set the endpoint's idle timeout low (5 s) for the benchmark, or each iteration waits it out.

Each sample is classified (Oct 14, arm D: FlashBoot kept the loaded process for 8 of 10 jobs):

- cold: a new process started after the submit (platform estimate >= 0);
- resumed: the endpoint reported no running worker, yet the job ran in a process that was
  already up before the submit (platform estimate < 0, i.e. the worker's ready mark is older
  than Runpod's delay). FlashBoot does this. Reported separately, not mixed into cold stats;
- suspect warm: under SUSPECT_WARM_DELAY_S with no timeline to tell which, excluded.

--gap S waits S seconds after the endpoint goes cold before submitting, to measure how long
FlashBoot keeps a process resumable. Sweep it with one output dir per gap.
The API key comes from $RUNPOD_API_KEY. Writes run-NN.json per iteration and summary.json, and
prints a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunpodAPI:
    """The three queue-endpoint operations this script needs, over stdlib urllib."""

    def __init__(self, endpoint_id: str, token: str, base: str = "https://api.runpod.ai/v2"):
        self.url = f"{base}/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _call(self, path: str, body: dict | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=self.headers,
            method="POST" if body is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def health(self) -> dict[str, Any]:
        return self._call("/health")

    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        return self._call("/run", job)

    def status(self, job_id: str) -> dict[str, Any]:
        return self._call(f"/status/{job_id}")


def is_cold(health: dict[str, Any]) -> bool:
    """No worker running or starting, and nothing queued or in progress."""
    w, j = health.get("workers", {}), health.get("jobs", {})
    return (
        w.get("running", 0) == 0
        and w.get("initializing", 0) == 0
        and j.get("inQueue", 0) == 0
        and j.get("inProgress", 0) == 0
    )


def wait_until(predicate, poll, interval: float, timeout: float, sleep=time.sleep, clock=time.time):
    """Call poll() every `interval` s until predicate(result); return the last result."""
    t0 = clock()
    while True:
        result = poll()
        if predicate(result):
            return result
        if clock() - t0 > timeout:
            raise TimeoutError(f"condition not met after {timeout:.0f}s: {result}")
        sleep(interval)


def run_job(api, job, interval=1.0, timeout=1800.0, sleep=time.sleep, clock=time.time) -> dict:
    """Submit and poll to a terminal state; returns the final /status plus client wall time."""
    t0 = clock()
    submitted_unix = time.time()  # wall clock, comparable with the worker's origin_unix
    job_id = api.run(job)["id"]
    final = wait_until(lambda s: s.get("status") in TERMINAL, lambda: api.status(job_id),
                       interval, timeout, sleep, clock)  # fmt: skip
    return {**final, "client_wall_s": round(clock() - t0, 3), "submitted_unix": submitted_unix}


def segments(marks: dict[str, float], fetch_s: float | None) -> dict[str, float | None]:
    """Durations between consecutive timeline marks (the marks themselves are cumulative)."""

    def gap(a: str, b: str) -> float | None:
        return round(marks[b] - marks[a], 3) if a in marks and b in marks else None

    return {
        "startup_s": marks.get("engine_init"),  # process start -> engine init (Python, imports)
        "fetch_s": fetch_s,
        "import_s": gap("snapshot_resolved", "model_code_imported"),  # torch, transformers, jsm
        "weights_s": gap("model_code_imported", "weights_loaded"),
        "warmup_s": gap("weights_loaded", "warmup_done"),
    }


def extract(status: dict[str, Any]) -> dict[str, Any]:
    """The fields the benchmark keeps from one /status response."""
    out = status.get("output") or {}
    engine = (out.get("debug") or {}).get("engine") or {}
    timeline = engine.get("timeline") or {}
    marks, fetch = timeline.get("marks") or {}, timeline.get("fetch") or {}
    delay_s = (status.get("delayTime") or 0) / 1000
    ready_s = marks.get("ready")
    return {
        "status": status.get("status"),
        "worker_id": status.get("workerId"),
        "delay_s": round(delay_s, 3),
        "exec_s": round((status.get("executionTime") or 0) / 1000, 3),
        "client_wall_s": status.get("client_wall_s"),
        "gpu": engine.get("gpu"),
        "snapshot_source": fetch.get("source") or engine.get("snapshot_source"),
        "fetch_s": fetch.get("seconds"),
        # v2 (from image clef-<sha> after the Oct 14 fix): bytes written during the fetch,
        # counted on the disk and under HF_HOME, plus the resolved snapshot size. v1 only had
        # blob-dir growth, which saw ~130 KB for a 19 GB snapshot and is not a transfer measure.
        "fetch_metric": "v2" if "bytes_written" in fetch else ("v1" if fetch else None),
        "bytes_written": fetch.get("bytes_written"),
        "snapshot_bytes": fetch.get("snapshot_bytes"),
        "snapshot_dir_existed": fetch.get("snapshot_dir_existed"),
        "marks": marks,
        "segments": segments(marks, fetch.get("seconds")),
        "timeline_basis": timeline.get("basis"),
        "warmup_s": engine.get("warmup_s"),
        # submission -> process start, estimated as Runpod's delay minus the worker's own
        # process-start -> ready time: scheduling, image pull (if any), container create.
        "platform_est_s": round(delay_s - ready_s, 3) if ready_s is not None else None,
        # How long the process had been up when the job was submitted (worker clock vs this
        # machine's; NTP-level skew). Only images that report origin_unix (after Oct 14).
        "process_age_s": (
            round(status["submitted_unix"] - timeline["origin_unix"], 1)
            if status.get("submitted_unix") and timeline.get("origin_unix")
            else None
        ),
        "error": status.get("error"),
    }


SUSPECT_WARM_DELAY_S = 2.0  # a true cold start cannot reach execution this fast


def wait_cold(api, interval, timeout, consecutive=2, sleep=time.sleep, clock=time.time):
    """Wait for `consecutive` cold /health readings in a row: right as a job completes, /health
    may lag the worker's state, and one stale reading would turn a warm start into a "cold"
    sample."""
    streak = {"n": 0}

    def cold_streak(h):
        streak["n"] = streak["n"] + 1 if is_cold(h) else 0
        return streak["n"] >= consecutive

    return wait_until(cold_streak, api.health, interval, timeout, sleep, clock)


def classify(c: dict[str, Any]) -> str:
    """cold | resumed | suspect_warm | failed, from one extracted cold-job record."""
    if c.get("status") != "COMPLETED":
        return "failed"
    plat = c.get("platform_est_s")
    if plat is not None:
        return "resumed" if plat < 0 else "cold"
    return "suspect_warm" if c.get("delay_s", 0) < SUSPECT_WARM_DELAY_S else "cold"


def row_kind(r: dict[str, Any]) -> str:
    """Always re-derived from the cold record, so run files written before `kind` existed (whose
    suspect_warm flag lumped resumed processes in with warm starts) get the same treatment."""
    return classify(r["cold"])


def iteration(api, job, settle_timeout=900.0, interval=5.0, gap_s=0.0, sleep=time.sleep,
              clock=time.time):  # fmt: skip
    health = wait_cold(api, interval, settle_timeout, sleep=sleep, clock=clock)
    health_after_gap = None
    if gap_s > 0:
        sleep(gap_s)
        health_after_gap = api.health()  # recorded, not gated: a scale-up here is the platform's
    cold = run_job(api, job, sleep=sleep, clock=clock)
    warm = run_job(api, job, sleep=sleep, clock=clock) if cold.get("status") == "COMPLETED" else {}
    c = extract(cold)
    kind = classify(c)
    return {
        "health_before": health,
        "idle_before": health.get("workers", {}).get("idle"),
        "gap_s": gap_s,
        "health_after_gap": health_after_gap,
        "kind": kind,
        "suspect_warm": kind == "suspect_warm",
        "cold": c,
        "warm": extract(warm) if warm else None,
        # Runpod's responses as received, so a later extraction fix can be re-applied.
        "raw": {"cold": cold, "warm": warm or None},
    }


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile. (Oct 14: int(n * p) as an index made p90 the max at n = 10.)"""
    import math

    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def _stats(values: list[float]) -> dict[str, float] | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {
        "n": len(v),
        "median": round(statistics.median(v), 3),
        "p90": round(percentile(v, 0.9), 3),
        "min": v[0],
        "max": v[-1],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [row_kind(r) for r in rows]
    ok = [r for r, k in zip(rows, kinds, strict=True) if k == "cold"]
    resumed = [r["cold"] for r, k in zip(rows, kinds, strict=True) if k == "resumed"]
    cold = [r["cold"] for r in ok]
    warm = [r["warm"] for r in ok if r.get("warm")]

    def seg(key):  # recomputed from marks so old run files get the same treatment
        return _stats([segments(c["marks"], c.get("fetch_s")).get(key) for c in cold])

    gb = [c["bytes_written"] / 1e9 for c in cold if c.get("bytes_written") is not None]
    rate = [c["bytes_written"] / 1e9 / c["fetch_s"] for c in cold
            if c.get("bytes_written") and c.get("fetch_s")]  # fmt: skip
    return {
        "iterations": len(rows),
        "completed": len(ok),
        "resumed": len(resumed),
        "suspect_warm_excluded": kinds.count("suspect_warm"),
        "gaps_s": sorted({r.get("gap_s") or 0 for r in rows}),
        "gpus": sorted({c["gpu"] for c in cold if c["gpu"]}),
        # v1 told hf_download from hf_cache by blob growth (unsound: ~130 KB seen for 19 GB);
        # its model_store label came from the path check and stands.
        "snapshot_sources": sorted(
            {
                c["snapshot_source"]
                for c in cold
                if c["snapshot_source"]
                and (c.get("fetch_metric") == "v2" or c["snapshot_source"] == "model_store")
            }
        ),
        "fetch_metrics": sorted({c.get("fetch_metric") or "v1" for c in cold}),
        "idle_before": [r["idle_before"] for r in ok],
        "workers": [c["worker_id"] for c in cold],
        "workers_seen": len({c["worker_id"] for c in cold}),
        "cold_delay_s": _stats([c["delay_s"] for c in cold]),
        "platform_est_s": _stats([c["platform_est_s"] for c in cold]),
        "ready_s": _stats([c["marks"].get("ready") for c in cold]),
        "startup_s": seg("startup_s"),
        "fetch_s": seg("fetch_s"),
        "import_s": seg("import_s"),
        "weights_s": seg("weights_s"),
        "warmup_s": seg("warmup_s"),
        "fetch_gb": _stats(gb),
        "fetch_gb_per_s": _stats(rate),
        "cold_exec_s": _stats([c["exec_s"] for c in cold]),
        "warm_exec_s": _stats([w["exec_s"] for w in warm]),
        # resumed: no process start inside the delay; age needs an image reporting origin_unix
        "resumed_delay_s": _stats([c["delay_s"] for c in resumed]),
        "resumed_exec_s": _stats([c["exec_s"] for c in resumed]),
        "resumed_process_age_s": _stats([c.get("process_age_s") for c in resumed]),
        "resumed_workers_seen": len({c["worker_id"] for c in resumed}),
    }


def render(label: str, s: dict[str, Any]) -> str:
    rows = [
        ("cold: queue delay (Runpod delayTime)", "cold_delay_s"),
        ("  platform: submit -> process start (est.)", "platform_est_s"),
        ("  worker: process start -> ready", "ready_s"),
        ("    startup (Python, worker imports)", "startup_s"),
        ("    snapshot fetch", "fetch_s"),
        ("    import torch / transformers / model code", "import_s"),
        ("    weight load", "weights_s"),
        ("    warm-up forward", "warmup_s"),
        ("fetch: GB written during it", "fetch_gb"),
        ("fetch: GB/s", "fetch_gb_per_s"),
        ("cold: first execution", "cold_exec_s"),
        ("warm: next execution, same worker", "warm_exec_s"),
    ]
    out = [
        f"## Cold start: {label}",
        "",
        f"{s['completed']}/{s['iterations']} cold starts, {s.get('resumed', 0)} resumed "
        f"(process already up at submit), {s['suspect_warm_excluded']} excluded as suspected "
        f"warm; gap after cold {', '.join(f'{g:g}' for g in s.get('gaps_s', [0]))} s; "
        f"GPU {', '.join(s['gpus']) or '?'}; "
        f"snapshot {', '.join(s['snapshot_sources']) or 'not measured (v1)'}; "
        f"{s['workers_seen']} distinct workers; idle slots before each submit: {s['idle_before']}",
        "",
        "| segment (seconds unless noted) | median | p90 | min | max |",
        "|---|---|---|---|---|",
    ]
    for name, key in rows:
        st = s.get(key)
        if not st:
            continue
        out.append(f"| {name} | {st['median']} | {st['p90']} | {st['min']} | {st['max']} |")
    if s.get("resumed"):
        out += [
            "",
            f"Resumed samples ({s['resumed']}, {s['resumed_workers_seen']} distinct workers): "
            "the job ran in a process started before the submit.",
            "",
            "| resumed (seconds) | median | p90 | min | max |",
            "|---|---|---|---|---|",
        ]
        for name, key in [
            ("queue delay (Runpod delayTime)", "resumed_delay_s"),
            ("first execution", "resumed_exec_s"),
            ("process up before submit", "resumed_process_age_s"),
        ]:
            st = s[key]
            if not st:
                continue
            out.append(f"| {name} | {st['median']} | {st['p90']} | {st['min']} | {st['max']} |")
    if "v1" in s.get("fetch_metrics", []):
        out += [
            "",
            "Fetch source and bytes here come from the v1 metric (blob-directory growth), which "
            "saw ~130 KB for a 19 GB snapshot: they are not a transfer measurement.",
        ]
    return "\n".join(out)


def load_rows(out_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("run-*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("endpoint_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--label", default=None, help="arm name, e.g. disk-a40")
    ap.add_argument("--input", type=Path, default=ROOT / "test_input.json")
    ap.add_argument("--settle-timeout", type=float, default=900.0, help="max wait to go cold, s")
    ap.add_argument(
        "--gap",
        type=float,
        default=0.0,
        help="seconds to wait after the endpoint goes cold before submitting",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="recompute summary.json and the table from OUT_DIR's run files; no API calls",
    )
    a = ap.parse_args()
    label = a.label or a.out_dir.name
    if a.summarize:
        summary = {"label": label, "endpoint_id": a.endpoint_id, **summarize(load_rows(a.out_dir))}
        (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(render(label, summary))
        return
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        sys.exit("RUNPOD_API_KEY is not set")
    job = json.loads(a.input.read_text())
    if not (job.get("input") or {}).get("debug"):
        sys.exit("the input must set input.debug = true, or the worker's timeline is not returned")
    api = RunpodAPI(a.endpoint_id, token)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, a.runs + 1):
        gap = f", then {a.gap:g} s" if a.gap else ""
        print(
            f"[{i}/{a.runs}] waiting for the endpoint to go cold{gap} ...",
            file=sys.stderr,
            flush=True,
        )
        row = iteration(api, job, settle_timeout=a.settle_timeout, gap_s=a.gap)
        rows.append(row)
        (a.out_dir / f"run-{i:02d}.json").write_text(json.dumps(row, indent=1))
        c = row["cold"]
        print(
            f"[{i}/{a.runs}] {c['status']} {row['kind']} delay {c['delay_s']} s, "
            f"ready {c['marks'].get('ready')} s, platform~{c['platform_est_s']} s, "
            f"fetch {c['snapshot_source']} {c['fetch_s']} s "
            f"({(c['bytes_written'] or 0) / 1e9:.2f} GB written), "
            f"exec {c['exec_s']} s / warm {(row['warm'] or {}).get('exec_s')} s, "
            f"idle before {row['idle_before']}",
            file=sys.stderr,
            flush=True,
        )
    summary = {"label": label, "endpoint_id": a.endpoint_id, **summarize(rows)}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(render(label, summary))


if __name__ == "__main__":
    main()
