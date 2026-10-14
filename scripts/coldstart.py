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
    job_id = api.run(job)["id"]
    final = wait_until(lambda s: s.get("status") in TERMINAL, lambda: api.status(job_id),
                       interval, timeout, sleep, clock)  # fmt: skip
    return {**final, "client_wall_s": round(clock() - t0, 3)}


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
        "fetch_bytes": fetch.get("bytes_added_to_cache"),
        "fetch_s": fetch.get("seconds"),
        "marks": marks,
        "timeline_basis": timeline.get("basis"),
        "warmup_s": engine.get("warmup_s"),
        # submission -> process start, estimated as Runpod's delay minus the worker's own
        # process-start -> ready time: scheduling, image pull (if any), container create.
        "platform_est_s": round(delay_s - ready_s, 3) if ready_s is not None else None,
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


def iteration(api, job, settle_timeout=900.0, interval=5.0, sleep=time.sleep, clock=time.time):
    health = wait_cold(api, interval, settle_timeout, sleep=sleep, clock=clock)
    cold = run_job(api, job, sleep=sleep, clock=clock)
    warm = run_job(api, job, sleep=sleep, clock=clock) if cold.get("status") == "COMPLETED" else {}
    c = extract(cold)
    return {
        "health_before": health,
        "idle_before": health.get("workers", {}).get("idle"),
        "suspect_warm": c["status"] == "COMPLETED" and c["delay_s"] < SUSPECT_WARM_DELAY_S,
        "cold": c,
        "warm": extract(warm) if warm else None,
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return {
        "n": len(v),
        "median": round(statistics.median(v), 3),
        "p90": round(v[min(len(v) - 1, int(len(v) * 0.9))], 3),
        "min": v[0],
        "max": v[-1],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["cold"]["status"] == "COMPLETED" and not r.get("suspect_warm")]
    cold = [r["cold"] for r in ok]
    warm = [r["warm"] for r in ok if r.get("warm")]
    return {
        "iterations": len(rows),
        "completed": len(ok),
        "suspect_warm_excluded": sum(1 for r in rows if r.get("suspect_warm")),
        "gpus": sorted({c["gpu"] for c in cold if c["gpu"]}),
        "snapshot_sources": sorted({c["snapshot_source"] for c in cold if c["snapshot_source"]}),
        "idle_before": [r["idle_before"] for r in ok],
        "workers_seen": len({c["worker_id"] for c in cold}),
        "cold_delay_s": _stats([c["delay_s"] for c in cold]),
        "platform_est_s": _stats([c["platform_est_s"] for c in cold]),
        "ready_s": _stats([c["marks"].get("ready") for c in cold]),
        "fetch_s": _stats([c["fetch_s"] for c in cold]),
        "weights_loaded_s": _stats([c["marks"].get("weights_loaded") for c in cold]),
        "cold_exec_s": _stats([c["exec_s"] for c in cold]),
        "warm_exec_s": _stats([w["exec_s"] for w in warm]),
    }


def render(label: str, s: dict[str, Any]) -> str:
    rows = [
        ("cold: queue delay (Runpod delayTime)", "cold_delay_s"),
        ("  platform: submit -> process start (est.)", "platform_est_s"),
        ("  worker: process start -> ready", "ready_s"),
        ("    snapshot fetch", "fetch_s"),
        ("    weights loaded (mark)", "weights_loaded_s"),
        ("cold: first execution", "cold_exec_s"),
        ("warm: next execution, same worker", "warm_exec_s"),
    ]
    out = [
        f"## Cold start: {label}",
        "",
        f"{s['completed']}/{s['iterations']} completed cold starts "
        f"({s['suspect_warm_excluded']} excluded as suspected warm); "
        f"GPU {', '.join(s['gpus']) or '?'}; "
        f"snapshot {', '.join(s['snapshot_sources']) or '?'}; {s['workers_seen']} distinct "
        f"workers; idle slots before each submit: {s['idle_before']}",
        "",
        "| segment (seconds) | median | p90 | min | max |",
        "|---|---|---|---|---|",
    ]
    for name, key in rows:
        st = s.get(key)
        cells = ["-"] * 4 if not st else [st["median"], st["p90"], st["min"], st["max"]]
        out.append(f"| {name} | " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(out)


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
    a = ap.parse_args()
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        sys.exit("RUNPOD_API_KEY is not set")
    job = json.loads(a.input.read_text())
    if not (job.get("input") or {}).get("debug"):
        sys.exit("the input must set input.debug = true, or the worker's timeline is not returned")
    label = a.label or a.out_dir.name
    api = RunpodAPI(a.endpoint_id, token)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, a.runs + 1):
        print(
            f"[{i}/{a.runs}] waiting for the endpoint to go cold ...", file=sys.stderr, flush=True
        )
        row = iteration(api, job, settle_timeout=a.settle_timeout)
        rows.append(row)
        (a.out_dir / f"run-{i:02d}.json").write_text(json.dumps(row, indent=1))
        c = row["cold"]
        print(
            f"[{i}/{a.runs}] {c['status']} delay {c['delay_s']} s, ready {c['marks'].get('ready')} "
            f"s, platform~{c['platform_est_s']} s, fetch {c['snapshot_source']} {c['fetch_s']} s, "
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
