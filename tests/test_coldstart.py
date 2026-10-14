"""coldstart.py against a fake Runpod API: cold detection, polling, extraction, summary."""

import json
from pathlib import Path

import coldstart as cs

ROOT = Path(__file__).resolve().parents[1]
# The live /health response from endpoint 7bmbtjrmxdedjo, 19 min after its last job (Oct 14).
LIVE_HEALTH = {
    "jobs": {"completed": 1, "failed": 0, "inProgress": 0, "inQueue": 0, "retried": 0},
    "workers": {"idle": 1, "initializing": 0, "ready": 1, "running": 0, "throttled": 0,
                "unhealthy": 0},
}  # fmt: skip


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_idle_slots_still_count_as_cold():
    assert cs.is_cold(LIVE_HEALTH)  # idle = scaled down, not billed; ready is undocumented
    busy = json.loads(json.dumps(LIVE_HEALTH))
    busy["workers"]["running"] = 1
    assert not cs.is_cold(busy)
    starting = json.loads(json.dumps(LIVE_HEALTH))
    starting["workers"]["initializing"] = 1
    assert not cs.is_cold(starting)
    queued = json.loads(json.dumps(LIVE_HEALTH))
    queued["jobs"]["inQueue"] = 1
    assert not cs.is_cold(queued)
    in_flight = json.loads(json.dumps(LIVE_HEALTH))
    in_flight["jobs"]["inProgress"] = 1
    assert not cs.is_cold(in_flight)


def _status(delay_ms, exec_ms, ready, worker="w1", gpu="NVIDIA A40", source="hf_download"):
    return {
        "status": "COMPLETED", "delayTime": delay_ms, "executionTime": exec_ms, "workerId": worker,
        "output": {"answers": {}, "debug": {"engine": {
            "gpu": gpu, "warmup_s": None,
            "timeline": {"basis": "process_start",
                         "marks": {"snapshot_resolved": 14.0, "weights_loaded": 24.0,
                                   "ready": ready},
                         "fetch": {"source": source, "bytes_added_to_cache": 19_400_000_000,
                                   "seconds": 13.2}}}}},
    }  # fmt: skip


def test_extract_splits_platform_from_in_process_time():
    e = cs.extract({**_status(36483, 920, ready=29.5), "client_wall_s": 38.1})
    assert e["delay_s"] == 36.483 and e["exec_s"] == 0.92 and e["marks"]["ready"] == 29.5
    assert e["platform_est_s"] == round(36.483 - 29.5, 3)  # submit -> process start
    assert e["snapshot_source"] == "hf_download" and e["fetch_bytes"] == 19_400_000_000


def test_extract_tolerates_a_worker_without_a_timeline():
    # The real Oct 14 queue response, from an image built before the timeline existed.
    real = json.loads((ROOT / "runs" / "queue-a40" / "test_input_response.json").read_text())
    e = cs.extract(real)
    assert e["delay_s"] == 36.483 and e["gpu"] == "NVIDIA A40"
    assert e["marks"] == {} and e["platform_est_s"] is None


class FakeAPI:
    def __init__(self, healths, statuses):
        self.healths, self.statuses, self.submitted = iter(healths), statuses, []

    def health(self):
        return next(self.healths)

    def run(self, job):
        self.submitted.append(job)
        return {"id": f"job{len(self.submitted)}"}

    def status(self, job_id):
        return next(self.statuses[job_id])


def test_iteration_waits_for_cold_then_runs_cold_and_warm():
    busy = json.loads(json.dumps(LIVE_HEALTH))
    busy["workers"]["running"] = 1
    api = FakeAPI(
        healths=[busy, LIVE_HEALTH, busy, LIVE_HEALTH, LIVE_HEALTH],  # a lone cold blip resets
        statuses={
            "job1": iter([{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"},
                          _status(36483, 920, ready=29.5)]),
            "job2": iter([_status(12, 240, ready=29.5)]),
        },
    )  # fmt: skip
    clock = Clock()
    row = cs.iteration(api, {"input": {"debug": True}}, interval=5, sleep=clock.sleep,
                       clock=clock.time)  # fmt: skip
    assert len(api.submitted) == 2 and row["idle_before"] == 1
    assert row["cold"]["delay_s"] == 36.483 and row["warm"]["exec_s"] == 0.24
    assert row["cold"]["client_wall_s"] == 2.0  # two 1-s polls before COMPLETED


def test_a_failed_cold_job_skips_the_warm_one():
    api = FakeAPI([LIVE_HEALTH] * 2, {"job1": iter([{"status": "FAILED", "error": "oom"}])})
    clock = Clock()
    row = cs.iteration(api, {"input": {}}, sleep=clock.sleep, clock=clock.time)
    assert len(api.submitted) == 1 and row["warm"] is None and row["cold"]["error"] == "oom"


def test_wait_until_times_out():
    import pytest

    clock = Clock()
    with pytest.raises(TimeoutError):
        cs.wait_until(lambda r: False, lambda: {}, 5, 30, clock.sleep, clock.time)


def test_summary_and_render():
    rows = []
    for i, (delay, ready) in enumerate([(36483, 29.5), (31000, 28.0), (40000, 30.1)]):
        rows.append({
            "idle_before": 1,
            "cold": cs.extract({**_status(delay, 900, ready, worker=f"w{i}"),
                                "client_wall_s": delay / 1000 + 1}),
            "warm": cs.extract({**_status(10, 230, ready), "client_wall_s": 1.0}),
        })  # fmt: skip
    s = cs.summarize(rows)
    assert s["completed"] == 3 and s["workers_seen"] == 3 and s["gpus"] == ["NVIDIA A40"]
    assert s["cold_delay_s"]["median"] == 36.483 and s["ready_s"]["median"] == 29.5
    assert s["warm_exec_s"]["median"] == 0.23
    text = cs.render("disk-a40", s)
    assert "| cold: queue delay (Runpod delayTime) | 36.483 |" in text
    assert "platform: submit -> process start" in text


def test_a_one_poll_cold_blip_does_not_start_a_run():
    busy = json.loads(json.dumps(LIVE_HEALTH))
    busy["workers"]["running"] = 1
    healths = iter([LIVE_HEALTH, busy, LIVE_HEALTH, LIVE_HEALTH])
    polled = []

    class API:
        def health(self):
            polled.append(1)
            return next(healths)

    clock = Clock()
    cs.wait_cold(API(), 5, 300, sleep=clock.sleep, clock=clock.time)
    assert len(polled) == 4  # cold, busy (reset), cold, cold


def test_a_suspiciously_fast_cold_start_is_flagged_and_excluded():
    api = FakeAPI(
        [LIVE_HEALTH] * 2,
        {"job1": iter([_status(800, 230, ready=29.5)]), "job2": iter([_status(10, 230, 29.5)])},
    )
    clock = Clock()
    row = cs.iteration(api, {"input": {}}, sleep=clock.sleep, clock=clock.time)
    assert row["suspect_warm"] is True
    s = cs.summarize([row])
    assert s["completed"] == 0 and s["suspect_warm_excluded"] == 1
