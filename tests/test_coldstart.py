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
                         "marks": {"engine_init": 1.8, "snapshot_resolved": 14.0,
                                   "model_code_imported": 17.9, "weights_loaded": 24.0,
                                   "ready": ready},
                         "fetch": {"source": source, "snapshot_bytes": 19_400_000_000,
                                   "bytes_written": 19_390_000_000,
                                   "bytes_written_disk": 19_390_000_000,
                                   "bytes_written_hf_home": 19_380_000_000,
                                   "snapshot_dir_existed": False, "seconds": 13.2}}}}},
    }  # fmt: skip


def test_extract_splits_platform_from_in_process_time():
    e = cs.extract({**_status(36483, 920, ready=29.5), "client_wall_s": 38.1})
    assert e["delay_s"] == 36.483 and e["exec_s"] == 0.92 and e["marks"]["ready"] == 29.5
    assert e["platform_est_s"] == round(36.483 - 29.5, 3)  # submit -> process start
    assert e["snapshot_source"] == "hf_download" and e["bytes_written"] == 19_390_000_000
    assert e["fetch_metric"] == "v2" and e["snapshot_dir_existed"] is False
    assert e["segments"] == {"startup_s": 1.8, "fetch_s": 13.2, "import_s": 3.9,
                             "weights_s": 6.1, "warmup_s": None}  # fmt: skip


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
    assert row["raw"]["cold"]["delayTime"] == 36483  # Runpod's response kept as received


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


def test_a_process_older_than_the_submit_is_resumed_not_cold():
    # Arm D (FlashBoot, Oct 14): 0.5 s delay, the worker's ready mark 14 s old -> same process.
    api = FakeAPI(
        [LIVE_HEALTH] * 2,
        {"job1": iter([_status(511, 346, ready=13.967)]), "job2": iter([_status(10, 330, 13.967)])},
    )
    clock = Clock()
    row = cs.iteration(api, {"input": {}}, sleep=clock.sleep, clock=clock.time)
    assert row["kind"] == "resumed" and row["suspect_warm"] is False
    s = cs.summarize([row])
    assert s["completed"] == 0 and s["resumed"] == 1 and s["suspect_warm_excluded"] == 0
    assert s["resumed_delay_s"]["median"] == 0.511
    assert s["resumed_process_age_s"] is None  # this image did not report origin_unix
    text = cs.render("flashboot", s)
    assert "1 resumed" in text and "| queue delay (Runpod delayTime) | 0.511 |" in text
    assert "process up before submit" not in text


def test_process_age_comes_from_origin_unix_not_the_ready_mark():
    # The ready mark is seconds from process start to ready, not to the submit: a process
    # kept for 10 minutes still reports ready = 14 s. Age needs the worker's wall-clock origin.
    st = _status(511, 346, ready=13.967)
    st["output"]["debug"]["engine"]["timeline"]["origin_unix"] = 1_760_000_000.0
    c = cs.extract({**st, "submitted_unix": 1_760_000_600.0})
    assert c["process_age_s"] == 600.0 and c["platform_est_s"] < 0
    assert cs.summarize([{"idle_before": 1, "warm": None, "cold": c}])[
        "resumed_process_age_s"]["median"] == 600.0  # fmt: skip


def test_a_fast_start_without_a_timeline_is_still_suspect_warm():
    c = cs.extract(json.loads((ROOT / "runs" / "queue-a40" / "test_input_response.json")
                              .read_text()))  # fmt: skip
    assert cs.classify(c) == "cold"  # 36 s, no timeline: cold by delay
    assert cs.classify({**c, "delay_s": 0.8}) == "suspect_warm"
    assert cs.classify({**c, "status": "FAILED"}) == "failed"


def test_old_run_files_are_reclassified_from_the_cold_record():
    # An arm-D file written before `kind`: flagged suspect_warm, actually resumed.
    old = {"idle_before": 1, "suspect_warm": True, "warm": None,
           "cold": cs.extract(_status(497, 1267, ready=13.967))}  # fmt: skip
    assert cs.row_kind(old) == "resumed" and cs.summarize([old])["resumed"] == 1


def test_gap_waits_after_cold_and_records_health():
    api = FakeAPI(
        [LIVE_HEALTH, LIVE_HEALTH, LIVE_HEALTH],  # two for the cold streak, one after the gap
        {"job1": iter([_status(20000, 900, ready=14.0)]), "job2": iter([_status(10, 330, 14.0)])},
    )
    clock = Clock()
    row = cs.iteration(api, {"input": {}}, gap_s=600, interval=5, sleep=clock.sleep,
                       clock=clock.time)  # fmt: skip
    assert row["gap_s"] == 600 and row["health_after_gap"] == LIVE_HEALTH
    assert clock.t >= 605 and row["kind"] == "cold"  # one 5-s poll + the gap + job polls
    assert cs.summarize([row])["gaps_s"] == [600]


def test_p90_is_nearest_rank_not_the_max():
    v = [float(x) for x in range(1, 11)]  # n = 10, the size that exposed the Oct 14 bug
    assert cs.percentile(v, 0.9) == 9.0 and cs.percentile(v, 0.5) == 5.0
    assert cs.percentile([3.0], 0.9) == 3.0
    assert cs._stats(v)["p90"] == 9.0 and cs._stats(v)["max"] == 10.0


def test_summarize_mode_rederives_old_v1_runs_and_warns(tmp_path, capsys, monkeypatch):
    # An Oct 14 arm-A run file: extracted fields only, v1 fetch metric (blob growth).
    v1 = {
        "idle_before": 1,
        "suspect_warm": False,
        "warm": None,
        "cold": {
            "status": "COMPLETED",
            "worker_id": "uv5",
            "delay_s": 30.404,
            "exec_s": 0.937,
            "gpu": "NVIDIA A40",
            "snapshot_source": "hf_download",
            "fetch_bytes": 129851,
            "fetch_s": 11.687,
            "platform_est_s": 5.059,
            "timeline_basis": "process_start",
            "marks": {
                "engine_init": 1.78,
                "snapshot_resolved": 13.468,
                "model_code_imported": 17.346,
                "weights_loaded": 25.345,
                "ready": 25.345,
            },
        },
    }
    (tmp_path / "run-01.json").write_text(json.dumps(v1))  # fmt: skip
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)  # summarize needs no key
    monkeypatch.setattr("sys.argv", ["coldstart.py", "ep", str(tmp_path), "--summarize"])
    cs.main()
    out = capsys.readouterr().out
    assert "startup (Python, worker imports) | 1.78 |" in out
    assert "import torch / transformers / model code | 3.878 |" in out
    assert "weight load | 7.999 |" in out
    assert "v1 metric" in out and "fetch: GB" not in out  # no byte rows from the broken metric
    assert "snapshot hf_download" not in out  # the label the v1 metric produced is not shown
    assert "not model_store (v1" in out  # but the run is not silently dropped from the label
    assert json.loads((tmp_path / "summary.json").read_text())["fetch_metrics"] == ["v1"]


def test_a_v1_model_store_label_stands_because_it_came_from_the_path():
    row = {"idle_before": 1, "suspect_warm": False, "warm": None, "cold": {
        "status": "COMPLETED", "worker_id": "w", "delay_s": 20.0, "exec_s": 0.9, "gpu": "g",
        "snapshot_source": "model_store", "fetch_metric": "v1", "fetch_s": 0.01,
        "platform_est_s": 6.0, "marks": {"ready": 14.0}}}  # fmt: skip
    assert cs.summarize([row])["snapshot_sources"] == ["model_store"]
    row["cold"]["snapshot_source"] = "hf_download"  # v1's byte-derived label: replaced, not hidden
    assert cs.summarize([row])["snapshot_sources"] == [cs.V1_NOT_STORE]
