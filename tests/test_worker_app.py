"""worker/app.py over FastAPI's TestClient with the fake engine from test_worker_core."""

import json
import os
import threading
import time

from fastapi.testclient import TestClient

from worker import app as wa
from worker.core import WorkerCore

from .test_worker_core import BODY, FakeEngine


def _client(core=None, loader=None):
    return TestClient(wa.create_app(loader=loader or wa.default_loader, core=core))


def test_decide_plain_and_debug_header():
    with _client(core=WorkerCore(FakeEngine())) as c:
        r = c.post("/v1/decide", json=BODY)
        assert r.status_code == 200 and "debug" not in r.json()
        r = c.post("/v1/decide", json=BODY, headers={"X-Decide-Debug": "1"})
        assert r.status_code == 200 and r.json()["debug"]["encoding"]["input_tokens"] == 3
        assert c.post("/v1/systemone", json=BODY).json() == c.post("/v1/decide", json=BODY).json()
        assert c.get("/ping").status_code == 200
        info = c.get("/").json()
        assert info["state"] == "ready" and info["engine"]["revision"] == "8b2e5fd"


def test_request_faults():
    with _client(core=WorkerCore(FakeEngine())) as c:
        r = c.post("/v1/decide", content=b"{not json", headers={"content-type": "application/json"})
        assert r.status_code == 400
        r = c.post("/v1/decide", json={"state": "s", "questions": {}})
        assert r.status_code == 400 and r.json()["problems"]
        big = json.dumps({**BODY, "state": "x" * (wa.MAX_BODY_BYTES + 1)}).encode()
        r = c.post("/v1/decide", content=big, headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_ping_is_204_while_loading_then_200_and_decide_is_503_meanwhile():
    gate = threading.Event()

    def slow_loader():
        gate.wait(timeout=10)
        return WorkerCore(FakeEngine())

    with _client(loader=slow_loader) as c:
        assert c.get("/ping").status_code == 204
        assert c.get("/").json()["state"] == "loading"
        r = c.post("/v1/decide", json=BODY)
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        gate.set()
        for _ in range(100):
            if c.get("/ping").status_code == 200:
                break
            time.sleep(0.02)
        assert c.get("/ping").status_code == 200
        assert c.post("/v1/decide", json=BODY).status_code == 200


def test_ping_is_500_when_loading_failed():
    def bad_loader():
        raise RuntimeError("joint_schema_model.py sha mismatch")

    with _client(loader=bad_loader) as c:
        for _ in range(100):
            if c.get("/ping").status_code == 500:
                break
            time.sleep(0.02)
        r = c.get("/ping")
        assert r.status_code == 500 and "sha mismatch" in r.json()["error"]
        assert c.post("/v1/decide", json=BODY).status_code == 503


def test_main_rejects_unknown_transport(monkeypatch):
    import pytest

    from worker import __main__ as wm

    monkeypatch.setenv("TRANSPORT", "carrier-pigeon")
    with pytest.raises(SystemExit, match="TRANSPORT"):
        wm.main()


def test_fix_hf_home_only_rewrites_when_workspace_is_the_mount(monkeypatch, tmp_path):
    from worker import __main__ as wm

    env = {"HF_HOME": "/runpod-volume/.cache/huggingface"}
    real_exists = wm.Path.exists
    monkeypatch.setattr(wm.Path, "exists", lambda p: str(p) == "/workspace" or real_exists(p))
    assert wm.fix_hf_home(env) == "/workspace/.cache/huggingface"
    env = {"HF_HOME": "/elsewhere"}
    assert wm.fix_hf_home(env) is None and env["HF_HOME"] == "/elsewhere"


def test_pod_script_clones_then_execs(monkeypatch, tmp_path):
    import subprocess

    from worker import __main__ as wm

    calls = []
    outcomes = iter([1, 1, 0])  # DNS not up yet, twice; then the clone succeeds

    def fake_run(cmd, capture_output, text):
        calls.append(("run", cmd))
        return subprocess.CompletedProcess(cmd, next(outcomes), "", "could not resolve host")

    monkeypatch.setattr(wm, "POD_CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(wm.subprocess, "run", fake_run)
    monkeypatch.setattr(wm.time, "sleep", lambda s: calls.append(("sleep", s)))
    monkeypatch.setattr(wm.os, "chdir", lambda p: calls.append(("chdir", str(p))))
    monkeypatch.setattr(wm.os, "execvp", lambda f, a: calls.append(("exec", a)))
    wm.run_pod_script("scripts/pod_http_parity.sh", {"REPO_REF": "m2/x"}, wait=1.5)
    runs = [c for c in calls if c[0] == "run"]
    assert len(runs) == 3 and "--branch" in runs[0][1] and "m2/x" in runs[0][1]
    assert calls.count(("sleep", 1.5)) == 2
    assert calls[-2] == ("chdir", str(tmp_path / "checkout"))
    assert calls[-1] == ("exec", ["bash", "scripts/pod_http_parity.sh"])


def test_pod_script_gives_up_after_tries(monkeypatch, tmp_path):
    import subprocess

    import pytest

    from worker import __main__ as wm

    monkeypatch.setattr(wm, "POD_CHECKOUT", tmp_path / "never")
    monkeypatch.setattr(
        wm.subprocess,
        "run",
        lambda c, capture_output, text: subprocess.CompletedProcess(c, 1, "", "x"),
    )
    monkeypatch.setattr(wm.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="giving up"):
        wm.run_pod_script("s.sh", {}, tries=3)


def test_pod_script_pops_pod_script_from_the_environment(monkeypatch, tmp_path):
    """The script launches `python -m worker`; a lingering POD_SCRIPT would recurse."""
    import subprocess

    from worker import __main__ as wm

    monkeypatch.setenv("POD_SCRIPT", "scripts/pod_http_parity.sh")
    monkeypatch.setattr(wm, "POD_CHECKOUT", tmp_path / "co")
    monkeypatch.setattr(
        wm.subprocess,
        "run",
        lambda c, capture_output, text: subprocess.CompletedProcess(c, 0, "", ""),
    )
    monkeypatch.setattr(wm.os, "chdir", lambda p: None)
    monkeypatch.setattr(wm.os, "execvp", lambda f, a: None)
    wm.run_pod_script("scripts/pod_http_parity.sh")
    assert "POD_SCRIPT" not in os.environ


def test_root_reports_live_batcher_stats_when_batching():
    from worker.core import BatchSettings

    from .test_worker_core import BatchEngine, body_of

    with _client(core=WorkerCore(FakeEngine())) as c:
        assert c.get("/").json()["batcher"] is None
    core = WorkerCore(BatchEngine(), BatchSettings(max_batch=4))
    with _client(core=core) as c:
        assert c.post("/v1/decide", json=body_of(50)).status_code == 200
        info = c.get("/").json()
        assert info["batcher"]["requests"] == 1 and info["engine"]["batching"]["max_batch"] == 4
    core.batcher.close()
