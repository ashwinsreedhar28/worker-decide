"""worker/app.py over FastAPI's TestClient with the fake engine from test_worker_core."""

import json
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
    from worker import __main__ as wm

    calls = []
    monkeypatch.setattr(wm, "POD_CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(wm.subprocess, "run", lambda cmd, check: calls.append(("run", cmd)))
    monkeypatch.setattr(wm.os, "chdir", lambda p: calls.append(("chdir", str(p))))
    monkeypatch.setattr(wm.os, "execvp", lambda f, a: calls.append(("exec", a)))
    wm.run_pod_script("scripts/pod_http_parity.sh", {"REPO_REF": "m2/x"})
    assert calls[0][0] == "run" and "--branch" in calls[0][1] and "m2/x" in calls[0][1]
    assert calls[1] == ("chdir", str(tmp_path / "checkout"))
    assert calls[2] == ("exec", ["bash", "scripts/pod_http_parity.sh"])
