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
