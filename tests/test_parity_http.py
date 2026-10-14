"""parity_http.py: wire bodies, reply -> record, and an end-to-end run against the fake worker."""

import base64
import json

import parity
import parity_http as ph
from fastapi.testclient import TestClient

from deciders.clef.engine import Decision, request_sha
from worker import app as wa
from worker.core import WorkerCore

from .test_worker_core import BODY

ROOT = ph.ROOT


def test_wire_body_inlines_images_as_data_urls():
    line = {"id": "x", "request": BODY, "images": ["fixtures/requests/images/shapes.png"]}
    body = ph.wire_body(line)
    assert body["images"][0].startswith("data:image/png;base64,")
    raw = base64.b64decode(body["images"][0].split(",", 1)[1])
    assert raw == (ROOT / "fixtures/requests/images/shapes.png").read_bytes()
    assert "images" not in ph.wire_body({"id": "y", "request": BODY})
    assert request_sha(BODY) == request_sha(body)  # images do not change the hash


def test_record_from_reply_success_error_and_sha_drift():
    line = {"id": "r", "request": BODY}
    reply = {
        "model": "m", "answers": {}, "usage": {"input_tokens": 3, "output_tokens": 0},
        "debug": {"request_sha": request_sha(BODY), "encoding": {"input_tokens": 3},
                  "questions": [{"id": "q", "probs": [1.0]}], "timing_ms": {"forward": 2.0}},
    }  # fmt: skip
    rec = ph.record_from_reply(line, 200, reply)
    assert rec["error"] is None and rec["encoding"]["input_tokens"] == 3
    assert rec["response"] == {k: v for k, v in reply.items() if k != "debug"}
    assert rec["questions"][0]["probs"] == [1.0] and "request_sha_local" not in rec
    bad = ph.record_from_reply(line, 400, {"error": "schema too long"})
    assert bad["error"] == "http 400: schema too long" and bad["response"] is None
    reply["debug"]["request_sha"] = "other"
    drift = ph.record_from_reply(line, 200, reply)
    assert drift["request_sha"] == "other" and drift["request_sha_local"] == request_sha(BODY)


class EchoEngine:
    """Answers from a table of per-record probabilities so the test controls the numbers."""

    def __init__(self, table):
        self.table = table

    def manifest(self):
        return {"model": "m", "revision": "r", "dtype": "bfloat16", "files": {}}

    def decide(self, body, images=None, rid=""):
        probs = self.table[body["state"]]
        return Decision(
            request_sha=request_sha(body),
            encoding={
                "input_tokens": 5,
                "ids_sha": "ids-" + body["state"],
                "spans_sha": "s",
                "questions": [],
            },  # fmt: skip
            questions=[
                {
                    "id": "q",
                    "type": "noul",
                    "option_ids": ["true", "false"],
                    "logits": [0.0, 0.0],
                    "probs": probs,
                }
            ],  # fmt: skip
            response={
                "model": "m",
                "answers": {"q": {"probability": probs[0]}},
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },  # fmt: skip
            timing_ms={"total": 1.0},
        )


def _lines():
    return [
        {"id": "a", "request": {"state": "a", "questions": {"q": {"type": "noul"}}}},
        {"id": "b", "request": {"state": "b", "questions": {"q": {"type": "noul"}}}},
        {"id": "boom", "request": {"state": "boom", "questions": {"q": {"type": "noul"}}}},
    ]


def test_end_to_end_against_the_fake_worker(tmp_path):
    ref_table = {"a": [0.7, 0.3], "b": [0.2, 0.8]}
    ref = tmp_path / "ref"
    ref.mkdir()
    for line in _lines()[:2]:
        d = EchoEngine(ref_table).decide(line["request"], rid=line["id"])
        rec = {"id": line["id"], "request_sha": d.request_sha, "encoding": d.encoding,
               "questions": d.questions, "response": d.response, "error": None}  # fmt: skip
        (ref / f"{line['id']}.json").write_text(json.dumps(rec))
    boom = {
        "id": "boom",
        "request_sha": request_sha(_lines()[2]["request"]),
        "questions": [],
        "error": "schema too long",
    }
    (ref / "boom.json").write_text(json.dumps(boom))

    class FailingOnBoom(EchoEngine):
        def decide(self, body, images=None, rid=""):
            if body["state"] == "boom":
                raise ValueError("schema too long")
            return super().decide(body, images, rid)

    run_table = {"a": [0.702, 0.298], "b": [0.2, 0.8]}  # inside a 0.005 bar
    with TestClient(wa.create_app(core=WorkerCore(FailingOnBoom(run_table)))) as c:

        def poster(body):
            r = c.post("/v1/decide", json=body, headers={"X-Decide-Debug": "1"})
            return r.status_code, r.json()

        counts = ph.collect(_lines(), tmp_path / "run", poster)
    assert counts == {"ok": 2, "error": 1}
    rep = parity.compare(ref, tmp_path / "run", max_dp=0.005, mean_dp=0.002)
    assert rep.passed, [r.reasons for r in rep.records if r.status == "FAIL"]
    assert {r.record: r.span for r in rep.records} == {"a": "ok", "b": "ok", "boom": "n/a"}
    assert all(r.request == "ok" for r in rep.records)
    run_table["a"] = [0.72, 0.28]  # now outside the bar
    with TestClient(wa.create_app(core=WorkerCore(FailingOnBoom(run_table)))) as c:
        ph.collect(
            _lines(),
            tmp_path / "run2",
            lambda b: (lambda r: (r.status_code, r.json()))(
                c.post("/v1/decide", json=b, headers={"X-Decide-Debug": "1"})
            ),
        )
    assert not parity.compare(ref, tmp_path / "run2", max_dp=0.005).passed
