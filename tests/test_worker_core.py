"""WorkerCore with a fake engine: validation, images, errors, debug block, queue-job mapping."""

import base64
import io
import json

import pytest

from deciders.clef.engine import Decision, request_sha
from worker import core as wc

BODY = {"state": "s", "questions": {"q": {"type": "noul"}}}


class FakeEngine:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def manifest(self):
        return {
            "model": "Cloudflare/clef-flash", "revision": "8b2e5fd", "dtype": "bfloat16",
            "sdp": "efficient", "gdn_fast_path": False, "gpu": "fake", "load_s": 0.1,
            "versions": {"torch": "0"}, "worker_decide_commit": "abc",
            "files": {"joint_schema_model.py": {"sha256": "deadbeef"}},
        }  # fmt: skip

    def decide(self, body, images=None, rid=""):
        self.calls.append((body, images, rid))
        if self.fail:
            raise self.fail
        return Decision(
            request_sha=request_sha(body),
            encoding={"input_tokens": 3, "ids_sha": "i", "spans_sha": "s", "questions": []},
            questions=[
                {
                    "id": "q",
                    "type": "noul",
                    "option_ids": ["true", "false"],
                    "logits": [1.0, 0.0],
                    "probs": [0.7311, 0.2689],
                }
            ],  # fmt: skip
            response={
                "model": "Cloudflare/clef-flash",
                "answers": {"q": {"probability": 0.7311}},
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },  # fmt: skip
            timing_ms={"encode": 1.0, "forward": 2.0, "total": 3.0},
        )


def _png_url():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_answer_without_and_with_debug():
    c = wc.WorkerCore(FakeEngine())
    o = c.decide(BODY)
    assert (
        o.status == 200
        and "debug" not in o.body
        and o.body["answers"]["q"]["probability"] == 0.7311
    )
    o = c.decide(BODY, debug=True)
    d = o.body["debug"]
    assert d["request_sha"] == request_sha(BODY) and d["encoding"]["input_tokens"] == 3
    assert d["questions"][0]["probs"] == [0.7311, 0.2689]
    assert d["engine"]["code_sha"] == "deadbeef" and d["engine"]["revision"] == "8b2e5fd"


def test_validation_and_image_errors_are_400_and_never_reach_the_engine():
    eng = FakeEngine()
    c = wc.WorkerCore(eng)
    assert c.decide("nope").status == 400
    o = c.decide({"state": "s", "questions": {}})
    assert o.status == 400 and o.body["problems"]
    assert c.decide({**BODY, "images": "x"}).status == 400
    assert c.decide({**BODY, "images": ["https://x/y.png"]}).status == 400
    assert c.decide({**BODY, "images": [_png_url()] * (wc.MAX_IMAGES + 1)}).status == 400
    assert eng.calls == []
    o = c.decide({**BODY, "images": [_png_url()]})
    assert o.status == 200 and len(eng.calls[0][1]) == 1 and eng.calls[0][1][0].size == (2, 2)


def test_engine_errors_map_to_400_or_500():
    assert (
        wc.WorkerCore(FakeEngine(ValueError("schema requires 300000 tokens"))).decide(BODY).status
        == 400
    )
    o = wc.WorkerCore(FakeEngine(RuntimeError("CUDA out of memory. Tried to allocate"))).decide(
        BODY
    )
    assert o.status == 500 and o.body["error"].startswith("inference failed: CUDA out of memory")


def test_handle_job_pops_debug_and_marks_failures_with_error():
    eng = FakeEngine()
    c = wc.WorkerCore(eng)
    out = c.handle_job({"input": {**BODY, "debug": True}})
    assert "error" not in out and out["debug"]["request_sha"] == request_sha(BODY)
    assert "debug" not in eng.calls[0][0]  # popped before hashing
    assert c.handle_job({"input": "x"})["error"]
    bad = c.handle_job({"input": {"state": "s", "questions": {}}})
    assert bad["error"] == "invalid request" and bad["status"] == 400 and bad["problems"]


def test_config_from_env_defaults_and_overrides():
    cfg = wc.config_from_env({})
    assert cfg.dtype == "bfloat16" and cfg.revision == wc.PINNED_REVISION
    assert cfg.expect_code_sha == wc.PINNED_CODE_SHA and cfg.max_length == 16384
    cfg = wc.config_from_env(
        {"DTYPE": "float32", "MODEL_REVISION": "", "CODE_SHA": "", "MAX_LENGTH": "4096"}
    )
    assert cfg.dtype == "float32" and cfg.revision is None and cfg.expect_code_sha is None
    assert cfg.max_length == 4096


def test_pins_match_the_oracle_manifest():
    m = json.load(
        open(wc.__file__.replace("worker/core.py", "fixtures/clef-flash/float32/manifest.json"))
    )
    assert m["revision"] == wc.PINNED_REVISION
    assert m["files"]["joint_schema_model.py"]["sha256"] == wc.PINNED_CODE_SHA


def test_test_input_json_is_a_valid_request():
    job = json.load(open(wc.__file__.replace("worker/core.py", "test_input.json")))
    body = dict(job["input"])
    body.pop("debug", None)
    from deciders.systemone import validate_request

    assert validate_request(body) == []


@pytest.mark.parametrize("key", ["state", "questions"])
def test_missing_top_level_keys_are_rejected(key):
    body = {k: v for k, v in BODY.items() if k != key}
    assert wc.WorkerCore(FakeEngine()).decide(body).status == 400
