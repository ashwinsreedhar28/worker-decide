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
            "gdn_kernels": {"chunk_gated_delta_rule": "fla", "fla_device": "cuda"},
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


def test_warmup_is_off_unless_asked():
    assert wc.config_from_env({}).warmup is False
    assert wc.config_from_env({"WARMUP": "1"}).warmup is True
    assert wc.config_from_env({"WARMUP": "true"}).warmup is True
    assert wc.config_from_env({"WARMUP": "0"}).warmup is False


def test_core_marks_ready_and_exposes_the_timeline():
    import time

    from deciders.clef.engine import Timeline

    class TimedEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.timeline = Timeline(origin=time.time() - 5.0)
            self.timeline.mark("weights_loaded")

        def manifest(self):
            return {**super().manifest(), "timeline": self.timeline.as_dict()}

    c = wc.WorkerCore(TimedEngine())
    marks = c.info["timeline"]["marks"]
    assert list(marks) == ["weights_loaded", "ready"] and marks["ready"] >= marks["weights_loaded"]
    o = c.decide(BODY, debug=True)
    assert o.body["debug"]["engine"]["timeline"]["marks"]["ready"] == marks["ready"]


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


def test_info_carries_the_gdn_kernels_report():
    info = wc.WorkerCore(FakeEngine()).info
    assert info["gdn_kernels"] == {"chunk_gated_delta_rule": "fla", "fla_device": "cuda"}
    assert info["gdn_fast_path"] is False


# -- batched path (BATCH_MAX > 1) ----------------------------------------------------------


class BatchEngine(FakeEngine):
    """FakeEngine plus the batched API: p(true) = 0.5 + tokens / 1000, so each request's answer
    is its own and a mis-routed one shows."""

    def __init__(self, fail_batched=False, fail_single=None):
        super().__init__()
        self.batches = []
        self.fail_batched = fail_batched
        self.fail_single = fail_single
        self.gate = None
        self.entered = None
        self.rids = []
        self.delay = 0.0

    def prepare(self, body, images=None, rid=""):
        from types import SimpleNamespace

        from deciders.clef.engine import Decision, Prepared

        if body["state"] == "reject":
            raise ValueError("schema exceeds max_length")
        self.rids.append(rid)
        enc = SimpleNamespace(input_ids=list(range(len(body["state"]))), questions=["q"])
        return Prepared(rid=rid, body=body, encoded=enc,
                        decision=Decision(request_sha=request_sha(body)), t0=0.0)  # fmt: skip

    def forward_batch(self, encoded_list):
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            self.gate.wait(2)
        if self.delay:
            import time

            time.sleep(self.delay)
        self.batches.append([len(e.input_ids) for e in encoded_list])
        if self.fail_batched and len(encoded_list) > 1:
            raise RuntimeError("CUDA out of memory")
        if (
            self.fail_single
            and len(encoded_list) == 1
            and len(encoded_list[0].input_ids) == self.fail_single
        ):
            raise RuntimeError("kernel failed")
        return [[([0.0, 0.0], [0.5 + len(e.input_ids) / 1000, 0.5 - len(e.input_ids) / 1000])]
                for e in encoded_list]  # fmt: skip

    def finish(self, p, per_q, forward_ms):
        d = p.decision
        d.timing_ms["forward"] = round(forward_ms, 2)
        d.response = {"answers": {"q": {"probability": per_q[0][1][0]}},
                      "usage": {"input_tokens": p.tokens, "output_tokens": 0}}  # fmt: skip
        return d


def body_of(n):
    return {"state": "s" * n, "questions": {"q": {"type": "noul"}}}


def test_batch_settings_from_env():
    env = {"BATCH_MAX": "8", "BATCH_MAX_TOKENS": "16384", "BATCH_MAX_RATIO": "1.5",
           "BATCH_MAX_WAIT_MS": "2"}  # fmt: skip
    s = wc.batch_settings_from_env(env)
    assert s == wc.BatchSettings(8, 16384, 1.5, 2.0)
    assert wc.batch_settings_from_env({}) == wc.BatchSettings(1, 32768, 2.0, 0.0)


def test_batching_is_off_by_default_and_without_a_batch_api():
    assert wc.WorkerCore(BatchEngine()).batcher is None
    core = wc.WorkerCore(FakeEngine(), wc.BatchSettings(max_batch=8))
    assert core.batcher is None and core.info["batching"] is None and core.concurrency == 1


def test_concurrent_requests_share_a_forward_and_each_gets_its_own_answer():
    import threading
    import time

    eng = BatchEngine()
    eng.gate, eng.entered = threading.Event(), threading.Event()
    core = wc.WorkerCore(eng, wc.BatchSettings(max_batch=8))
    assert core.concurrency == 8 and core.info["batching"]["max_batch"] == 8
    results = {}

    def call(n):
        results[n] = core.decide(body_of(n), debug=True)

    first = threading.Thread(target=call, args=(100,))
    first.start()
    assert eng.entered.wait(2)  # the first request is inside the forward, holding it
    rest = [threading.Thread(target=call, args=(n,)) for n in (110, 120, 130)]
    for t in rest:
        t.start()
    deadline = time.monotonic() + 2
    while len(core.batcher.pending) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    eng.gate.set()
    for t in [first, *rest]:
        t.join(2)
    assert eng.batches[0] == [100] and sorted(eng.batches[1]) == [110, 120, 130]
    for n, o in results.items():
        assert o.status == 200
        assert o.body["answers"]["q"]["probability"] == 0.5 + n / 1000  # routed to its request
        assert o.body["usage"]["input_tokens"] == n
        assert n in o.body["debug"]["batch"]["tokens"]
    assert results[100].body["debug"]["batch"]["size"] == 1
    assert {results[n].body["debug"]["batch"]["size"] for n in (110, 120, 130)} == {3}
    core.batcher.close()


def test_batched_path_maps_rejections_and_failures_to_statuses():
    eng = BatchEngine(fail_single=7)
    core = wc.WorkerCore(eng, wc.BatchSettings(max_batch=4))
    assert core.decide({"state": "reject", "questions": {"q": {"type": "noul"}}}).status == 400
    bad = core.decide(body_of(7))
    assert bad.status == 500 and "kernel failed" in bad.body["error"]
    assert core.decide(body_of(9)).status == 200
    core.batcher.close()


def test_image_requests_bypass_the_batcher():
    eng = BatchEngine()
    core = wc.WorkerCore(eng, wc.BatchSettings(max_batch=4))
    o = core.decide({**BODY, "images": [_png_url()]}, debug=True)
    assert o.status == 200 and eng.calls and eng.batches == []
    assert o.body["debug"]["batch"] is None
    core.batcher.close()


def test_settings_reach_the_batcher_and_forward_time_and_rid_reach_the_answer():
    eng = BatchEngine()
    eng.delay = 0.01
    settings = wc.BatchSettings(max_batch=3, max_padded_tokens=999, max_ratio=1.5, max_wait_ms=7)
    core = wc.WorkerCore(eng, settings)
    b = core.batcher
    assert (b.max_batch, b.max_padded_tokens, b.max_ratio, b.max_wait_s) == (3, 999, 1.5, 0.007)
    o = core.decide(body_of(40), debug=True)
    dbg = o.body["debug"]
    assert dbg["timing_ms"]["forward"] == dbg["batch"]["forward_ms"] >= 10.0
    assert len(eng.rids) == 1 and len(eng.rids[0]) == 12  # the request id goes to prepare
    core.batcher.close()


def test_a_request_that_waits_past_the_timeout_gets_a_500():
    import threading

    eng = BatchEngine()
    eng.gate = threading.Event()  # never set: the forward hangs
    core = wc.WorkerCore(eng, wc.BatchSettings(max_batch=2, timeout_s=0.1))
    o = core.decide(body_of(10))
    assert o.status == 500 and "no answer from the batcher in 0.1 s" in o.body["error"]
    eng.gate.set()
    core.batcher.close()


def test_a_dry_run_engine_is_not_batched():
    eng = BatchEngine()
    eng.model = None
    assert wc.WorkerCore(eng, wc.BatchSettings(max_batch=4)).batcher is None


def test_a_stopped_batcher_answers_500_not_a_hang():
    eng = BatchEngine()
    core = wc.WorkerCore(eng, wc.BatchSettings(max_batch=2))
    core.batcher.close()
    o = core.decide(body_of(10))
    assert o.status == 500 and "closed" in o.body["error"]
