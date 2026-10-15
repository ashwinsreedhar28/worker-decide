"""engine.py's torch-free pieces: request hashing, answer assembly, data-URL image decoding."""

import base64
import io
import json
from dataclasses import dataclass

import pytest

from deciders.clef import engine

ROOT = engine.ROOT


def test_request_sha_ignores_images_and_matches_the_committed_oracle():
    body = {"state": "s", "questions": {"q": {"type": "noul"}}}
    with_images = {**body, "images": ["data:image/png;base64,AAAA"]}
    assert engine.request_sha(body) == engine.request_sha(with_images)
    n = 0
    for p in sorted((ROOT / "fixtures" / "requests").glob("*.jsonl")):
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            line = json.loads(ln)
            oracle = ROOT / "fixtures" / "clef-flash" / "float32" / f"{line['id']}.json"
            rec = json.loads(oracle.read_text())
            assert engine.request_sha(line["request"]) == rec["request_sha"], line["id"]
            n += 1
    assert n == 87


@dataclass(frozen=True)
class Q:
    question_id: str
    question_type: int
    option_ids: tuple


def test_assemble_builds_questions_and_answers_in_encoder_order():
    qs = (Q("b", 1, ("x", "y")), Q("a", 0, ("true", "false")))
    per_q = [([1.0, 0.0], [0.7311, 0.2689]), ([0.0, 2.0], [0.1192, 0.8808])]
    body = {"questions": {"a": {"type": "noul"}, "b": {"type": "choice"}}}
    seen = []

    def answer(spec, probs_by_id):
        seen.append((spec["type"], probs_by_id))
        return {"p": probs_by_id}

    questions, answers = engine.assemble(qs, per_q, body, answer)
    assert [q["id"] for q in questions] == ["b", "a"]
    assert questions[0] == {
        "id": "b", "type": "choice", "option_ids": ["x", "y"],
        "logits": [1.0, 0.0], "probs": [0.7311, 0.2689],
    }  # fmt: skip
    assert answers["a"] == {"p": {"true": 0.1192, "false": 0.8808}}
    assert seen[0] == ("choice", {"x": 0.7311, "y": 0.2689})


def _png_data_url() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_images_from_data_urls_roundtrip_and_rejections():
    (im,) = engine.images_from_data_urls([_png_data_url()])
    assert im.size == (4, 3) and im.mode == "RGB"
    for bad in ["https://example.com/x.png", "data:image/png;base64,!!!", 42]:
        with pytest.raises(ValueError, match="images\\[0\\]"):
            engine.images_from_data_urls([bad])


def _store(root, model, revisions, main=None):
    repo = root / ("models--" + model.replace("/", "--"))
    for rev in revisions:
        (repo / "snapshots" / rev).mkdir(parents=True)
    if main:
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text(main + "\n")
    return repo


PINNED = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"


def test_model_store_serves_the_pinned_revision(tmp_path):
    repo = _store(tmp_path, "Cloudflare/clef-flash", [PINNED, "17f0b0a" + "0" * 33])
    got = engine.resolve_model_store("Cloudflare/clef-flash", PINNED, tmp_path)
    assert got == repo / "snapshots" / PINNED


def test_model_store_with_another_revision_fails_loudly(tmp_path):
    _store(tmp_path, "Cloudflare/clef-flash", ["17f0b0a" + "0" * 33])
    with pytest.raises(RuntimeError, match=r"\['17f0b0a'\].*pinned to 8b2e5fd"):
        engine.resolve_model_store("Cloudflare/clef-flash", PINNED, tmp_path)


def test_no_model_store_falls_back_to_the_hub(tmp_path):
    assert engine.resolve_model_store("Cloudflare/clef-flash", PINNED, tmp_path) is None
    _store(tmp_path, "someone/else", [PINNED])  # a different model in the store
    assert engine.resolve_model_store("Cloudflare/clef-flash", PINNED, tmp_path) is None


def test_unpinned_follows_refs_main(tmp_path):
    rev = "17f0b0a" + "0" * 33
    repo = _store(tmp_path, "Cloudflare/clef-flash", [rev], main=rev)
    assert (
        engine.resolve_model_store("Cloudflare/clef-flash", None, tmp_path)
        == repo / "snapshots" / rev
    )
    _store(tmp_path, "x/y", [rev])  # no refs/main: nothing to follow
    assert engine.resolve_model_store("x/y", None, tmp_path) is None


def test_model_store_root_is_runpods_documented_path():
    assert str(engine.MODEL_STORE_ROOT) == "/runpod-volume/huggingface-cache/hub"


def test_process_start_is_in_the_past_and_recent():
    import time

    start = engine.process_start_time()
    if start is None:  # non-Linux: the timeline falls back to engine-init basis
        assert engine.Timeline().basis == "engine_init"
        return
    now = time.time()
    assert now - 3600 * 24 < start <= now + 1  # this test process started today, not 1970


def test_timeline_marks_are_seconds_since_origin_and_monotonic():
    import time

    origin = time.time() - 10.0
    tl = engine.Timeline(origin=origin)  # pretend the process started 10 s ago
    a = tl.mark("snapshot_resolved")
    b = tl.mark("weights_loaded")
    assert 9.9 < a <= b < 60 and tl.basis == "process_start"
    tl.fetch = {"source": "hf_download", "bytes_added_to_cache": 5, "seconds": 1.0}
    d = tl.as_dict()
    assert list(d["marks"]) == ["snapshot_resolved", "weights_loaded"] and d["fetch"]["source"]
    assert d["origin_unix"] == round(origin, 3)


def test_fetch_accounting(tmp_path):
    assert engine.dir_bytes(tmp_path / "absent") == 0
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "a").write_bytes(b"x" * 100)
    (tmp_path / "blobs" / "b").write_bytes(b"y" * 23)
    (tmp_path / "blobs" / "link").symlink_to(tmp_path / "blobs" / "a")  # not double-counted
    assert engine.dir_bytes(tmp_path / "blobs") == 123


def test_snapshot_bytes_follow_symlinks_to_wherever_the_bytes_live(tmp_path):
    # The HF layout: snapshots/<rev>/file -> a blob that may live elsewhere (e.g. a xet store).
    store = tmp_path / "elsewhere"
    store.mkdir()
    (store / "shard").write_bytes(b"w" * 1000)
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(store / "shard")
    (snap / "config.json").write_bytes(b"{}")
    assert engine.snapshot_bytes(snap) == 1002
    assert engine.dir_bytes(snap) == 2  # the Oct 14 metric's blind spot: symlinks not counted
    assert engine.snapshot_bytes(tmp_path / "absent") == 0


def test_disk_used_resolves_a_missing_path_to_its_nearest_ancestor(tmp_path):
    assert engine.disk_used(tmp_path / "not" / "yet" / "created") == engine.disk_used(tmp_path)


def test_fetch_source_needs_the_bytes_to_match_the_snapshot():
    snap = 19_400_000_000
    assert engine.fetch_source(True, 0, snap) == "model_store"
    assert engine.fetch_source(False, 19_380_000_000, snap) == "hf_download"
    assert (
        engine.fetch_source(False, 129_851, snap) == "hf_cache"
    )  # Oct 14's 130 KB: not a download
    assert engine.fetch_source(False, 0, 0) == "hf_cache"


def _fake_transformers(monkeypatch, **bound):
    import sys
    import types

    names = ["transformers", "transformers.models", "transformers.models.qwen3_5"]
    mods = {n: types.ModuleType(n) for n in names}
    m = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    for k, v in bound.items():
        setattr(m, k, v)
    mods["transformers.models"].qwen3_5 = mods["transformers.models.qwen3_5"]
    mods["transformers"].models = mods["transformers.models"]
    mods["transformers.models.qwen3_5"].modeling_qwen3_5 = m
    for n, mod in {**mods, m.__name__: m}.items():
        monkeypatch.setitem(sys.modules, n, mod)


def _from(module):
    def f():
        pass

    f.__module__ = module
    return f


def test_gdn_kernels_reads_what_transformers_bound(monkeypatch):
    import sys
    import types

    rms = type("FusedRMSNormGated", (), {"__module__": "fla.modules.fused_norm_gate"})
    chunk = _from("fla.ops.gated_delta_rule.chunk")
    _fake_transformers(monkeypatch, chunk_gated_delta_rule=chunk, FusedRMSNormGated=rms,
                       causal_conv1d_fn=None)  # fmt: skip
    fla_utils = types.ModuleType("fla.utils")
    fla_utils.device_platform = "cuda"
    monkeypatch.setitem(sys.modules, "fla.utils", fla_utils)
    assert engine.gdn_kernels() == {
        "source": "transformers", "chunk_gated_delta_rule": "fla", "fused_rms_norm_gated": "fla",
        "causal_conv1d": "torch", "fla_device": "cuda",
    }  # fmt: skip
    assert engine.gdn_fast_path() is False  # the full fast path needs causal-conv1d too


def test_gdn_kernels_pure_torch_and_full_fast_path(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "fla.utils", raising=False)
    _fake_transformers(monkeypatch, chunk_gated_delta_rule=None, FusedRMSNormGated=None,
                       causal_conv1d_fn=None)  # fmt: skip
    k = engine.gdn_kernels()
    assert (k["chunk_gated_delta_rule"], k["fused_rms_norm_gated"], k["causal_conv1d"]) == (
        "torch", "torch", "torch")  # fmt: skip
    assert k["fla_device"] is None  # fla not imported: not probed here
    conv = _from("causal_conv1d.cpp")
    _fake_transformers(monkeypatch, chunk_gated_delta_rule=_from("fla.ops"),
                       FusedRMSNormGated=None, causal_conv1d_fn=conv)  # fmt: skip
    assert engine.gdn_fast_path() is True


def test_gdn_kernels_without_transformers_reports_installed_packages(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)  # import fails
    # ...and so does the submodule, which a cached entry would otherwise serve (test_fused.py
    # imports it into the same process)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5", None)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "fla" else (None if name ==
                        "causal_conv1d" else real(name, *a)))  # fmt: skip
    assert engine.gdn_kernels() == {
        "source": "find_spec", "chunk_gated_delta_rule": "fla", "fused_rms_norm_gated": "fla",
        "causal_conv1d": "torch", "fla_device": None,
    }  # fmt: skip
    assert engine.gdn_fast_path() is False


def test_gdn_kernels_find_spec_fallback_both_and_neither(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5", None)
    real = importlib.util.find_spec
    for installed, want in (({"fla", "causal_conv1d"}, ("fla", "fla", "causal_conv1d")),
                            (set(), ("torch", "torch", "torch"))):  # fmt: skip

        def spec(name, *a, inst=installed):
            if name in {"fla", "causal_conv1d"}:
                return object() if name in inst else None
            return real(name, *a)

        monkeypatch.setattr(importlib.util, "find_spec", spec)
        k = engine.gdn_kernels()
        assert (k["chunk_gated_delta_rule"], k["fused_rms_norm_gated"], k["causal_conv1d"]) == want
        assert engine.gdn_fast_path() is (want[2] != "torch")


def test_gdn_kernels_fla_imported_without_a_probe_value_is_none(monkeypatch):
    import sys
    import types

    _fake_transformers(monkeypatch, chunk_gated_delta_rule=None, FusedRMSNormGated=None,
                       causal_conv1d_fn=None)  # fmt: skip
    monkeypatch.setitem(sys.modules, "fla.utils", types.ModuleType("fla.utils"))
    assert engine.gdn_kernels()["fla_device"] is None  # not the string "None"


def test_records_from_output_takes_one_entry_per_record_and_refuses_the_rest():
    out = [["a1", "a2"], ["b1"], ["c1", "c2", "c3"], ["extra"]]
    assert engine.records_from_output(out, 3) == [["a1", "a2"], ["b1"], ["c1", "c2", "c3"]]
    assert engine.records_from_output(out, 0) == []
    with pytest.raises(RuntimeError, match="no entry per record for 5 records"):
        engine.records_from_output(out, 5)

    class Tensor:  # an output that is one tensor, not a list per record
        shape = (5,)

        def __len__(self):
            return 5

        def __getitem__(self, i):
            return Tensor()

    with pytest.raises(RuntimeError, match="entry 0 is not a list of question logits"):
        engine.records_from_output(Tensor(), 2)


# -- the batched API against decide(), on an engine built without weights -----------------


class FT:
    """Just enough of a tensor for ClefEngine.forward/forward_batch."""

    def __init__(self, xs):
        self.xs = [float(x) for x in xs]

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.xs)

    def softmax(self, dim):
        import math

        m = max(self.xs)
        e = [math.exp(x - m) for x in self.xs]
        return FT([x / sum(e) for x in e])


def _q(qid, qtype, options):
    from types import SimpleNamespace

    return SimpleNamespace(question_id=qid, question_type=qtype, question_span=(0, 1),
                           option_spans=[(i, i + 1) for i in range(len(options))],
                           option_ids=list(options))  # fmt: skip


def _bare_engine(logits_for=None):
    """A ClefEngine without __init__: fake jsm, processor and torch. Each record's encoding
    is one token per word of its state; each question's logits come from `logits_for`."""
    import contextlib
    import threading
    from types import SimpleNamespace

    e = engine.ClefEngine.__new__(engine.ClefEngine)
    e.cfg = engine.EngineConfig()
    e.lock, e.encode_lock = threading.Lock(), threading.Lock()
    e.device = SimpleNamespace(type="cpu")
    e.model = object()
    e._sdp_backends = None
    e.torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                              cuda=SimpleNamespace(synchronize=lambda: None))  # fmt: skip
    e.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    logits_for = logits_for or (lambda rec, q: [0.3 * len(rec["state"].split()), 0.0])

    def encode_record(tok, rec, max_length, processor):
        qs = [_q(qid, 0, ["true", "false"]) for qid in rec["questions"]]
        return SimpleNamespace(input_ids=list(range(len(rec["state"].split()))), questions=qs,
                               rec=rec)  # fmt: skip

    def collate_records(recs, pad, dev):
        return {"recs": [r.rec for r in recs], "qs": [r.questions for r in recs]}

    def model(batch):  # one list of per-question logits per record, as the reference returns
        return [[FT(logits_for(r, q)) for q in qs]
                for r, qs in zip(batch["recs"], batch["qs"], strict=True)]  # fmt: skip

    e.model = model

    def answer(spec, probs):
        return {"probability": probs["true"]}

    e.jsm = SimpleNamespace(encode_record=encode_record, collate_records=collate_records,
                            systemone_answer=answer)  # fmt: skip
    return e


def test_prepare_then_finish_equals_decide_for_one_request():
    e = _bare_engine()
    body = {"state": "a b c d", "questions": {"q1": {"type": "noul"}, "q2": {"type": "noul"}}}
    d = e.decide(body, rid="r1")
    p = e.prepare(body, rid="r1")
    assert p.tokens == 4 and p.rid == "r1"
    per = e.forward_batch([p.encoded])
    b = e.finish(p, per[0], forward_ms=12.345)
    assert (b.request_sha, b.encoding, b.questions, b.response) == (
        d.request_sha, d.encoding, d.questions, d.response)  # fmt: skip
    assert b.response["usage"] == {"input_tokens": 4, "output_tokens": 0}
    assert b.encoding["input_tokens"] == 4 and b.encoding["questions"][0]["id"] == "q1"
    assert b.timing_ms["forward"] == 12.35 and "encode" in b.timing_ms and "total" in b.timing_ms


def test_forward_batch_returns_each_records_own_outputs_in_order():
    e = _bare_engine()
    bodies = [{"state": "w " * n, "questions": {f"q{i}": {"type": "noul"} for i in range(k)}}
              for n, k in ((3, 1), (7, 2), (5, 1))]  # fmt: skip
    ps = [e.prepare(b) for b in bodies]
    per = e.forward_batch([p.encoded for p in ps])
    assert [len(r) for r in per] == [1, 2, 1]
    assert [r[0][0][0] for r in per] == pytest.approx([0.9, 2.1, 1.5])  # 0.3 x words


def test_forward_batch_refuses_outputs_padded_to_the_widest_question():
    import pytest

    e = _bare_engine(logits_for=lambda rec, q: [1.0, 0.0, -1.0])  # 3 values for 2 options
    p = e.prepare({"state": "a b", "questions": {"q": {"type": "noul"}}})
    with pytest.raises(RuntimeError, match="3 outputs for 2 options"):
        e.forward_batch([p.encoded])


def test_encode_serializes_tokenizer_use_across_threads():
    import threading
    import time

    e = _bare_engine()
    inside, peak = [0], [0]
    real = e.jsm.encode_record

    def slow(tok, rec, max_length, processor):
        inside[0] += 1
        peak[0] = max(peak[0], inside[0])
        time.sleep(0.02)
        inside[0] -= 1
        return real(tok, rec, max_length, processor)

    e.jsm.encode_record = slow
    body = {"state": "a", "questions": {"q": {"type": "noul"}}}
    ts = [threading.Thread(target=e.prepare, args=(body,)) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(2)
    assert peak[0] == 1


def test_forward_batch_refuses_an_output_with_too_few_question_entries():
    e = _bare_engine()
    real = e.model
    e.model = lambda batch: [qs[:1] for qs in real(batch)]  # drops each record's second question
    p = e.prepare({"state": "a b", "questions": {"q1": {"type": "noul"}, "q2": {"type": "noul"}}})
    with pytest.raises(RuntimeError, match="1 question outputs for 2 questions"):
        e.forward_batch([p.encoded])


def test_forward_batch_does_not_take_only_the_first_records_output():
    # The Oct 15 A40 bug: indexing the batch's output [0] gave every batch the first record's
    # answers. Two records with different word counts must get different answers.
    e = _bare_engine()
    ps = [e.prepare({"state": "w " * n, "questions": {"q": {"type": "noul"}}}) for n in (2, 9)]
    per = e.forward_batch([p.encoded for p in ps])
    assert per[0][0][0][0] == pytest.approx(0.6) and per[1][0][0][0] == pytest.approx(2.7)
