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
