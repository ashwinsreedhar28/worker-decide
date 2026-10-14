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

    tl = engine.Timeline(origin=time.time() - 10.0)  # pretend the process started 10 s ago
    a = tl.mark("snapshot_resolved")
    b = tl.mark("weights_loaded")
    assert 9.9 < a <= b < 60 and tl.basis == "process_start"
    tl.fetch = {"source": "hf_download", "bytes_added_to_cache": 5, "seconds": 1.0}
    d = tl.as_dict()
    assert list(d["marks"]) == ["snapshot_resolved", "weights_loaded"] and d["fetch"]["source"]


def test_fetch_accounting(tmp_path):
    assert engine.dir_bytes(tmp_path / "absent") == 0
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "a").write_bytes(b"x" * 100)
    (tmp_path / "blobs" / "b").write_bytes(b"y" * 23)
    (tmp_path / "blobs" / "link").symlink_to(tmp_path / "blobs" / "a")  # not double-counted
    assert engine.dir_bytes(tmp_path / "blobs") == 123
    assert engine.fetch_source(True, 0, 0) == "model_store"
    assert engine.fetch_source(False, 123, 123) == "hf_cache"  # nothing new written
    assert engine.fetch_source(False, 0, 19_000_000_000) == "hf_download"
