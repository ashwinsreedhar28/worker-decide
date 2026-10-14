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
