"""hosted_to_run.py lines hosted probabilities up with the oracle's option ids."""

import json

import hosted_to_run


def test_convert(tmp_path):
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "pulse-1.json").write_text(
        json.dumps(
            {
                "id": "pulse-1",
                "request_sha": "abc",
                "questions": [
                    {
                        "id": "urgency",
                        "type": "score",
                        "option_ids": ["0", "1", "2"],
                        "probs": [0.2, 0.5, 0.3],
                    }
                ],
            }
        )
    )
    hosted = tmp_path / "hosted.jsonl"
    hosted.write_text(
        json.dumps(
            {
                "id": "pulse-1",
                "question": "urgency",
                "probabilities": [0.21, 0.49, 0.3],
                "confidence": 0.1,
                "expected": 1.1,
                "prompt_tokens": 10,
                "wall_ms": 5,
                "model": "m",
            }
        )
        + "\n"
        + json.dumps({"id": "pulse-404", "question": "urgency", "probabilities": [1, 0, 0]})
        + "\n"
    )
    n = hosted_to_run.convert(hosted, tmp_path / "run", oracle)
    assert n == 1
    rec = json.loads((tmp_path / "run" / "pulse-1.json").read_text())
    assert rec["questions"][0]["option_ids"] == ["0", "1", "2"]
    assert rec["questions"][0]["probs"] == [0.21, 0.49, 0.3]
    assert "encoding" not in rec and rec["request_sha"] == "abc"


def test_align_by_option_id_and_skip_mismatches():
    assert hosted_to_run._align({"y": 0.3, "x": 0.7}, ["x", "y"]) == [0.7, 0.3]
    assert hosted_to_run._align([0.2, 0.8], ["x", "y"]) == [0.2, 0.8]
    assert hosted_to_run._align({"x": 0.7, "z": 0.3}, ["x", "y"]) is None  # wrong ids
    assert hosted_to_run._align([0.2, 0.5, 0.3], ["x", "y"]) is None  # wrong length
