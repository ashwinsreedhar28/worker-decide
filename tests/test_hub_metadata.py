""".runpod/hub.json and .runpod/tests.json stay consistent with the worker's code.

The Hub builds from a GitHub release and runs tests.json on a fresh endpoint, so a typo in an
env key or a request the worker rejects would only surface after publishing. These catch it
in CI instead.
"""

import json
from pathlib import Path

from deciders.systemone import validate_request
from worker import core as wc

ROOT = Path(__file__).resolve().parents[1]
HUB = json.loads((ROOT / ".runpod" / "hub.json").read_text())
TESTS = json.loads((ROOT / ".runpod" / "tests.json").read_text())

# Keys the platform itself reads, not the worker (documented in Runpod Hub workers).
PLATFORM_KEYS = {"RUNPOD_INIT_TIMEOUT"}
SAMPLE = {
    "MODEL_NAME": "x/y",
    "MODEL_REVISION": "abc",
    "DTYPE": "float32",
    "DEVICE": "cpu",
    "MAX_LENGTH": "4096",
    "SDP": "math",
    "CODE_SHA": "def",
}


def test_required_hub_files_exist():
    for name in ("handler.py", "Dockerfile", "README.md"):
        assert (ROOT / name).is_file(), name
    assert HUB["type"] == "serverless" and HUB["config"]["runsOn"] == "GPU"
    assert HUB["category"] in {"audio", "embedding", "language", "video", "image"}


def test_every_env_key_is_read_by_the_worker():
    default = wc.config_from_env({})
    for e in HUB["config"]["env"]:
        key = e["key"]
        if key in PLATFORM_KEYS:
            continue
        assert key in SAMPLE, f"{key} is not something config_from_env reads"
        assert wc.config_from_env({key: SAMPLE[key]}) != default, f"{key} has no effect"


def test_pinned_defaults_match_the_code_and_the_oracle():
    defaults = {e["key"]: e["input"].get("default") for e in HUB["config"]["env"]}
    assert defaults["MODEL_REVISION"] == wc.PINNED_REVISION
    assert defaults["CODE_SHA"] == wc.PINNED_CODE_SHA
    assert defaults["DTYPE"] == wc.config_from_env({}).dtype == "bfloat16"
    assert defaults["MAX_LENGTH"] == wc.config_from_env({}).max_length


def test_presets_only_set_declared_keys():
    declared = {e["key"] for e in HUB["config"]["env"]}
    for preset in HUB["config"]["presets"]:
        assert set(preset["defaults"]) <= declared, preset["name"]


def test_hub_and_tests_agree_on_cuda():
    assert HUB["config"]["allowedCudaVersions"] == TESTS["config"]["allowedCudaVersions"]
    assert HUB["config"]["allowedCudaVersions"] == ["13.0"]  # the image is torch 2.11.0+cu130


def test_every_hub_test_is_a_request_the_worker_accepts():
    # A Hub test passes only on a 200; handle_job turns a rejected request into a FAILED job.
    for t in TESTS["tests"]:
        body = dict(t["input"])
        body.pop("debug", None)  # handle_job pops it before validating
        assert validate_request(body) == [], t["name"]


def test_timeouts_cover_a_cold_start_download():
    init_s = next(
        int(e["value"]) for e in TESTS["config"]["env"] if e["key"] == "RUNPOD_INIT_TIMEOUT"
    )
    for t in TESTS["tests"]:
        assert t["timeout"] >= init_s * 1000, t["name"]  # any test may land on a cold worker
