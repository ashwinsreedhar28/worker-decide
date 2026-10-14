"""Clef-flash engine: load the checkpoint once, answer SystemOne requests one at a time.

One code path for the oracle (``deciders/clef/oracle.py``) and the worker (``worker/``), so
parity between them is by construction: both call ``ClefEngine.decide``. The forward pass is
Cloudflare's own ``joint_schema_model.py`` from the pinned snapshot; nothing here touches logits
except the softmax that turns them into the probabilities SystemOne reports.

The torch-free helpers at the top (canonical, sha, request_sha, encoding_block, assemble,
images_from_data_urls) are importable without torch and are what the tests cover; the engine
itself needs the pinned environment in requirements-oracle.txt and a GPU.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Cloudflare/clef-flash"
# Runpod Model Store ("cached model" on an endpoint) mounts models here, in Hugging Face hub-cache
# layout; custom workers must resolve the path themselves (docs.runpod.io model-caching).
MODEL_STORE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
TYPE_NAMES = {0: "noul", 1: "choice", 2: "score"}
CODE_FILE = "joint_schema_model.py"


# ---- pure helpers (importable without torch) ---------------------------------------------


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_sha(body: dict[str, Any]) -> str:
    """sha256 of the canonical request without `images`: the oracle fixtures keep image paths
    outside the request, the worker receives them inline, and both must hash the same."""
    return sha(canonical({k: v for k, v in body.items() if k != "images"}))


def encoding_block(encoded: Any) -> dict[str, Any]:
    """Span bookkeeping for the parity gate, from an EncodedRecord-like object."""
    questions = [
        {
            "id": q.question_id,
            "type": TYPE_NAMES[q.question_type],
            "question_span": list(q.question_span),
            "option_spans": [list(s) for s in q.option_spans],
            "option_ids": list(q.option_ids),
        }
        for q in encoded.questions
    ]
    return {
        "input_tokens": len(encoded.input_ids),
        "ids_sha": sha(",".join(map(str, encoded.input_ids))),
        "spans_sha": sha(canonical(questions)),
        "questions": questions,
    }


def assemble(
    encoded_questions: Any, per_question: list[tuple[list[float], list[float]]], body: dict, answer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the per-question block and the SystemOne `answers` from (logits, probs) pairs.

    `answer(question_spec, {option_id: prob}) -> answer dict` is the checkpoint's own
    ``systemone_answer``; it is passed in so this stays importable without the snapshot.
    """
    questions: list[dict[str, Any]] = []
    answers: dict[str, Any] = {}
    for q, (logits, probs) in zip(encoded_questions, per_question, strict=True):
        questions.append(
            {
                "id": q.question_id,
                "type": TYPE_NAMES[q.question_type],
                "option_ids": list(q.option_ids),
                "logits": logits,
                "probs": probs,
            }
        )
        answers[q.question_id] = answer(
            body["questions"][q.question_id], dict(zip(q.option_ids, probs, strict=True))
        )
    return questions, answers


def images_from_data_urls(urls: list[str]) -> list[Any]:
    """Decode `data:image/...;base64,...` strings to RGB PIL images. Raises ValueError."""
    from PIL import Image

    out = []
    for i, u in enumerate(urls):
        if not isinstance(u, str) or not u.startswith("data:") or ";base64," not in u:
            raise ValueError(f"images[{i}]: expected a base64 data URL")
        try:
            raw = base64.b64decode(u.split(";base64,", 1)[1], validate=True)
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as e:  # noqa: BLE001 - any decode failure is the client's problem
            raise ValueError(f"images[{i}]: not a decodable image ({type(e).__name__})") from e
    return out


def resolve_model_store(model: str, revision: str | None, root: Path) -> Path | None:
    """The Model Store snapshot to load from, or None to fall back to the Hub.

    Only the pinned revision is accepted: if the store holds this model at a different commit,
    this raises instead of silently serving unpinned weights (the parity reference and the
    CODE_SHA pin both refer to one commit). Unpinned (revision None) follows refs/main.
    """
    repo = root / ("models--" + model.replace("/", "--"))
    if not repo.is_dir():
        return None
    snapshots = repo / "snapshots"
    if revision:
        snap = snapshots / revision
        if snap.is_dir():
            return snap
        present = sorted(p.name for p in snapshots.iterdir()) if snapshots.is_dir() else []
        raise RuntimeError(
            f"Model Store has {model} at {[r[:7] for r in present] or 'no snapshots'}, but this "
            f"worker is pinned to {revision[:7]}. Set the endpoint's cached model to that "
            "revision, or clear MODEL_REVISION and CODE_SHA to serve what the store holds "
            "(voids parity)."
        )
    ref = repo / "refs" / "main"
    if ref.is_file() and (snapshots / ref.read_text().strip()).is_dir():
        return snapshots / ref.read_text().strip()
    return None


def process_start_time() -> float | None:
    """Wall-clock time this process started (Linux /proc), or None elsewhere.

    The cold-start timeline is measured from here, so the in-process segment (fetch, imports,
    weight load, warm-up) is comparable across runs. What happens before the process exists —
    scheduling, image pull, container create — is the platform's segment, visible only from
    outside (Runpod's delayTime, or a client's clock).
    """
    try:
        with open("/proc/self/stat") as f:
            after_comm = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(after_comm[19])  # field 22 (starttime); after_comm[0] is field 3
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return time.time() - uptime + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def dir_bytes(path: Path) -> int:
    """Total size of regular files under path (symlinks not followed); 0 if absent."""
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def fetch_source(from_store: bool, bytes_before: int, bytes_after: int) -> str:
    """How the snapshot reached this worker: model_store, hf_cache (nothing new written to the
    HF cache) or hf_download (the cache's blobs grew)."""
    if from_store:
        return "model_store"
    return "hf_download" if bytes_after > bytes_before else "hf_cache"


class Timeline:
    """Startup marks in seconds since process start (or since engine init, if /proc is not
    available; `basis` says which)."""

    def __init__(self, origin: float | None = None):
        start = process_start_time() if origin is None else origin
        self.basis = "process_start" if start is not None else "engine_init"
        self.origin = start if start is not None else time.time()
        self.marks: dict[str, float] = {}
        self.fetch: dict[str, Any] = {}

    def mark(self, name: str) -> float:
        self.marks[name] = round(time.time() - self.origin, 3)
        return self.marks[name]

    def as_dict(self) -> dict[str, Any]:
        return {"basis": self.basis, "marks": dict(self.marks), "fetch": dict(self.fetch)}


def git_commit() -> str | None:
    """HEAD of the checkout, or WORKER_DECIDE_COMMIT (baked into the image at build time)."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("WORKER_DECIDE_COMMIT") or None


def file_shas(snapshot: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("joint_head.safetensors", "joint_head_config.json", CODE_FILE, "config.json"):
        p = snapshot / name
        if p.exists():
            out[name] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    for p in sorted(snapshot.glob("model-*.safetensors")):
        out[p.name] = {"bytes": p.stat().st_size}  # 5 GB shards: size only, revision pins them
    return out


# ---- the engine --------------------------------------------------------------------------


@dataclass
class EngineConfig:
    model: str = DEFAULT_MODEL
    revision: str | None = None  # HF commit sha; None = main, recorded in the manifest
    dtype: str = "bfloat16"  # serving dtype; the oracle passes float32
    device: str = "cuda"
    max_length: int = 16384
    sdp: str = "efficient"  # efficient | math | auto, see oracle.py --sdp
    dry_run: bool = False  # encode only, no weights
    expect_code_sha: str | None = None  # sha256 of joint_schema_model.py; mismatch refuses to load
    warmup: bool = False  # one throwaway forward at startup so the first request is not cold


@dataclass
class Decision:
    request_sha: str
    encoding: dict[str, Any] | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClefEngine:
    def __init__(self, cfg: EngineConfig):
        from huggingface_hub import snapshot_download

        self.cfg = cfg
        self.timeline = Timeline()
        self.timeline.mark("engine_init")
        store_root = Path(os.environ.get("MODEL_STORE_ROOT", str(MODEL_STORE_ROOT)))
        stored = resolve_model_store(cfg.model, cfg.revision, store_root)
        t_fetch = time.perf_counter()
        if stored is not None:
            self.snapshot, before, after = stored, 0, 0
        else:
            from huggingface_hub import constants

            blobs = (
                Path(constants.HF_HUB_CACHE) / ("models--" + cfg.model.replace("/", "--")) / "blobs"
            )
            before = dir_bytes(blobs)
            self.snapshot = Path(snapshot_download(cfg.model, revision=cfg.revision))
            after = dir_bytes(blobs)
        self.snapshot_source = fetch_source(stored is not None, before, after)
        self.timeline.fetch = {
            "source": self.snapshot_source,
            "bytes_added_to_cache": after - before,
            "seconds": round(time.perf_counter() - t_fetch, 3),
        }
        self.timeline.mark("snapshot_resolved")
        code_sha = hashlib.sha256((self.snapshot / CODE_FILE).read_bytes()).hexdigest()
        if cfg.expect_code_sha and code_sha != cfg.expect_code_sha:
            raise RuntimeError(
                f"{CODE_FILE} sha {code_sha[:12]} != expected {cfg.expect_code_sha[:12]}; "
                "refusing to execute unpinned model code"
            )
        sys.path.insert(0, str(self.snapshot))
        import joint_schema_model as jsm  # the checkpoint's own code
        import torch
        from transformers import AutoProcessor

        self.jsm, self.torch = jsm, torch
        self.lock = threading.Lock()
        self.timeline.mark("model_code_imported")
        t0 = time.perf_counter()
        if cfg.dry_run:
            self.model, self.processor = None, AutoProcessor.from_pretrained(self.snapshot)
        else:
            self.model, self.processor = jsm.load_release_model(
                self.snapshot, device=cfg.device, dtype=getattr(torch, cfg.dtype)
            )
        self.load_s = round(time.perf_counter() - t0, 1)
        self.timeline.mark("weights_loaded")
        self.device = torch.device(cfg.device if not cfg.dry_run else "cpu")
        self._sdp_backends = None
        if self.device.type == "cuda" and cfg.sdp != "auto":
            from torch.nn.attention import SDPBackend

            self._sdp_backends = (
                [
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                ]
                if cfg.sdp == "efficient"
                else [SDPBackend.MATH]
            )
        if self.device.type == "cuda" and cfg.sdp == "efficient":
            # transformers passes enable_gqa=True when the mask is None (Qwen3.5: 16 q heads,
            # 4 kv heads); torch's fused kernels then refuse and only the math kernel is left,
            # which materializes T×T. Forcing repeat_kv makes the fused kernels eligible.
            # Verified against transformers 5.10.2's sdpa_attention.py: use_gqa_in_sdpa → False
            # means repeat_kv, nothing else.
            import transformers.integrations.sdpa_attention as sdpa_mod

            sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False

        self.warmup_s = None
        if cfg.warmup and self.model is not None:
            self.warmup_s = self.warmup()
            self.timeline.mark("warmup_done")

    WARMUP_BODY = {"state": "warmup", "questions": {"w": {"type": "noul", "instructions": "ok?"}}}

    def warmup(self) -> float:
        """One throwaway forward: CUDA context, cuBLAS handles and lazily loaded kernels are paid
        here instead of by the first real request. Returns seconds; the answer is discarded."""
        t = time.perf_counter()
        self.decide(self.WARMUP_BODY, rid="warmup")
        return round(time.perf_counter() - t, 3)

    # -- pieces ----------------------------------------------------------------------------

    def encode(self, record: dict[str, Any]):
        return self.jsm.encode_record(
            self.processor.tokenizer,
            record,
            max_length=self.cfg.max_length,
            processor=self.processor,
        )

    def _forward(self, batch):
        if self._sdp_backends is None:
            return self.model(batch)[0]
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel(self._sdp_backends):
                return self.model(batch)[0]
        except RuntimeError as e:
            if "No available kernel" not in str(e) or self._sdp_backends == [SDPBackend.MATH]:
                raise
            with sdpa_kernel([SDPBackend.MATH]):  # last resort; OOMs at 16k in fp32
                return self.model(batch)[0]

    def forward(self, encoded) -> list[tuple[list[float], list[float]]]:
        """(logits, probs) per question, as Python lists. Batch size 1 on purpose."""
        torch = self.torch
        with torch.inference_mode():
            batch = self.jsm.collate_records(
                [encoded], self.processor.tokenizer.pad_token_id, self.device
            )
            logits = self._forward(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        out = []
        for ql in logits:
            lg = ql.float().cpu()
            out.append((lg.tolist(), lg.softmax(-1).tolist()))
        return out

    # -- the request path ------------------------------------------------------------------

    def decide(
        self, body: dict[str, Any], images: list[Any] | None = None, rid: str = ""
    ) -> Decision:
        """Answer one SystemOne request. Raises the encoder's ValueError for rejected requests
        and torch errors (OOM, kernel) from the forward pass; callers map them to records or
        HTTP statuses. Serialized by self.lock: one forward pass at a time."""
        d = Decision(request_sha=request_sha(body))
        rec = {k: v for k, v in body.items() if k != "images"}
        rec["id"] = rid
        if images:
            rec["images"] = images
        t0 = time.perf_counter()
        with self.lock:
            encoded = self.encode(rec)
            d.encoding = encoding_block(encoded)
            d.timing_ms["encode"] = round((time.perf_counter() - t0) * 1000, 2)
            if self.model is not None:
                t1 = time.perf_counter()
                per_q = self.forward(encoded)
                d.timing_ms["forward"] = round((time.perf_counter() - t1) * 1000, 2)
                d.questions, answers = assemble(
                    encoded.questions, per_q, body, self.jsm.systemone_answer
                )
                d.response = {
                    "model": self.cfg.model,
                    "answers": answers,
                    "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
                }
        d.timing_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        return d

    def manifest(self) -> dict[str, Any]:
        torch = self.torch
        import transformers

        return {
            "model": self.cfg.model,
            "revision": self.snapshot.name,  # snapshots/<commit sha>
            "snapshot_path": str(self.snapshot),
            "snapshot_source": self.snapshot_source,  # model_store | hf_cache | hf_download
            "timeline": self.timeline.as_dict(),
            "warmup_s": self.warmup_s,
            "files": file_shas(self.snapshot),
            "dtype": self.cfg.dtype,
            "device": self.cfg.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dry_run": self.cfg.dry_run,
            "max_length": self.cfg.max_length,
            "sdp": self.cfg.sdp,
            "gdn_fast_path": gdn_fast_path(),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
            },
            "worker_decide_commit": git_commit(),
            "load_s": self.load_s,
        }


def gdn_fast_path() -> bool:
    """Whether Qwen3.5 Gated DeltaNet runs the fla/causal-conv1d kernels or pure torch."""
    try:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401

        return True
    except ImportError:
        return False
