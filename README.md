# worker-decide

Answer classification, yes/no and ordinal-scoring questions about text, JSON or images with
[Cloudflare Clef-flash](https://huggingface.co/Cloudflare/clef-flash) on Runpod Serverless. It
speaks the SystemOne wire format: `{model, state, questions}` in, `{model, answers, usage}` out,
on `POST /v1/decide` (alias `/v1/systemone`). A client that already speaks SystemOne changes
only its base URL. Weights and model code are pinned, and every answer path is gated against
an fp32 reference within measured bf16 tolerance.

[![RunPod](https://api.runpod.io/badge/ashwinsreedhar28/worker-decide)](https://www.runpod.io/console/hub/ashwinsreedhar28/worker-decide)

---

## Status

- Milestones 1–4 are done: fp32/bf16 references and the parity gate; the worker shell with
  queue and HTTP transports and a CUDA 13 image; cold-start measurements; the SGLang
  comparison.
- Optimization tiers 0 (fla kernels) and 1 (dynamic batching, off by default) are merged and
  measured on the A40 and the H100.
- [REVIEW.md](REVIEW.md) has every result, the audits, and what each finding rests on.

## Deploy

### From the image

Image: `ghcr.io/ashwinsreedhar28/worker-decide:clef-<sha7>` (built by the `image` workflow;
`:v0.1.0` for the release). Hosts must allow CUDA 13.0 (endpoint setting "Allowed CUDA
versions"), the same filter worker-vllm uses.

- **GPU:** 48 GB for bf16 (A40, A6000, L40/L40S); 80 GB for fp32. The weights are 19.1 GB.
  Peak memory at 16k tokens has not been measured, so 24 GB is untested.
- **Queue endpoint:** default settings; the CMD is already the handler.
- **Load-balancing endpoint:** set `TRANSPORT=http`, expose HTTP port 8000, and set
  `PORT=8000` and `PORT_HEALTH=8000`. Calls go to
  `https://<ENDPOINT_ID>.api.runpod.ai/v1/decide` with `Authorization: Bearer`.

Getting the weights to the worker, fastest first (A40 medians, REVIEW.md milestone 3):

- **Model Store (recommended):** set the endpoint's cached model to `Cloudflare/clef-flash`.
  - The worker finds it under `/runpod-volume/huggingface-cache/hub` and skips the fetch
    (in-process ready 16 s vs 28 s from container disk).
  - The console caches the repo's `main` and cannot pin a revision. The worker serves only its
    pinned `MODEL_REVISION` (`8b2e5fd`, which is `main` today).
  - If upstream `main` moves, the worker fails at startup with an error naming the revisions
    present, rather than serve weights the parity gates never saw. Clear the cached model (the
    worker then downloads its pinned revision) or re-pin after a new parity run.
  - A host without the cached model downloads the pinned revision (2 of 10 runs missed the
    store).
- **FlashBoot on top:** a job within about 5 minutes of the last one usually ran in the kept
  process (4 of 5, 0.5 s queue delay). After 30+ minutes idle it was a cold start in 6 of 6.
- **Network volume** with the weights under `HF_HOME` (`/runpod-volume/.cache/huggingface`): no
  download, but slower and more variable loads (median 12.9 s, two of nine at about 38 s), and
  the endpoint is pinned to the volume's data center.
- **Container disk** (≥ 60 GB): every cold start downloads 19 GB (about 14 s at 1.4 GB/s); disk
  does not survive scale-to-zero.

### Endpoint configuration

Defaults are baked into the image.

| var | default | meaning |
|---|---|---|
| `MODEL_NAME` | `Cloudflare/clef-flash` | HF repo |
| `MODEL_REVISION` | `8b2e5fd1…` (the oracle's) | HF commit sha; empty = `main` |
| `CODE_SHA` | the oracle's `joint_schema_model.py` sha256 | refuse to load if the snapshot's code differs; empty disables |
| `DTYPE` | `bfloat16` | serving dtype (`float32` matches the truth reference) |
| `MAX_LENGTH` | `16384` | encoder truncation |
| `SDP` | `efficient` | SDPA kernel set, see `deciders/clef/oracle.py --sdp` |
| `TRANSPORT` | `queue` | `queue` or `http` |
| `PORT`, `PORT_HEALTH` | `8000` | HTTP transport; set both to the same value on the endpoint |
| `HF_HOME` | `/runpod-volume/.cache/huggingface` | weights cache (network volume) |
| `WARMUP` | `0` | `1` runs one throwaway forward at startup so the first request is not cold |
| `BATCH_MAX` | `1` | requests that may share one forward pass (`worker/batcher.py`); `1` = one at a time. On the queue transport it is also the number of jobs taken at once |
| `BATCH_MAX_TOKENS` | `32768` | cap on longest × count in a batch (the reference pads to the longest) |
| `BATCH_MAX_RATIO` | `2.0` | requests batch together only if their lengths are within this ratio of the oldest's |
| `BATCH_MAX_WAIT_MS` | `0` | how long to hold a request for more to arrive; `0` batches only what queued during the previous forward |
| `BATCH_TIMEOUT_S` | `600` | a request waiting longer than this for its batch is answered 500 |
| `FUSED` | empty | experimental tier 2 kernel fusions (`deciders/clef/fused.py`): `all`, or a comma list of `rmsnorm`, `mlp`, `gdn`, `conv`, `nocache`, `cublaslt`; an unknown name fails at startup. Not yet checked on a GPU |

Weights are found in this order: the Runpod Model Store if it holds the pinned revision (a
different revision there fails loudly), then the HF cache under `HF_HOME`, then a download.
Every worker reports how its snapshot arrived, and a startup timeline (seconds from process
start to snapshot, imports, weights, warm-up, ready), in `GET /` and the debug block.

## Usage

Request (the model card's example; `fixtures/requests/edge.jsonl`, `card_trio`):

```json
{
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]}
  }
}
```

Response (bf16, `fixtures/clef-flash/bfloat16/card_trio.json`):

```json
{
  "model": "Cloudflare/clef-flash",
  "answers": {
    "department": {"type": "choice", "choice": "technical", "confidence": 0.961,
                   "probabilities": {"billing": 0.039, "technical": 0.961}},
    "outage": {"type": "noul", "noul": 0.7763},
    "urgency": {"type": "score", "score": 1.8116, "confidence": 0.8803,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.0687, "1": 0.051, "2": 0.8803}}
  },
  "usage": {"input_tokens": 300, "output_tokens": 0}
}
```

- **Queue transport:** the job input is the request body (plus optional `"debug": true`), and
  the output is the response body. `test_input.json` is a three-question support ticket for
  the SDK's local run: `python -m worker.handler --test_input "$(cat test_input.json)"`.
- **HTTP transport:** header `X-Decide-Debug: 1` adds a `debug` block with the encoding hashes,
  full-precision probabilities and timings (what `scripts/parity_http.py` gates on). `GET /ping`
  answers 204 while the weights load and 200 when ready (Runpod's health contract). `GET /`
  reports engine info.
- **Images:** they go inline as base64 data URLs in `images` (at most 8; the whole body under
  25 MB).
- **Errors:** request faults are 400 with the problem list; forward-pass failures are 500.
- **Question order:** it changes Clef's answers (REVIEW.md milestone 4), so send questions in a
  fixed order. The worker preserves it.

## Performance

A40, HTTP, same requests and harness (REVIEW.md milestone 5).

| workload | tokens | c=1 p50 (worker / SGLang) | c=32 req/s (worker, `BATCH_MAX=8` / SGLang) |
|---|---|---|---|
| ticket | 300 | 94.9 / 80.3 ms | 15.39 / 22.02 |
| pulse | 843 | 190.1 / 155.3 ms | 6.31 / 8.14 |
| article | 2,486 | 497.0 / 403.1 ms | 2.22 / 2.75 |

- On the A40 this is 70–81% of SGLang's throughput at 32 concurrent requests, up from
  1.19–4.27 req/s on the pure-torch path.
- On the H100 the gap is larger (the worker without batching was at 15–56% of SGLang under
  load); batching numbers there are still to be measured.
- A first request after a cold start pays Triton compilation (about 34 s, plus about 10 s per
  new 2,048-token band) unless the kernel cache is persisted.

## Correctness

Two gates with different jobs:

- **Correctness, any GPU** (the default): against the fp32 truth, argmax agrees outside the
  near-tie window, max |Δp| ≤ 0.025, mean ≤ 0.0025.
  - Set from measured bf16 noise on an H100 (worst 0.0211) and an A40 (0.0163).
  - Confirmed on an L40S (0.0225, a thin margin set by one question).
  - The fla and batched paths pass it too; SGLang passes it on both GPUs.
- **Regression, same GPU as the reference** (`--regression`): against a bf16 run made on the
  same GPU, at 0.005. It catches code changes in the same code path.
  - Don't point it at a reference from different hardware: two bf16 runs on different GPUs
    disagree by up to about 0.03, while each stays within the correctness bar.
  - Don't use it for batched vs single either: batching changes matrix shapes, and answers
    moved by up to 0.023 on the A40.

```
python scripts/parity_http.py https://<ENDPOINT_ID>.api.runpod.ai runs/http-endpoint --token "$RUNPOD_API_KEY"
python scripts/parity_http.py http://127.0.0.1:8000 runs/http-h100 --regression fixtures/clef-flash/bfloat16
python scripts/parity_http.py http://127.0.0.1:8000 runs/http-batched --concurrency 8   # batched answers
python scripts/compare_runs.py fixtures/clef-flash/float32 runs/http-a40 runs/http-h100
```

The run's `manifest.json` records the worker's GPU. `compare_runs.py` puts two runs side by
side against the truth when they disagree, to tell a wrong run from two noisy ones.

## Development

```
make check     # ruff + pytest, no GPU needed (CI runs it on Python 3.11)
```

```
deciders/clef/engine.py  ClefEngine: load once, decide(body) -> probabilities + SystemOne answers;
                         prepare/forward_batch/finish for batching
deciders/clef/oracle.py  reference runner over the fixtures (drives the same engine)
deciders/typical/        Oz Labs' Typical behind the same interface (its image is not built yet)
worker/core.py           transport-agnostic request handling (validation, images, debug block)
worker/batcher.py        dynamic batching (BATCH_MAX)
worker/handler.py        Runpod queue endpoint entry (runpod.serverless)
worker/app.py            HTTP entry for load-balancing endpoints (FastAPI)
fixtures/requests/       SystemOne request bodies (Pulse 50, edge cases, coverage, images, order)
fixtures/clef-flash/     float32 (truth) and bfloat16 (serving) reference outputs + manifests
scripts/                 parity, load test, profiling, cold start, shape probe, cost, pod_*.sh
tests/                   CPU-only tests (fake engines)
```

GPU work runs as self-driving pod sessions. Deploy a pod from the worker image with env
`POD_SCRIPT=scripts/<script>.sh`, `REPO_REF=<branch>` (default `main`) and
`GH_TOKEN={{ RUNPOD_SECRET_GH_TOKEN }}`. The image clones the repo, runs the script, and pushes
results to a branch; review them as a PR. Each script's header documents its steps:

| script | session |
|---|---|
| `pod_oracle.sh` | fp32 and bf16 references for the fixtures (PyTorch template, H100) |
| `pod_http_parity.sh` | the worker over HTTP through the correctness and regression gates |
| `pod_m4.sh` | profile, load test, order oracle, SGLang's gate and load test on the same GPU |
| `pod_tier0.sh` | fla kernels: kernel check, first-request cost, gate, load, profile |
| `pod_shapes.sh` | per-token-count cost probe and batch preview |
| `pod_tier1.sh` | batching: batched gate, batch report, batched vs single, load |
| `pod_chain.sh` | several of the above in one session (`POD_CHAIN`) |

- **The token:** `GH_TOKEN` is a fine-grained GitHub token scoped to this repo only (Contents:
  read and write), stored as a Runpod secret. It never goes in the repo or in chat.
- **Pushing:** the scripts commit and push only on a Runpod pod (`RUNPOD_POD_ID` set). They stay
  open after pushing unless `EXIT_WHEN_DONE=1`, so stop the pod once the branch appears.
- **`pod_oracle.sh` predates the image:** it runs on a Runpod PyTorch template with a network
  volume at `/workspace` (weights and venv persist), with the start command
  `bash -c "git clone https://github.com/ashwinsreedhar28/worker-decide /root/worker-decide && cd /root/worker-decide && bash scripts/pod_oracle.sh"`
  and optional `REVISION` / `STAGE=dry`.

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for commit rules and
[REVIEW.md](REVIEW.md) for what to audit per milestone.

## License

worker-decide is released under the [MIT License](LICENSE).

It downloads and runs third-party models and libraries under their own licenses. None of them
are redistributed in this repository:

- Cloudflare Clef-flash weights and model code: Apache-2.0, fetched from Hugging Face at a
  pinned revision at runtime.
- PyTorch (BSD-3-Clause), transformers (Apache-2.0), Runpod SDK (MIT), and fla-core (MIT,
  tier 0).
- SGLang (Apache-2.0): only in the comparison pod script.
- typical-ai (Apache-2.0): for the Typical decider.
