# worker-decide

A Runpod Serverless worker for SystemOne-style decision models, with pluggable
deciders. Target models: Cloudflare Clef-flash (Apache-2.0) first, Oz Labs
Typical second. The wire format is SystemOne (`{model, state, questions}` in,
`{model, answers, usage}` out) on `POST /v1/decide` (alias `/v1/systemone`),
so a client that already speaks SystemOne changes only its base URL.

Status: milestone 1 done (Clef-flash fp32/bf16 references + parity gate);
milestone 2 in progress (worker shell: shared engine, queue and HTTP
transports, CUDA 13 image; HTTP parity run pending).

## Layout

```
deciders/clef/engine.py  ClefEngine: load once, decide(body) -> probabilities + SystemOne answers
deciders/clef/oracle.py  reference runner over the fixtures (drives the same engine)
worker/core.py           transport-agnostic request handling (validation, images, debug block)
worker/handler.py        Runpod queue endpoint entry (runpod.serverless)
worker/app.py            HTTP entry for load-balancing endpoints (FastAPI)
fixtures/requests/       SystemOne request bodies (Pulse 50, edge cases, coverage, images)
fixtures/clef-flash/     float32 (truth) and bfloat16 (serving) reference outputs + manifests
fixtures/labels/         Haiku labels for the Pulse 50 (scripts/haiku_eval.py)
scripts/                 parity.py, parity_http.py, haiku_eval.py, make_fixtures.py, pod_*.sh
tests/                   CPU-only tests (fake engine for the worker)
Dockerfile               CUDA 13.0 image; .github/workflows/image.yml publishes to GHCR
```

## Checks

```
make check     # ruff + pytest, no GPU needed
```

## The worker

One engine, two transports, picked by `TRANSPORT`:

- `queue` (default): Runpod queue endpoint. Job input is the SystemOne request
  body (plus optional `"debug": true`); output is the SystemOne response body.
  `test_input.json` is a three-question support ticket for the SDK's local run:
  `python -m worker.handler --test_input "$(cat test_input.json)"`.
- `http`: Runpod load-balancing endpoint. `POST /v1/decide` (alias
  `/v1/systemone`) takes the SystemOne body and returns the SystemOne body;
  header `X-Decide-Debug: 1` adds a `debug` block with the encoding hashes and
  full-precision probabilities (what `scripts/parity_http.py` gates on).
  `GET /ping` answers 204 while the weights load and 200 when ready, which is
  Runpod's health contract; `GET /` reports engine info.

Images go inline as base64 data URLs in `images` (at most 8; the whole body
under 25 MB). Request faults are 400 with the problem list, forward-pass
failures 500.

Environment variables (defaults baked into the image):

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

### Deploying on Runpod Serverless

Image: `ghcr.io/ashwinsreedhar28/worker-decide:clef-<sha7>` (built by the
`image` workflow on `main`; the package must be public for Runpod to pull it
without registry credentials). GPU: 24 GB is enough for bf16 at 16k tokens
with the efficient SDPA kernel; 80 GB for fp32. Hosts must be CUDA 13.0
(endpoint setting "Allowed CUDA versions"), the same filter worker-vllm uses.
Attach a network volume with the weights, or give the worker ≥ 60 GB container
disk and let the first start download them.

- Queue endpoint: default settings; the CMD is already the handler.
- Load-balancing endpoint: env `TRANSPORT=http`, expose HTTP port 8000, env
  `PORT=8000` and `PORT_HEALTH=8000`. Calls go to
  `https://<ENDPOINT_ID>.api.runpod.ai/v1/decide` with `Authorization: Bearer`.

### Gating a deployed or local worker

```
python scripts/parity_http.py http://127.0.0.1:8000 runs/http-bf16 \
    --reference fixtures/clef-flash/bfloat16 --max-dp 0.005 --also fixtures/clef-flash/float32
python scripts/parity_http.py https://<ENDPOINT_ID>.api.runpod.ai runs/http-endpoint --token $RUNPOD_API_KEY
```

`scripts/pod_http_parity.sh` runs that on a pod deployed from the worker image
(same self-driving pattern as the oracle below; results land on a
`parity/http-<rev7>-<date>` branch), plus a bf16 re-check of the oracle path
and the fp32 math-kernel bound.

## Running the oracle on a Runpod pod

The pod drives itself; nothing to SSH into. Results come back as a branch.

1. Create a fine-grained GitHub token scoped to this repo only, permission
   Contents: read and write. Store it in Runpod as a secret named `GH_TOKEN`.
   It never goes in the repo or in chat.
2. Pod: H100 80 GB (fp32 needs ~36 GB for weights plus activations), a Runpod
   PyTorch template image, a network volume attached (Runpod mounts it at
   `/workspace`; weights and the venv persist there, so a re-run skips the
   19 GB download).
3. Environment variables on the pod: `GH_TOKEN` = `{{ RUNPOD_SECRET_GH_TOKEN }}`, optional
   `REVISION` (HF commit sha to pin; default `main`, recorded in the manifest),
   optional `STAGE=dry` for a smoke test that stops after encoding.
4. Start command (the "Docker command" field):

   ```
   bash -c "git clone https://github.com/ashwinsreedhar28/worker-decide /root/worker-decide && cd /root/worker-decide && bash scripts/pod_oracle.sh"
   ```

What it does, in order: install the pinned environment, download the snapshot,
dry run (encoding only) with token-length checks, fp32 oracle over all 87
records into `fixtures/clef-flash/float32/`, bf16 pass into
`fixtures/clef-flash/bfloat16/` (the serving-dtype reference) with a parity
report in `runs/`, self-check, parity against the hosted Workers AI
responses, then commit everything to `oracle/clef-flash-<rev7>-<yyyymmdd>` and
push. Open that branch as a PR; its `runs/parity_*.md` files are the review.

Set `KEEP_ALIVE=1` to keep the pod up afterwards for inspection; otherwise
stop it once the branch appears.

See `CONTRIBUTING.md` for commit rules and `REVIEW.md` for what to audit per
milestone.
