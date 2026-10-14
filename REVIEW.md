# Review guide

One section per milestone. Each says what to run, what to expect, and what is
out of scope, so an auditor (human or agent) can check the milestone without
reconstructing intent from the diff.

## Milestone 1 — Clef-flash oracle + fixtures

Goal: a reference every later serving arm is gated on. fp32 outputs from
Cloudflare's own `joint_schema_model.py` on a fixed request set, plus a parity
checker with a span gate (encoding drift) and a numeric gate (probability
drift).

Run (CPU):

```
make check
python scripts/parity.py fixtures/clef-flash/float32 fixtures/clef-flash/float32   # self-check
python scripts/parity.py fixtures/clef-flash/float32 fixtures/clef-flash/bfloat16  # dtype characterization
```

Expect:

- `make check` green (25 tests).
- Parity self-check: all 87 records PASS, 0 span mismatches, max |Δp| = 0.
- bf16 vs fp32: span gate ok on every record (same encoder, same ids). The
  numeric comparison characterizes the dtype; it is not the serving gate (see
  Results: one question sits just over the 0.02 bar). The Core AI port saw
  max |Δp| 0.011 for bf16. `fixtures/clef-flash/bfloat16` self-check PASS.
- `runs/dry` checks: long_2k 1500-3000 tokens, trunc_16k exactly 16384,
  schema_overflow errored, pulse-11714037 == 843 tokens (what Workers AI
  billed), board2048 under 4000 tokens.
- Hosted drift (`runs/parity_hosted.md`): numbers only, span n/a. Whatever
  the result, it is recorded, not judged: it measures Workers AI against the
  published weights, and the hosted `confidence` field is already known not
  to be max p (0.0973 on a 0.4016 argmax for pulse-11714037).

Audit points:

- `fixtures/requests/*.jsonl` bodies are byte-identical to what Brian's table
  sent (compare against `bench/quality.py:systemone_body` in Pulse).
- `fixtures/clef-flash/manifest.json` pins the HF revision and file shas, and
  the torch/transformers versions match the model card's tested pair.
- `scripts/parity.py` reports the span gate before the numeric gate, and
  near-ties (oracle top-2 margin ≤ 0.02) are listed, not counted as failures.
- `deciders/clef/engine.py` (shared by the oracle and the worker) calls the
  snapshot's own `joint_schema_model` functions and nothing else touches
  logits except the softmax; batch size is 1.
- The oracle branch commit was made by the pod (`worker-decide pod`), and the
  manifest's `worker_decide_commit` matches a commit on `main`.

Out of scope: the HTTP worker, Runpod endpoint template, SGLang, caching arms,
Typical backend, any change to the head.

### Results (branch `oracle/clef-flash-8b2e5fd-20261014`, H100 80 GB, Oct 14 2026)

Environment: torch 2.11.0+cu130 (host driver 580.126; cu130 needs ≥ 580, so
the serving image will use cu128), transformers 5.10.2, HF revision `8b2e5fd`.
fp32 load 11 s from the network volume; 86/87 records ok, `schema_overflow`
rejected at 475,198 schema tokens as designed. Dry-run checks 6/6.

Two kernel caveats, neither measured yet:

- SDPA ran on the memory-efficient kernel (`--sdp efficient`, with
  `use_gqa_in_sdpa` forced off so it is eligible; verified against transformers
  5.10.2 that this only switches to `repeat_kv`). Same fp32 arithmetic as the
  math kernel, different reduction order. To bound: `--sdp math` on the
  ≤ 2k-token records in the next pod run (math OOMs at 16k in fp32).
- Gated DeltaNet layers ran on the pure-torch path (`gdn_fast_path: false`, no
  `fla`/`causal-conv1d`). Whether Cloudflare's tested environment used the
  fused kernels is not stated on the model card.

- **bf16 vs fp32** (`runs/parity_bfloat16.md`): 0 span mismatches; 86/87 pass;
  one question (`img_sign448/word`) at max |Δp| 0.0211 vs the 0.02 bar, argmax
  unchanged (stop 0.511 → 0.490); mean-of-means 0.00155 (bar 0.002); 7 near-ties
  listed. Decision: fp32→bf16 is a dtype characterization. The serving gate
  for every later arm is same-dtype against this bf16 run (promoted to
  `fixtures/clef-flash/bfloat16`), with fp32 drift reported, not judged.
- **Workers AI vs fp32** (`runs/parity_hosted.md`, Pulse 50 only): hosted
  `prompt_tokens` equals the oracle's `input_tokens` on 50/50 records, so the
  request bodies and the encoder match byte for byte and the drift is
  downstream of encoding. Argmax agrees 45/50; max |Δp| median 0.043, worst
  0.1435 (pulse-10545020); mean-of-means 0.026. That is ~7× the bf16 noise
  floor, so the hosted service is not these weights through this code path
  as-is (quantization, checkpoint or serving differs; undetermined). Hosted
  `confidence` is not max p: max p − confidence ranges 0.152–0.334 over the
  50. Closest simple hypothesis is 1 − H/ln n (within 0.08); formula
  unidentified.
- **Against Haiku on the 50** (same harness as Brian's table): hosted argmax
  46% exact / MAE 0.54 (reproduces the table); fp32 oracle 56% / 0.44; bf16
  52% / 0.48; urgent P/R identical (80% / 44%, 5 flagged) across all three.
  Reproduce: `python scripts/haiku_eval.py fixtures/clef-flash/float32
  fixtures/clef-flash/bfloat16 runs/hosted` (labels in
  `fixtures/labels/haiku-pulse50.json`; the `ev` decoder rows are the
  harness's "+ev" variant).
- **Vision twins**: every image record answers correctly at 0.93–0.98 while its
  text-only twin sits near uniform (receipt total 0.947 vs 0.416, chart Q3
  0.979 vs 0.296). Quality miss, not parity: `img_board2048` reads boarding
  14:05 as after 14:30 on the downscaled image.
- Timings (fp32, batch 1): ~150 ms for 200–300 tokens, ~350 ms for 850, 6.1 s
  for 16,384; bf16 roughly 0.6× of that.


## Milestone 2 — worker shell (engine, queue + HTTP transports, image)

Goal: serve Clef-flash on Runpod with the oracle's numbers. One engine
(`deciders/clef/engine.py`) behind two transports; a parity run over HTTP
against the bf16 fixture is the gate.

Run (CPU):

```
make check                                                   # 54 tests, fake engine for the worker
python -c "import worker.app"                                 # the app imports without torch
```

Run (GPU, via `scripts/pod_http_parity.sh` on a pod from the image):

```
python scripts/parity_http.py http://127.0.0.1:8000 runs/http-bfloat16 \
    --reference fixtures/clef-flash/bfloat16 --max-dp 0.005 --also fixtures/clef-flash/float32
```

Expect:

- HTTP parity vs `fixtures/clef-flash/bfloat16`: 0 request mismatches, 0 span
  mismatches, 87/87 pass at max |Δp| 0.005 (same dtype, same kernels; on a
  different GPU type a few 1e-3 differences are plausible, a 1e-2 one is not).
  `schema_overflow` errors on both sides. The informational report vs fp32
  should look like `runs/parity_bfloat16.md` (one marginal question).
- bf16 re-check (oracle path after the engine refactor) vs the bf16 fixture:
  PASS at 0.005; on an H100 expect max |Δp| ≈ 0.
- fp32 math-kernel bound (`runs/parity_fp32_math.md`, `--subset`, trunc_16k
  excluded): whatever it is, it is recorded; the REVIEW.md caveat is replaced
  by the number.
- Image build: `image` workflow green; the build asserts torch 2.11.0+cu130
  and transformers 5.10.2.

Audit points:

- `worker/core.py` is the only place requests are interpreted; both
  transports call it, and the tests drive it with a fake engine
  (`tests/test_worker_core.py`, `tests/test_worker_app.py`).
- `worker/core.py` pins `MODEL_REVISION` and `CODE_SHA` to the oracle manifest
  (`tests/test_worker_core.py::test_pins_match_the_oracle_manifest`); the
  engine refuses to execute snapshot code whose sha differs.
- `debug` is a request-header opt-in on HTTP and an `input.debug` flag on the
  queue path; neither changes `request_sha` (popped/stripped before hashing).
- `scripts/parity_http.py` computes `request_sha` locally from the fixture, so
  a worker that hashed a different body fails the request gate.
- `Dockerfile`: pins come from `requirements-oracle.txt`; weights are not in
  the image; `CMD` not `ENTRYPOINT`.

Out of scope: dynamic batching, SGLang, caching/cold-start arms, Typical
backend, Hub listing (`.runpod/hub.json`, next).

### Results (branch `parity/http-8b2e5fd-20261014`, H100 80 GB, Oct 14 2026)

Image `ghcr.io/ashwinsreedhar28/worker-decide:clef-13a432a` on a pod with the
weights volume; worker in `TRANSPORT=http`, bf16, efficient SDPA, GDN on the
pure-torch path. Engine ready 6.3 s after start (warm volume).

- **Worker over HTTP vs `fixtures/clef-flash/bfloat16`**
  (`runs/parity_http_bfloat16.md`): **PASS**, 87/87. 0 request mismatches,
  0 span mismatches, worst question max |Δp| **0.0000** against the 0.005 bar,
  mean of record means 0.00000. `schema_overflow` errored on both sides. The
  serving path reproduces the reference exactly — same token ids, same spans,
  same probabilities — so the debug block's probabilities are the engine's, not
  a re-derivation.
- **bf16 re-check of the oracle path** (`runs/parity_bf16_recheck.md`):
  **PASS**, 87/87, max |Δp| 0.0000. The milestone-1 records survive the engine
  refactor bit for bit on the same hardware.
- **fp32 math-kernel bound** (`runs/parity_fp32_math.md`, `--subset`,
  `trunc_16k` excluded because the math kernel OOMs there): **PASS** on 86
  records, max |Δp| **0.0000** against a 0.001 bar. This closes the
  milestone-1 caveat: the memory-efficient SDPA kernel and the math kernel
  agree to the printed precision, so the oracle's kernel choice does not move
  the numbers.
- **Latency** (batch 1, H100, wall measured client-side over HTTP): median
  225 ms, p90 449 ms, max 2849 ms (`trunc_16k`, 16,384 tokens); 212 ms median
  for the 47 records under 400 tokens. HTTP + JSON overhead above the engine's
  own encode+forward is **1.5 ms median, 2.8 ms p90** — the transport is not
  the cost, the forward pass is.
- Engine identity at serve time (`runs/worker_info.json`): revision
  `8b2e5fd…`, `joint_schema_model.py` sha `0e304cf7c650` (matches the pinned
  value in `worker/core.py`), torch 2.11.0+cu130, transformers 5.10.2.

Known gaps, carried forward: the queue transport has not been exercised on a
real endpoint (only `WorkerCore.handle_job` under test); no load-balancing
endpoint has been deployed; concurrency is one forward at a time, so
throughput is 1/latency; parity is proven on Hopper (H100) only — the Hub
listing targets Ampere/Ada 48 GB pools, where bf16 kernels differ, so the
first endpoint run there should go through `parity_http.py` before the
"matches the reference" claim extends to them; the worker has not been run in
fp32 over HTTP; peak GPU memory at 16k tokens is unmeasured.
