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
  **Superseded Oct 14** (milestone 2 results, A40 run): a same-dtype gate is
  valid only on the GPU the bf16 reference was made on. Correctness is now
  judged against fp32 on any GPU; the bf16 comparison is the same-GPU
  regression gate.
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
  **Prediction falsified Oct 14:** on an A40 the worker differed from this
  H100 reference by up to 0.0291 while staying closer to fp32 than the H100
  did. See the A40 results below.
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

### Results: load-balancing endpoint on an A40 (branch `parity/http-a40-20261014`)

Image `clef-b455d5a` on a Runpod **load-balancing endpoint**, A40 only, CUDA
13.0, no network volume (the snapshot came down at first start, as on the
Hub's test endpoint). Driven from a laptop over the public URL.

- **Endpoint behavior**: `/ping` answered 204 within seconds of start and 200
  at **32 s** (13 s snapshot fetch, 6 s weight load). 87 requests: 86 × 200 and
  one 400, the 8th request, which is `schema_overflow` — the record that must
  be rejected. The worker scaled to zero 300 s after the last request, per the
  idle timeout.
- **Same-dtype gate against the H100 bf16 fixture: FAIL**, 13 of 87 records
  over 0.005 (all on |Δp|, none on argmax), worst 0.0291, mean of record means 0.00152, with 0 request and 0
  span mismatches — tokenization and spans bit-identical, so the difference is
  purely numeric, from different kernels on sm_86 vs sm_90.
- **But against the fp32 truth the A40 is the better run**
  (`python scripts/compare_runs.py fixtures/clef-flash/float32
  fixtures/clef-flash/bfloat16 runs/http-a40 --labels H100-bf16,A40-bf16`):

  | per-question max \|Δp\| | max | p99 | median | mean |
  |---|---|---|---|---|
  | H100 bf16 vs fp32 | 0.0211 | 0.0127 | 0.00108 | 0.00202 |
  | A40 bf16 vs fp32 | 0.0163 | 0.0115 | 0.00102 | 0.00172 |
  | A40 vs H100 | 0.0291 | 0.0147 | 0.00115 | 0.00221 |

  Zero clear argmax flips in any of the three comparisons (every flip sits on
  a near-tie with fp32 margin < 0.04). The A40 is closer to fp32 on 89
  questions, the H100 on 84. The two bf16 runs err in different directions:
  on `invoice_extract/overdue`, H100 is 0.0127 from fp32, A40 is 0.0163, and
  they are 0.0291 apart — nearly the sum. Two bf16 runs on different hardware
  disagree with each other more than either disagrees with the truth.

**Decision (Oct 14): two gates, with different jobs.**

- **Correctness, any GPU** — `parity_http.py`'s default. Against the fp32
  truth: argmax agrees where the fp32 margin exceeds the near-tie window
  (2 × max-dp = 0.05; 19 of 179 questions fall inside it); max |Δp| ≤ 0.025;
  mean of record means ≤ 0.0025. Both measured GPUs pass with headroom (H100
  0.0211 / 0.00155, A40 0.0163 / 0.00122). Set from two GPUs, so provisional
  until an Ada run confirms or tightens it. **Confirmed on Ada (L40S) the same
  night, with thin margin: 0.0225 / 0.00143** — see the L40S results below.
- **Regression, same GPU as the reference** — `--regression
  fixtures/clef-flash/bfloat16` at 0.005 / 0.001. Detects code changes, not
  hardware. `pod_http_parity.sh` enables it only when the pod's GPU name equals
  the fixture manifest's.

### Results: queue endpoint on an A40 (`runs/queue-a40/test_input_response.json`)

Same image (`clef-b455d5a`) on a Runpod **queue endpoint**, A40, CUDA 13.0, no
volume, default CMD (`TRANSPORT` unset). `test_input.json` submitted to `/run`
and polled on `/status`: **COMPLETED**, 36.5 s queue delay for the cold
worker, 920 ms execution (forward 824 ms — a first request, kernel warm-up
included, not a steady-state figure). Checked against the response:

- `request_sha` equals the hash of `test_input.json` with `debug` removed, so
  `handle_job` strips the flag before hashing on a real endpoint, as the unit
  test claims.
- The score answer is Σ i·pᵢ of its five probabilities (3.4265), and every
  `confidence` equals the question's max p — unlike hosted Workers AI, where
  confidence ran 0.15–0.33 below max p.
- Engine: A40, bf16, revision `8b2e5fd…`, `code_sha` equal to the pin,
  commit `b455d5a`; `output_tokens` 0.

`test_input.json` is not one of the 87 fixtures, so this is a transport check;
numerics on this path are covered by the shared `WorkerCore` and the HTTP
parity runs.

### Results: load-balancing endpoint on an L40S (branch `parity/http-l40s-20261014`)

Same image (`clef-b455d5a`), same requests, a **load-balancing endpoint on an
L40S** (Ada, sm_89), CUDA 13.0, no volume — the GPU is the only variable
against the A40 run.

- **Correctness gate: PASS**, 87/87, worst question 0.0225 (bar 0.025), mean
  of record means 0.00143, 0 request and 0 span mismatches, no argmax failure
  outside the near-tie window. The bar holds on a third architecture, but the
  headroom is 0.0025 (H100 0.0039, A40 0.0087). It is not widened for that: it
  was set as "confirm or tighten on Ada", and widening a bar after seeing data
  so that it keeps passing is what a gate exists to prevent. The distance to a
  genuinely different code path stays large — hosted Workers AI was off by
  0.1435.
- **The bar is set by one question.** Per GPU, worst question vs fp32
  (`python scripts/compare_runs.py ...`):

  | GPU | worst | 2nd | worst excluding `img_sign448/word` |
  |---|---|---|---|
  | H100 | 0.0211 `img_sign448/word` | 0.0127 `invoice_extract/overdue` | 0.0127 |
  | A40 | 0.0163 `invoice_extract/overdue` | 0.0115 `img_sign448/word` | 0.0163 |
  | L40S | 0.0225 `img_sign448/word` | 0.0133 `pulse-9707528/urgency` | 0.0133 |

  Two of three GPUs land closest to the bar on the same question — one word
  read off a 448-px rendered image — and without it nothing on any GPU passes
  0.0163. If a future GPU crosses 0.025, look at `img_sign448/word` before
  the GPU.
- Across all three, zero clear argmax flips in any pairwise comparison, and
  every pair of bf16 runs disagrees with each other by more than either
  disagrees with fp32 (A40 vs L40S 0.0295, A40 vs H100 0.0291).
- **Cold start, read from the worker log**: ~90 s from the first `/ping` to the
  worker process starting (scheduling onto an L40S host, image pull, container
  create — this endpoint's first worker ever), then ~29 s in-process (15 s
  snapshot fetch, 3 s weight load) to `/ping` 200. In-process time matches the
  A40's 32 s; the difference is entirely the platform segment. This is what
  milestone 3's driver now measures per sample.

Known gaps, carried forward: concurrency is one forward at a time, so
throughput is 1/latency; correctness is measured on the
H100 (sm_90), A40 (sm_86) and L40S (sm_89) only — the listing's A100-class
pool (sm_80) has not been gated yet; the worker has not been run in fp32 over
HTTP; peak GPU memory at 16k tokens is unmeasured; and the 13 s snapshot fetch
was either a fast link or a host-side cache, undetermined (milestone 3).

## Milestone 3 — cold-start benchmark

Goal: time-to-first-answer for a cold Clef-flash worker on Runpod, split into
where the time goes, across the ways weights reach the GPU. Every arm keeps
passing the correctness gate.

Arms (A40 queue endpoint, reconfigured between arms): **A** container disk
(download at start, billed); **B** Model Store (host cache, not billed);
**C** network volume (`clef-oracle`, pins the endpoint to US-CA-2); **D**
FlashBoot on A or B. The original plan's compile-cache arm is dropped: the
transformers path runs eagerly, so there is nothing to cache (it belongs to the
SGLang arm, milestone 4).

What the Oct 14 runs already showed (hand-read from logs): the L40S cold start
took ~2 min, ~90 s of it before the worker process existed (scheduling, image
pull, container create) and ~29 s inside it; the A40 took 32 s in-process.
Two hosts fetched the 19 GB snapshot in 13 s and 15 s, which points to a fast
link rather than a host cache — the timeline's bytes count settles it. So the
benchmark has to measure outside the process, not only inside it.

Run:

```
make check
python scripts/coldstart.py <ENDPOINT_ID> runs/coldstart-<arm>-<gpu> --runs 10 --label <arm>-<gpu>
```

(endpoint idle timeout 5 s during the benchmark; `$RUNPOD_API_KEY` set; the
image must include the timeline, i.e. be built from this milestone's code).

Audit points:

- `resolve_model_store` serves only the pinned revision and raises naming the
  revisions present otherwise; the CODE_SHA check still runs after it.
- The timeline is measured from process start (`/proc`), labelled
  `engine_init` where `/proc` is absent. `platform_est_s` (Runpod's delayTime
  minus the worker's process-start → ready) is an estimate and is labelled so.
- "Cold" means no running or initializing worker for two consecutive /health
  readings. Idle slots (scaled down, not billed) count as cold, but their count
  is recorded per sample, because a start onto one may skip the image pull. A
  first start on a fresh host is measured once per new endpoint and labelled.
- Cold samples under 2 s of queue delay are flagged `suspect_warm` and excluded
  from the summary.
- `WARMUP` is off by default until the cold-vs-warm execution times show it
  pays for itself.

Out of scope: SGLang/CUDA-graph caching, lazy image loading (eStargz, SOCI,
Nydus), concurrency, cost per 1k decisions (milestone 5).
