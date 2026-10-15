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
(download at start); **B** Model Store (Runpod's host cache; per Runpod's docs
the download is not billed); **C** network volume (planned on `clef-oracle` in
US-CA-2, which had no A40 capacity on Oct 14, so a new volume `clef-coldstart`
in CA-MTL-1); **D** FlashBoot on B. The original plan's compile-cache arm is dropped: the
transformers path runs eagerly, so there is nothing to cache (it belongs to the
SGLang arm, milestone 4).

What the Oct 14 runs already showed (hand-read from console logs, not in the
repo): the L40S cold start
took ~2 min, ~90 s of it before the worker process existed (scheduling, image
pull, container create) and ~29 s inside it; the A40 took 32 s in-process.
Two hosts fetched the 19 GB snapshot in 13 s and 15 s; whether that was a
download or a host cache was open until arm A v2 counted the bytes (a
download, below). So the benchmark has to measure outside the process, not
only inside it.

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
  readings (in the driver since its first version). Idle slots (scaled down,
  not billed) count as cold, but their count is recorded per sample, because a
  start onto one may skip the image pull. A first start on a fresh host was not
  measured (see caveats).
- Each sample is classified from its own record: `cold` (platform estimate
  ≥ 0, a process started after the submit), `resumed` (< 0: the job ran in a
  process that predates the submit, which is FlashBoot), `suspect_warm` (under
  2 s with no timeline to tell), `failed`. Resumed samples get their own table
  and never enter cold statistics. Old run files are re-classified on
  `--summarize`.
- `process_age_s` = client submit time − the worker's `origin_unix`. It mixes
  two clocks; on cold samples it reads −2 to −5 s against a platform estimate
  of 3–6 s, so the skew is about a second, small next to the minutes it
  measures. It is not derivable from the marks (ready is process start →
  ready, not → submit); an earlier draft did that and was caught before merge.
- `WARMUP` is off by default until the cold-vs-warm execution times show it
  pays for itself.

Out of scope: SGLang/CUDA-graph caching, lazy image loading (eStargz, SOCI,
Nydus), concurrency, cost per 1k decisions (milestone 5).

### Results (A40 queue endpoint `7bmbtjrmxdedjo`, idle timeout 5 s, Oct 14–15 2026)

Data: `runs/coldstart-*/` (per-run JSON, `summary.json`, `summary.md`; Runpod's
raw responses are kept from A v2 on, not in A v1 or B, which predate that);
re-derive any table with `python scripts/coldstart.py X runs/<dir> --summarize`.
Every sample in every arm ran on an A40. Images `e30b28e` and `acf2fc4` are
confirmed by the responses; `f37ce55` for A v1 and B is the endpoint
configuration at the time, not recorded in those files. Medians, with min–max
where the spread matters.

| arm (image) | n | queue delay | platform | in-process ready | fetch | weight load |
|---|---|---|---|---|---|---|
| A disk v2 (`e30b28e`) | 10 | 33.8 s (27.4–42.3) | 4.3 s | 27.8 s | 13.7 s (10.4–19.3), 19.08 GB written every run | 7.7 s |
| B Model Store (`f37ce55`) | 10 | 25.5 s (18.3–73.8) | 5.7 s | 16.2 s | 0 on 8/10 | 9.6 s |
| C network volume, runs 2–10 (`e30b28e`) | 9 | 26.8 s (17.8–48.7) | 5.9 s | 19.4 s | 0.06–1.2 s, 0 bytes (`hf_cache`) | 12.9 s (8.7–38.6) |
| D Model Store + FlashBoot, cold (`e30b28e`) | 2 | 20.1 s | 5.9 s | 14.2 s | 0 | 8.8 s |
| D, resumed | 8 | **0.54 s** (0.49–1.18) | — | — | — | — |
| Sweep, cold (`acf2fc4`) | 10 | 17.9 s (16.0–24.1) | 3.9 s | 13.9 s | 0 | — |
| Sweep, resumed | 5 | **0.54 s** (0.53–1.31) | — | — | — | — |

Per-arm medians: startup (Python, worker imports) 1.7–1.8 s, torch /
transformers / model-code import 3.6–4.0 s, first execution on a cold worker
0.88–1.01 s (all samples 0.68–1.21 s), the next execution 0.32–0.43 s.

**A, container disk: every cold start is a real 19 GB download.** All ten runs
wrote 19.08 GB at 1.4 GB/s (0.99–1.84). Only three workers served the ten runs,
so workers that had run before downloaded again: container disk does not
survive scale-to-zero. An earlier A run on `f37ce55` (`runs/coldstart-disk-a40`)
gave ready 24.6 s with the same configuration; its fetch metric (blob-directory
growth, v1) saw ~130 KB and cannot tell download from cache, which is why A was
re-run. The two A runs differ by 3.2 s in ready time, 2.5 s of it in the fetch;
that mixes run-to-run noise with v2's added byte counting inside the fetch
segment, so ~3 s is an upper bound on the noise to read comparisons against,
not a measurement of it.

**B, Model Store: the fetch disappears.** Ready 16.2 s vs 27.8 s (A v2) or
24.6 s (A v1, same image as B). Weight load was 9.6 s here vs 7.7 s from
disk, but the same Model Store path loaded in 8.8 s (D) and 8.2 s (sweep), so
that difference is within noise. Two misses, runs 1 and 10, were the same
worker (`uv5zckwkzigm6y`): a host without the cached model, despite Runpod
placing workers on hosts that hold it. Their fetches took 12.2 s and 10.0 s,
in line with A v2's downloads, so the worker most likely downloaded; the v1
metric cannot confirm it. Either way a miss cost time, not a failure. Platform
outliers (57.4 s, 27.7 s) are outside the process (scheduling, image pull or
container create; the estimate cannot separate them); one was on a host
(`96rly022s0nsmu`) that was also slow in A v1.

**C, network volume: bimodal.** Run 1 populated the empty volume (19.08 GB at
1.07 GB/s, ready 69.9 s). Runs 2–10 loaded from it in 8.7–16.1 s, except two
at ~38 s (runs 4 and 6). Run 6 was the first load on worker
`mpwslasetesbbf` (its next, run 10, took 8.7 s), which fits a cold first read
on a host. Run 4 does not: it was the third load on worker `8ov76kk0xceeu6`,
whose first load (run 2) took 16.1 s. So first reads explain at most one of
the two; for run 4, page-cache eviction or volume-backend contention would both
fit, and these files cannot tell them apart. The volume also pins the endpoint to one data center.

**D, FlashBoot: a resumed process answers in about half a second.** With
FlashBoot on, 8 of 10 jobs submitted after `/health` reported no running worker
(the driver waits for two consecutive such readings; these run files keep only
the last one) ran in the process started for run 1 or 2: identical ready
marks, platform estimate ≈ −13.5 s, delay 0.49–1.18 s. The two true cold starts
(20.1 s vs B's 25.5 s) are n = 2 and say nothing either way about the cold
path; what FlashBoot does is avoid the cold start.

**How long FlashBoot keeps a process (gap sweep, `--gap`, 3 runs per gap).**
One worker (`upcd8zneb1pez2`) served all 15 iterations (30 jobs, cold + warm)
over 5.7 hours, 00:56–06:35 UTC Oct 15.

| idle before submit (since the previous job finished) | resumed / runs | process age when resumed |
|---|---|---|
| 76–331 s | 4/5 | 100, 189, 348, 497 s |
| 906–921 s | 1/3 | 938 s |
| 1806–3631 s | 0/6 | — |

(The first iteration after the image change, with no earlier process, is left
out.) There is no clean cutoff: a 305 s idle once ended in a cold start, with
a process that would have been 805 s old, while a process aged 938 s after
921 s idle was resumed. The platform reclaims kept processes for reasons a
client cannot see; idle time only moves the odds. Resumed first executions
were 0.30–0.40 s in the sweep, as fast as warm ones, but 3 of 8 in arm D took
1.27–1.33 s, slower than a cold first execution: a resume usually, not always,
skips the warm-up.
Cold starts in the sweep (17.9 s delay, 13.9 s ready) are faster than B's; a
single worker in a quiet period is a plausible reason, so read them as a lower
bound, not a new baseline.

**What this means for a deployment.** Model Store is the default: lowest
median ready time with no volume and no region pin, and on a host without the
model the worker downloads instead. FlashBoot on top usually (4 of 5 here)
turns a job arriving within ~5 minutes of the last one into a sub-second start;
its effect on the cold path and its billing were not measured. A network volume
only pays if the same hosts keep serving and its slow tail is tolerable;
container disk downloads 19 GB on every cold start.

**Caveats.** Samples per row are in the table (10 per arm, of which C reports
9 and D splits 2 cold / 8 resumed; 3 per gap); one GPU type, one model, queue
transport only, a 5 s idle timeout. B ran on an image that predates the byte
counter (its `model_store` label comes from the path check and stands; its two
misses are known only as "not the Model Store"). Not measured: first start on
a fresh host (image pull, ~90 s on the Oct 14 L40S), billing for FlashBoot or
idle workers, other GPUs.

Not built, noted: overlapping the torch / transformers import (3.8 s) with the
fetch would take it off the critical path on A; on B and D, where the fetch is
0, it would not help.


## Milestone 4 — SGLang comparison, and milestone 5a — where our time goes

Goal: find out whether SGLang computes Clef-flash (the correctness gate against the fp32
truth), how fast it is on the same GPU and inputs (the backbone floor any correct
implementation sits above), and where our own request time goes. Those three numbers set the
optimization target. The order after this milestone (decided Oct 15): profile → target
(SGLang's time + a lean head, on the A40, with an H100 number for context) → optimize in tiers,
each through the correctness gate (vectorized head, then CUDA graphs / compile, then batching
concurrent requests) → latency/throughput/cost tables on the optimized worker (5b).

What SGLang is, verified from its source (main 5b12ec1, and the pinned commit 37ae292e6f):
- Clef support (`joint_schema_head.py`, `systemone/joint_schema.py`) merged after v0.5.21 in
  sgl-project/sglang#42721; the cookbook pins nightly `0.5.22.dev20261009+g37ae292e6f`
  (cu130, Python 3.10, torch 2.14.1) and image `lmsysorg/sglang:dev-clef`. Release 0.5.21
  answers `/v1/systemone` through vocabulary label tokens, which is not Clef's head.
- For Clef it disables CUDA graphs, the radix cache and chunked prefill; the head is batched
  (`forward_batch`). Speed comes from batching and kernels, not graphs.
- `confidence` uses System One adapter formulas, not max p, so only probabilities are compared.
  It refuses a noul with blank instructions and no criteria (422); the reference answers
  `empty_instructions` from the question id. That record is SKIP, not FAIL.
- Published accuracy is GSM8K only (Clef Flash 67.3% on H200). No per-probability comparison
  with the reference has been published; this is the first.

Run (one pod from the worker image, `POD_SCRIPT=scripts/pod_m4.sh`, A40, ≥ 80 GB container
disk, no volume; pushes `m4/pod-<gpu>-<date>`):

```
make check
python scripts/parity_sglang.py http://127.0.0.1:30000 runs/sglang-a40      # gate
python scripts/loadtest.py http://127.0.0.1:8000 runs/load-ours-a40          # either engine
python scripts/profile_engine.py runs/profile-ours-a40 --trace               # in-process split
```

Fixture finding (Oct 15): `make_fixtures.write_jsonl` writes with `sort_keys=True`, so on disk
`unsorted_choice` is sorted and card_trio's questions are alphabetical; no committed fixture
exercises a non-alphabetical order. All parity results stand (the oracle and every run saw the
same bodies). `fixtures/requests-order/order.jsonl` (written order-preserving) covers options
out of order and three question orders, with same-run references; its oracle is made on the
pod (`fixtures/clef-flash/float32-order`, A40 fp32).

Audit points:
- `parity_sglang` maps choice/score probabilities by option id in the head's order (sorted
  ids), rebuilds noul from p(true), and has no encoding block: `usage.input_tokens` against the
  oracle's input_tokens stands in for the span gate on the 85 records both answer. A server that
  permuted questions would be caught only through the numbers (request_sha sorts keys).
- `loadtest` is closed-loop; nearest-rank percentiles; exits non-zero on a failed preflight or
  any errored cell. It adds a `model` field only when asked (SGLang requires one).
- `profile_engine` splits by CUDA events around the outermost hooked backbone module that ran,
  falling back to first→last decoder layer; it does not time image decode, assemble or HTTP.
  Our image runs Gated DeltaNet in pure torch (`gdn_fast_path=False`), so backbone shares are
  higher here than on SGLang's Triton path.
- `parity.compare` fails when zero records were compared and ignores non-record JSON files.
- Pod scripts commit and push only on a Runpod pod (`RUNPOD_POD_ID`): a local smoke test in a
  workspace with a GitHub token pushed a junk branch on Oct 15.

Independent audit (Oct 15, before any GPU run): 19 findings on the first draft (order findings
written into the oracle directory, empty oracle gating as PASS, no gcc for Triton in the image,
the SGLang-refused record failing the gate, loadtest always exiting 0, six surviving test
mutations, among others), all fixed and re-verified; the re-audit's four follow-ups (hung-server
guard, oracle error records, loadtest exit test, profile header) fixed too.

### Results so far: H100 pod session, Oct 15 (`m4/pod-h100-80gb-hbm3-20261015`)

A40 capacity was out, so this first session ran on an H100 SXM (the planned "context"
number). Steps 1–3 completed; SGLang installed with Clef's joint head present but did not
start: its JIT kernels call `ninja` by name and the venv's `bin` was not on PATH (fixed in
`a000805`; SGLang step to be rerun).

**Question order changes Clef's answers; option order does not** (fp32, one run, one GPU).
- Options out of order on the wire encode to the same token ids as sorted and give identical
  probabilities (Δp 0): the encoder sorts choice ids, as the reference documents.
- The model card's request in three question orders, same 300 tokens:

  | question order | department p(technical) | outage p(true) | urgency p(Today) |
  |---|---|---|---|
  | alphabetical (department, outage, urgency) | 0.9606 | 0.7774 | 0.8795 |
  | card (department, urgency, outage) | 0.9608 | 0.8363 | 0.8582 |
  | reversed (outage, urgency, department) | 0.9494 | 0.6238 | 0.9057 |

  Outage moves by up to 0.154, six times the 0.025 correctness bar, from order alone. The
  questions share one prompt and attend to each other, so this is the model, not noise: the
  alphabetical row reproduces the committed H100 fp32 card_trio fixture exactly. Implications:
  a client should send questions in a fixed order, and any layer that reorders them (sorting
  keys, as our own fixture writer did) changes the answers. Our worker preserves order: its
  bf16 answers on the order records pass the gate against this truth (worst 0.0032).

**Where our request time goes** (bf16, `gdn_fast_path=False`, median of 20 passes):

| record | tokens | encode | backbone | head | total | backbone share |
|---|---|---|---|---|---|---|
| card_trio | 300 | 1.8 ms | 128.6 ms | 4.6 ms | 135.4 ms | 95% |
| pulse-11714037 | 843 | 2.3 ms | 219.2 ms | 4.5 ms | 226.7 ms | 97% |
| long_2k | 2,486 | 5.2 ms | 353.0 ms | 5.2 ms | 364.4 ms | 97% |

- The head plus everything that isn't the backbone is 3–5% of request time, under the plan's
  15% bar for native head work: milestone 8a/b (vectorized or fused head) is not justified.
- The backbone is the target. The long_2k trace has about 197 ms of kernel time inside a
  353 ms backbone (the GPU idles ~44% of it), spread over thousands of small kernels (≈4,750
  `bmm`, ≈5,800 `mul`, ≈6,400 `copy` calls) plus fp32 GEMMs. That is consistent with
  transformers' pure-torch Gated DeltaNet fallback, which the image uses because
  flash-linear-attention and causal-conv1d are not installed. First optimization tier:
  enable that fast path, through the correctness gate.
- The split timed `language_model.model.language_model` (7.94B of 9.53B parameters; the
  outer wrappers are hooked but not entered through their own forward), as designed.

**Our worker under load** (HTTP, one forward at a time):

| workload | tokens | c=1 req/s, p50 | c=8 req/s, p50 | c=32 req/s, p50 |
|---|---|---|---|---|
| ticket (card_trio) | 300 | 7.13, 138 ms | 6.51, 1,219 ms | 6.21, 5,145 ms |
| pulse | 843 | 5.63, 176 ms | 5.22, 1,535 ms | 5.14, 6,201 ms |
| article (long_2k) | 2,486 | 3.50, 284 ms | 3.33, 2,370 ms | 3.10, 9,834 ms |

Concurrency adds queueing, not throughput, as expected from the engine lock. HTTP and
transport add about 2.4 ms (c=1 p50 138 ms vs the in-process 135 ms). Second tier: batch
concurrent requests.

Revised optimization order from these numbers: (0) GDN fast path, (1) batching, then CUDA
graphs / compile if the gap to SGLang's floor remains; the native head drops out.

### Results: SGLang on the H100, Oct 15 (`m4/pod-h100-sglang-20261015`)

SGLang `0.5.22.dev20261009+g37ae292e6f` (the cookbook's pin, Clef's joint head present),
default attention backend, after the PATH fix.

**SGLang computes Clef-flash.** Correctness gate against the fp32 truth: PASS on all 87
records (worst question 0.0159, mean of record means 0.00138, 0 clear argmax flips;
`empty_instructions` refused with 422 as documented and reported SKIP; `schema_overflow`
refused like the reference). Token counts equal the reference encoder's on all 85 records
both answer, so the prompt is the same. The order records pass too (worst 0.0060): it keeps
question order. Against our own bf16 run on the same GPU it differs by up to 0.0286
(`invoice_extract/overdue`) while each is within the bar of fp32 (ours 0.0211, SGLang
0.0159): two bf16 implementations erring in different directions, as two GPUs did in
milestone 2. SGLang is closer to fp32 on 108 of 178 questions, ours on 70.

**SGLang does not reproduce Workers AI.** On the 50 Pulse questions: SGLang vs fp32 worst
0.0082, Workers AI vs fp32 worst 0.1435, SGLang vs Workers AI worst 0.1374; SGLang is closer to
fp32 on all 50, and Workers AI has 2 clear argmax flips that SGLang does not. Cloudflare says
it moved Clef to SGLang on Workers AI (Oct 9 post; it does not say so for Clef-flash). Running
the published weights in the published SGLang build does not explain the hosted drift, so the
open question for Brian stands, sharper: the encoder and token counts match, the engine is
ruled out, what else differs (precision, a different revision, request handling)?

**SGLang is the speed target** (H100, same requests, same harness):

| workload | tokens | ours c=1 p50 | SGLang c=1 p50 | ours c=32 req/s | SGLang c=32 req/s |
|---|---|---|---|---|---|
| ticket | 300 | 138 ms | 60 ms | 6.2 | 95.7 |
| pulse | 843 | 176 ms | 62 ms | 5.1 | 45.9 |
| article | 2,486 | 284 ms | 77 ms | 3.1 | 16.4 |

2.3–3.7x lower latency per request, 5–15x the throughput under load (it batches concurrent
requests; ours serializes). Its single-request latency grows only from 60 to 77 ms between
300 and 2,486 tokens, where ours grows from 138 to 284 ms: consistent with our backbone time
being dominated by the pure-torch Gated DeltaNet path rather than by sequence length. Tier 0
(fla kernels) tests that directly; the target for our worker is SGLang's 60–77 ms on this GPU,
and the A40 numbers are still owed.

## Milestone 5 — optimization tiers

Each tier goes through the same correctness gate (87 fixtures against the fp32 truth, plus
the order records) before its speed counts. Code for tier 0: `scripts/pod_tier0.sh`,
`scripts/first_requests.py`, `gdn_kernels` in the engine manifest (PR for `m5/tier0-fla`).

### Results: tier 0, fla kernels for Gated DeltaNet, H100, Oct 15 (`m5/tier0-fla-h100-80gb-hbm3-20261015-1835`)

fla-core 0.5.2 (Triton 3.6.0) installed into the worker image's Python, plus gcc and Python
headers for Triton's launcher. transformers 5.10.2 bound fla's `chunk_gated_delta_rule` and
`FusedRMSNormGated`; `causal_conv1d_fn` stayed None (the short convolution still runs as
torch's conv1d). Before anything was measured, fla's kernel ran on the GPU next to
transformers' torch fallback on Qwen3.5-9B's shapes (300 tokens, 32 heads × 128) and agreed
to 4.9e-4 on an output scale of 7.1e-2; `fla.utils.device_platform` was `cuda`. The
pure-torch latency and profile baselines below are milestone 4's H100 runs on the same pod and
image.

**It passes the gate, slightly closer to the truth.** 87 records, worst question 0.0189 (bar
0.025), mean of record means 0.00137 (bar 0.0025), 0 failed, 19 near-ties not judged; order
records worst 0.0101. Against the pure-torch bf16 reference run on an H100 (milestone 1's
bf16 fixture) it differs by up to 0.0229
(`invoice_extract/overdue`, truth margin 0.204), with 3 argmax flips, none clear; fla is
closer to fp32 on 95 questions, pure torch on 72.

**2.5–3.2× lower latency, 2.3–3.0× the throughput** (HTTP, same harness and requests):

| workload | tokens | c=1 p50 torch → fla | SGLang c=1 p50 | c=32 req/s torch → fla | SGLang c=32 req/s |
|---|---|---|---|---|---|
| ticket | 300 | 137.8 → 54.1 ms | 60.0 ms | 6.21 → 14.02 | 95.69 |
| pulse | 843 | 175.8 → 54.2 ms | 62.5 ms | 5.14 → 14.40 | 45.88 |
| article | 2,486 | 283.6 → 102.6 ms | 77.2 ms | 3.10 → 9.25 | 16.38 |

- One request at a time, our worker now beats SGLang on 300- and 843-token inputs (54 vs
  60–62 ms) and loses on 2,486 (103 vs 77 ms).
- Under load SGLang still does 6.8× / 3.2× / 1.8× our throughput, because it batches
  concurrent requests and ours runs one forward at a time. Concurrency now costs us
  throughput: ticket drops from 18.1 req/s at c=1 to 14.2 at c=8 (22%; 9% on the torch
  path). Unverified explanation: the forward is CPU-bound on short inputs (below), so anything
  else holding the GIL (request parsing, encoding of queued requests) slows it.

**Where the time goes now** (in-process, median of 20 passes):

| record | tokens | backbone torch → fla | head | total | head + overhead |
|---|---|---|---|---|---|
| card_trio | 300 | 128.6 → 43.6 ms | 4.5 ms | 50.3 ms | 13.3% |
| pulse-11714037 | 843 | 219.2 → 50.3 ms | 4.1 ms | 57.0 ms | 11.7% |
| long_2k | 2,486 | 353.0 → 90.1 ms | 4.9 ms | 100.8 ms | 10.6% |

- Backbone 2.9× / 4.4× / 3.9× faster. The head is unchanged, so its share rose from 3–5% to
  11–13%: still under the 15% bar for native head work, closer to it.
- Short inputs are now CPU-bound. Under the profiler, card_trio's kernels sum to 16.3 ms inside
  a 43.6 ms backbone (pulse 31.5 in 50.3); fla's `ChunkGatedDeltaRuleFunction` alone takes
  17.1 ms of CPU over its 24 calls (Triton launch overhead, inflated by the profiler). Batching
  amortizes that per-forward overhead across requests; CUDA graphs would remove it, but SGLang
  disables them for Clef.
- long_2k is GPU-bound (87.9 ms of kernels in 90.1 ms). Of those kernels (the categories
  overlap a little): GEMMs 43.6 ms (50%), copies 14.2 ms (16%), the torch short convolution
  9.2 ms (10%), fla's recurrence 5.5 ms (6%).
  The copies and the convolution are the next kernel-level targets (causal-conv1d or fla's
  Triton convolution; fewer layout conversions).

**A new cold-start cost: Triton compiles on first sight of each shape band.** The worker
started with WARMUP=0 and an empty `TRITON_CACHE_DIR`; ready took 12.8 s (weights from local
disk, no compile yet).

| workload | empty cache: first request, first − steady | warm cache: first request, first − steady |
|---|---|---|
| ticket (300 tokens, first shape) | 34,304.5 ms, 34,247.6 ms | 1,171.5 ms, 1,118.8 ms |
| pulse (843) | 208.9 ms, 153.4 ms | 186.9 ms, 133.4 ms |
| article (2,486) | 9,919.6 ms, 9,813.2 ms | 263.0 ms, 160.1 ms |

- Persisting the cache saves 33.1 s on the first request and 9.7 s on the first 2,486-token
  request. The cache was 54 MB after this run. What remains on a warm cache (1.1 s on the first
  request) is first-use setup that Triton's cache does not hold; WARMUP=1, the worker's
  default, pays it before ready.
- The parity run on the same worker paid it again for longer inputs: 5,538 tokens 10.4 s,
  8,641 tokens 10.8 s, 16,384 tokens 12.2 s. Requests in a band already compiled took under
  0.7 s (3,321 tokens: 0.54 s), except the first image request (2.2 s, the vision tower's first
  use).
- Why, from fla 0.5.2's source: the gated RMSNorm and L2-norm kernels autotune with
  `NB = cdiv(rows, 65536)` in the key (9 and up to 5 × len(BT_LIST) configs). With one row per
  token per head (32 heads), NB steps every 2,048 tokens. Every slow and fast request above
  fits: 300/843/1,379 tokens NB 1; 2,486/3,321 NB 2; 5,538 NB 3; 8,641 NB 5; 16,384 NB 8.
- So a fresh worker with an empty cache can pay about 34 s plus about 10 s per new
  2,048-token band, up to Clef's 16,384-token limit. Autotuning runs the kernels, so the cache
  has to be built on the target GPU type, not on a GPU-less CI runner. Options for the
  cold-start work:
  compile all eight bands on a pod per GPU type and persist the cache (network volume, the
  worker-vllm #351 pattern, or an image layer per GPU type), or pin the autotune configs.

Status line `ready_empty_cache` / `ready_warm_cache` printed empty in this run (the script read
the wrong JSON path; fixed in `93a07dc`). Values from the saved info files: 12.836 s and
11.91 s.

Next: tier 1, batching concurrent requests. That is where SGLang's lead is (6.8× at c=32 on
ticket), and it also attacks the CPU-bound short-input path. The A40 numbers are still owed.
