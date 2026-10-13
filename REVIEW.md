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
python scripts/parity.py fixtures/clef-flash/float32 runs/bfloat16                 # on the oracle branch
```

Expect:

- `make check` green (25 tests).
- Parity self-check: all 87 records PASS, 0 span mismatches, max |Δp| = 0.
- bf16 vs fp32: span gate ok on every record (same encoder, same ids), numeric
  gate within the bar; the Core AI port saw max |Δp| 0.011 for bf16.
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
- `deciders/clef/oracle.py` calls the snapshot's own `joint_schema_model`
  functions and nothing else touches logits; batch size is 1.
- The oracle branch commit was made by the pod (`worker-decide pod`), and the
  manifest's `worker_decide_commit` matches a commit on `main`.

Out of scope: the HTTP worker, Runpod endpoint template, SGLang, caching arms,
Typical backend, any change to the head.

(Filled in as the milestone lands: bf16 gap, hosted-drift numbers.)
