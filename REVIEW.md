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
python scripts/parity.py fixtures/clef-flash/fp32 fixtures/clef-flash/fp32   # self-check, must pass trivially
```

Expect:

- `make check` green.
- Parity self-check: all records PASS, 0 span mismatches, max |Δp| = 0.

Audit points:

- `fixtures/requests/*.jsonl` bodies are byte-identical to what Brian's table
  sent (compare against `bench/quality.py:systemone_body` in Pulse).
- `fixtures/clef-flash/manifest.json` pins the HF revision and file shas, and
  the torch/transformers versions match the model card's tested pair.
- `scripts/parity.py` reports the span gate before the numeric gate, and
  near-ties (oracle top-2 margin ≤ 0.02) are listed, not counted as failures.

Out of scope: the HTTP worker, Runpod endpoint template, SGLang, caching arms,
Typical backend, any change to the head.

(Filled in as the milestone lands: bf16 gap, hosted-drift numbers.)
