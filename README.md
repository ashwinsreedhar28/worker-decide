# worker-decide

A Runpod Serverless worker for SystemOne-style decision models, with pluggable
deciders. Target models: Cloudflare Clef-flash (Apache-2.0) first, Oz Labs
Typical second. The wire format is `POST /v1/systemone` (`{model, state,
questions}` in, `{model, answers, usage}` out), so clients written for hosted
Jev or the Runpod OpenJev tutorial work by changing the base URL.

Status: milestone 1 (Clef-flash fp32 oracle + fixtures + parity gate). No
worker yet.

## Layout

```
deciders/clef/     encode + oracle for Cloudflare/clef-flash (uses the HF snapshot's own joint_schema_model.py)
fixtures/requests/ SystemOne request bodies (Pulse 50, edge cases, images)
fixtures/clef-flash/ fp32 reference outputs + manifest.json (produced by a pod, lands via PR)
scripts/           make_fixtures.py, parity.py, pod_oracle.sh
tests/             CPU-only tests against a synthetic fixture
```

## Checks

```
make check     # ruff + pytest, no GPU needed
```

See `CONTRIBUTING.md` for commit rules and `REVIEW.md` for what to audit per
milestone.
