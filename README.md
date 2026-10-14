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
records into `fixtures/clef-flash/float32/`, bf16 pass into `runs/bfloat16/`
with a parity report, self-check, parity against the hosted Workers AI
responses, then commit everything to `oracle/clef-flash-<rev7>-<yyyymmdd>` and
push. Open that branch as a PR; its `runs/parity_*.md` files are the review.

Set `KEEP_ALIVE=1` to keep the pod up afterwards for inspection; otherwise
stop it once the branch appears.

See `CONTRIBUTING.md` for commit rules and `REVIEW.md` for what to audit per
milestone.
