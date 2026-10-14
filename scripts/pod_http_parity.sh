#!/usr/bin/env bash
# Self-driving parity run for the worker image on a Runpod GPU pod (H100 80 GB).
#
# Deploy a pod FROM THE WORKER IMAGE (ghcr.io/ashwinsreedhar28/worker-decide:clef-<sha7>, so
# torch/transformers are the image's), attach the network volume that holds the weights, and
# set the start command to
#
#   bash -c "git clone https://github.com/ashwinsreedhar28/worker-decide /root/worker-decide \
#            && cd /root/worker-decide && bash scripts/pod_http_parity.sh"
#
# What it does, in order:
#   1. starts the worker (TRANSPORT=http) on 127.0.0.1:$PORT and waits for /ping = 200
#   2. scripts/parity_http.py: all 87 fixtures over HTTP, gated against
#      fixtures/clef-flash/bfloat16 (same dtype, bar 0.005), fp32 drift reported
#   3. stops the worker, then re-runs the oracle itself in bf16 (the engine refactor check:
#      must match fixtures/clef-flash/bfloat16) and in fp32 with the math SDPA kernel on
#      every record but trunc_16k (bounds the memory-efficient-kernel caveat from REVIEW.md)
#   4. commits runs/ to parity/http-<rev7>-<yyyymmdd> and pushes
#
# Pod env (template; GH_TOKEN as a Runpod secret, never in the repo):
#   GH_TOKEN      fine-grained PAT, contents:read+write on this repo only   (required to push)
#   GH_REPO       default ashwinsreedhar28/worker-decide
#   HF_HOME       the image sets /runpod-volume/.cache/huggingface; a pod mounts the volume at
#                 /workspace, so this falls back to /workspace/.cache/huggingface (where the
#                 Oct 14 oracle run left the weights)
#   PORT          default 8000
#   STAGE         default all; "http" stops after step 2
#   KEEP_ALIVE    default 0; 1 = sleep forever at the end so the pod can be inspected
set -euo pipefail

cd "$(dirname "$0")/.."
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
PORT="${PORT:-8000}"
STAGE="${STAGE:-all}"
export TRANSPORT=http PORT
if [ ! -d "${HF_HOME:-/nonexistent}" ] && [ -d /workspace ]; then
  export HF_HOME=/workspace/.cache/huggingface
fi
mkdir -p runs
LOG=runs/pod_http_parity.log
exec > >(tee -a "$LOG") 2>&1

echo "== pod_http_parity.sh $(date -u +%FT%TZ) =="
echo "commit $(git rev-parse --short HEAD)  image commit ${WORKER_DECIDE_COMMIT:-?}  HF_HOME $HF_HOME  dtype ${DTYPE:-bfloat16}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

# ---- 1. worker up ---------------------------------------------------------------------------
python3 -m worker > runs/worker.log 2>&1 &
WORKER_PID=$!
trap 'kill $WORKER_PID 2>/dev/null || true' EXIT
echo "worker pid $WORKER_PID on :$PORT"

# ---- 2. parity over HTTP (the gate) ---------------------------------------------------------
REV7=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['revision'][:7])")
python3 scripts/parity_http.py "http://127.0.0.1:$PORT" runs/http-bfloat16 \
  --reference fixtures/clef-flash/bfloat16 --max-dp 0.005 --mean-dp 0.001 \
  --also fixtures/clef-flash/float32 --json runs/parity_http_bfloat16.json \
  > runs/parity_http_bfloat16.md 2> runs/parity_http_bfloat16.err || echo "HTTP PARITY: FAIL (see runs/parity_http_bfloat16.md)"
head -14 runs/parity_http_bfloat16.md
curl -s "http://127.0.0.1:$PORT/" > runs/worker_info.json || true
kill $WORKER_PID 2>/dev/null || true; wait $WORKER_PID 2>/dev/null || true
trap - EXIT
[ "$STAGE" = "http" ] && { echo "STAGE=http, stopping"; exit 0; }

# ---- 3a. engine refactor check: oracle in bf16 must reproduce the bf16 fixture ----------
python3 -m deciders.clef.oracle --requests 'fixtures/requests/*.jsonl' --out runs/bf16-recheck \
  --dtype bfloat16 --revision "$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['revision'])")"
python3 scripts/parity.py fixtures/clef-flash/bfloat16 runs/bf16-recheck --max-dp 0.005 --mean-dp 0.001 \
  --json runs/parity_bf16_recheck.json > runs/parity_bf16_recheck.md || echo "BF16 RECHECK: FAIL"
head -14 runs/parity_bf16_recheck.md

# ---- 3b. fp32 math-kernel bound on everything but trunc_16k ---------------------------------
python3 -m deciders.clef.oracle --requests 'fixtures/requests/*.jsonl' --out runs/fp32-math \
  --dtype float32 --sdp math --exclude trunc_16k \
  --revision "$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/float32/manifest.json'))['revision'])")"
python3 scripts/parity.py fixtures/clef-flash/float32 runs/fp32-math --subset --max-dp 0.001 --mean-dp 0.0001 \
  --json runs/parity_fp32_math.json > runs/parity_fp32_math.md || echo "FP32 MATH BOUND: over 0.001 (see report)"
head -14 runs/parity_fp32_math.md

# ---- 4. push results on a branch -------------------------------------------------------------
BRANCH="parity/http-${REV7}-$(date -u +%Y%m%d)"
git config user.name "worker-decide pod"
git config user.email "pod@worker-decide.invalid"
git checkout -q -b "$BRANCH"
git add -f runs/*.md runs/*.json runs/*.log runs/http-bfloat16 runs/bf16-recheck runs/fp32-math 2>/dev/null || true
git commit -q -m "parity: worker image over HTTP vs bf16 fixture, bf16 recheck, fp32 math bound ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" \
  -m "Produced by scripts/pod_http_parity.sh. Image commit ${WORKER_DECIDE_COMMIT:-unknown}; repo commit $(git rev-parse --short HEAD)."
if [ -n "${GH_TOKEN:-}" ]; then
  # Token in a header via -c, not in the URL (keeps it out of ps and remote config), the way actions/checkout does.
  AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c http.extraheader="AUTHORIZATION: basic ${AUTH}" push -q "https://github.com/${GH_REPO}" "$BRANCH"
  echo "pushed $BRANCH"
else
  echo "GH_TOKEN unset: results committed locally on $BRANCH, not pushed"
fi
echo "== done $(date -u +%FT%TZ) =="
[ "${KEEP_ALIVE:-0}" = "1" ] && sleep infinity
exit 0
