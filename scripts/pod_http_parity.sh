#!/usr/bin/env bash
# Self-driving parity run for the worker image on a Runpod GPU pod (H100 80 GB).
#
# Deploy a pod FROM THE WORKER IMAGE (ghcr.io/ashwinsreedhar28/worker-decide:clef-<sha7>, so
# torch/transformers are the image's), attach the network volume that holds the weights, and
# set the environment variable
#
#   POD_SCRIPT=scripts/pod_http_parity.sh        (optional REPO_REF=<branch>, default main)
#
# The image entry point (python -m worker) then clones the repo to /root/worker-decide and runs
# this script. (A template start command would do the same, but that field has twice not
# reached the container on Runpod; env vars always do.)
#
# What it does, in order:
#   1. starts the worker (TRANSPORT=http) on 127.0.0.1:$PORT and waits for /ping = 200
#   2. scripts/parity_http.py: all 87 fixtures over HTTP, one pass, two gates —
#      correctness against the fp32 truth (any GPU, 0.025 / 0.0025) and, only when this pod's
#      GPU is the one fixtures/clef-flash/bfloat16 was made on, regression against that bf16
#      fixture (0.005 / 0.001). Two bf16 runs on different GPUs disagree by up to ~0.03 while
#      each is within the correctness bar (Oct 14, A40 vs H100), so regression is same-GPU only.
#   3. stops the worker, then re-runs the oracle itself in bf16 (the engine refactor check,
#      against the bf16 fixture on the same GPU, otherwise against fp32 at the correctness bar)
#      and in fp32 with the math SDPA kernel on every record but trunc_16k (bounds the
#      memory-efficient-kernel caveat from REVIEW.md)
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
#   EXIT_WHEN_DONE default 0; 1 = exit after pushing instead of holding the pod open
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

# ---- 1. worker up (its output is tailed to the console so a failed start is visible) ---------
mkdir -p runs/http-bfloat16
env -u POD_SCRIPT TRANSPORT=http python3 -m worker > runs/worker.log 2>&1 &   # -u: or it re-runs this script
WORKER_PID=$!
tail -n +1 -F runs/worker.log | sed -u 's/^/[worker] /' &
TAIL_PID=$!
trap 'kill $WORKER_PID $TAIL_PID 2>/dev/null || true' EXIT
echo "worker pid $WORKER_PID on :$PORT"

# ---- 2. parity over HTTP (correctness always; regression only on the reference's GPU) --------
REV7=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['revision'][:7])")
REF_GPU=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['gpu'])")
POD_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
REGRESSION_ARGS=()
if [ "$POD_GPU" = "$REF_GPU" ]; then
  REGRESSION_ARGS=(--regression fixtures/clef-flash/bfloat16)
  echo "regression gate ON: pod GPU matches the bf16 fixture's ($REF_GPU)"
else
  echo "regression gate SKIPPED: pod GPU '$POD_GPU' is not the bf16 fixture's '$REF_GPU'"
fi
HTTP_OK=1
python3 scripts/parity_http.py "http://127.0.0.1:$PORT" runs/http-bfloat16 \
  ${REGRESSION_ARGS[@]+"${REGRESSION_ARGS[@]}"} --json runs/parity_http_correctness.json --ready-timeout "${READY_TIMEOUT:-600}" \
  > runs/parity_http.md 2> >(tee runs/parity_http.err | sed -u 's/^/[http] /' >&2) || HTTP_OK=0
sleep 1
grep -E '^## |^\| (records|failed|max|mean of record|result)' runs/parity_http.md || true
if [ "$HTTP_OK" = 0 ]; then
  echo "HTTP PARITY: FAIL"; echo "--- worker.log tail:"; tail -n 40 runs/worker.log
fi
python3 - "$PORT" > runs/worker_info.json <<'PY' || true
import json, sys, urllib.request
print(json.dumps(json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/", timeout=10)), indent=1))
PY
kill $WORKER_PID 2>/dev/null || true; wait $WORKER_PID 2>/dev/null || true
kill $TAIL_PID 2>/dev/null || true
trap - EXIT
[ "$STAGE" = "http" ] && { echo "STAGE=http, stopping"; exit 0; }

# ---- 3a. engine refactor check: the oracle path in bf16 ----------------------------------
python3 -m deciders.clef.oracle --requests 'fixtures/requests/*.jsonl' --out runs/bf16-recheck \
  --dtype bfloat16 --revision "$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['revision'])")"
if [ ${#REGRESSION_ARGS[@]} -gt 0 ]; then   # same GPU: must reproduce the bf16 fixture
  python3 scripts/parity.py fixtures/clef-flash/bfloat16 runs/bf16-recheck --max-dp 0.005 --mean-dp 0.001 \
    --json runs/parity_bf16_recheck.json > runs/parity_bf16_recheck.md || echo "BF16 RECHECK: FAIL"
else                                        # other GPU: within bf16 tolerance of the truth
  python3 scripts/parity.py fixtures/clef-flash/float32 runs/bf16-recheck --max-dp 0.025 --mean-dp 0.0025 \
    --json runs/parity_bf16_recheck.json > runs/parity_bf16_recheck.md || echo "BF16 RECHECK: FAIL"
fi
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
git checkout -q -B "$BRANCH"
for p in runs/*.md runs/*.json runs/*.log runs/*.err runs/http-bfloat16 runs/bf16-recheck runs/fp32-math; do
  [ -e "$p" ] && git add -f "$p"   # add what exists; a missing dir must not empty the commit
done
git commit -q --allow-empty -m "parity: worker image over HTTP vs bf16 fixture, bf16 recheck, fp32 math bound ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" \
  -m "Produced by scripts/pod_http_parity.sh. Image commit ${WORKER_DECIDE_COMMIT:-unknown}; repo commit $(git rev-parse --short HEAD). HTTP gates (correctness$([ ${#REGRESSION_ARGS[@]} -gt 0 ] && echo " + regression" || echo ", regression skipped: $POD_GPU")): $([ "${HTTP_OK:-0}" = 1 ] && echo PASS || echo FAIL)."
if [ -n "${GH_TOKEN:-}" ]; then
  # Token in a header via -c, not in the URL (keeps it out of ps and remote config), the way actions/checkout does.
  AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
  git -c http.extraheader="AUTHORIZATION: basic ${AUTH}" push -q "https://github.com/${GH_REPO}" "$BRANCH"
  echo "pushed $BRANCH"
else
  echo "GH_TOKEN unset: results committed locally on $BRANCH, not pushed"
fi
echo "== done $(date -u +%FT%TZ) =="
# Hold the container open: Runpod restarts the command when it exits, and a restart re-runs
# the whole thing against the existing checkout (Oct 14: a passing run looked like a crash
# loop because `[ x = 1 ] && sleep infinity` returns 1 under set -e). Stop the pod from the
# console when the branch has landed; EXIT_WHEN_DONE=1 to exit instead.
if [ "${EXIT_WHEN_DONE:-0}" = "1" ]; then
  exit 0
fi
echo "holding the pod open; stop it from the Runpod console"
exec sleep infinity
