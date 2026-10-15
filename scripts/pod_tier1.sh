#!/usr/bin/env bash
# Optimization tier 1 on a Runpod pod: dynamic batching (BATCH_MAX) on top of tier 0's fla.
#
# Tier 0 left our worker at ~14 req/s at 32 concurrent requests on an H100, SGLang at 96: it
# batches concurrent requests and we ran one forward at a time. worker/batcher.py now lets
# requests that arrive during a forward share the next one. Whether the reference collate and
# model return batched outputs in record order is checked here by the gate itself, run with
# requests in flight so the answers it checks come out of real batches.
#
#   1. fla-core installed and on the GPU (as pod_shapes.sh step 1)
#   2. worker, BATCH_MAX=$BATCH (default 8): correctness gate with --concurrency $BATCH (fp32
#      truth, 87 records), order records, then the load test; batcher stats from GET /
#   3. the same worker with BATCH_MAX=1 (tier 0) on this pod: load test, for a same-GPU baseline
#   4. on a Runpod pod only: commit runs/ to m5/tier1-<gpu>-<date>-<hhmm> and push
#
# Run from the worker image: POD_SCRIPT=scripts/pod_tier1.sh (REPO_REF=<branch> until merged),
# or on a pod that has the repo:
#   cd /root/worker-decide && git fetch -q --depth 1 origin REF && git stash -q \
#     ; git checkout -q FETCH_HEAD
#   EXIT_WHEN_DONE=1 setsid nohup bash scripts/pod_tier1.sh > /dev/null 2>&1 &
# After pod_shapes.sh in the same session: bash -c 'bash scripts/pod_shapes.sh; bash
# scripts/pod_tier1.sh' under the same setsid nohup line. Commits only on a Runpod pod
# (RUNPOD_POD_ID set, or FORCE_PUSH=1: never on a dev machine); pushes only if GH_TOKEN is set
# too. It does not stop the pod. Env: BATCH (default 8), FLA_SPEC, GH_TOKEN, EXIT_WHEN_DONE,
# FORCE_PUSH.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
FLA_SPEC="${FLA_SPEC:-fla-core==0.5.2}"
BATCH="${BATCH:-8}"
export HF_HOME="${HF_HOME_POD:-/root/hf}"
export TRITON_CACHE_DIR=/root/triton-cache-tier1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # the image's
mkdir -p runs "$HF_HOME"
LOG=runs/pod_tier1.log
exec > >(tee -a "$LOG") 2>&1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU" | sed -E 's/NVIDIA //; s/[^A-Za-z0-9]+/-/g' | tr 'A-Z' 'a-z' | sed 's/-$//')
echo "== pod_tier1.sh $(date -u +%FT%TZ) == commit $(git rev-parse --short HEAD)  gpu $GPU ($SLUG)  batch $BATCH"
STATUS=()
step() { echo; echo "==== $* ($(date -u +%T)) ===="; }
record() { STATUS+=("$1: $2"); echo "[status] $1: $2"; }

http_code() {  # url -> HTTP status, 0 if unreachable (the image has no curl)
  python3 -c 'import sys, urllib.request, urllib.error
try: print(urllib.request.urlopen(sys.argv[1], timeout=5).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception: print(0)' "$1"
}

wait_200() {  # url, timeout_s, pid
  local t0=$SECONDS code
  while true; do
    code=$(http_code "$1")
    [ "$code" = 200 ] && return 0
    [ "$code" = 500 ] && return 1
    ! kill -0 "$3" 2>/dev/null && return 1
    (( SECONDS - t0 > $2 )) && return 1
    sleep 1
  done
}

wait_gpu_free() {
  local t0=$SECONDS used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 1024 ] && return 0
    (( SECONDS - t0 > 120 )) && return 1
    sleep 2
  done
}

start_worker() {  # batch_max, log file; sets WPID
  env -u POD_SCRIPT TRANSPORT=http PORT=8000 WARMUP=0 BATCH_MAX="$1" python3 -m worker > "$2" 2>&1 &
  WPID=$!
  wait_200 http://127.0.0.1:8000/ping 900 $WPID
}

alive() {  # name of the next step, log file
  if kill -0 "$WPID" 2>/dev/null; then return 0; fi
  record worker_died "before $1"; tail -n 40 "$2"; return 1
}

stop_worker() {
  kill "$WPID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
  kill -9 "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; wait_gpu_free || true
}

save_root() {  # file: GET / (engine info and batcher stats)
  python3 -c 'import json, urllib.request
print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8000/", timeout=10)), indent=1))' \
    > "$1" 2>/dev/null || true
}

# ---- 1. fla ----------------------------------------------------------------------------------
step "1. fla-core"
PYINC=$(python3 -c 'import sysconfig; s = sysconfig.get_default_scheme(); s = "posix_prefix" if s == "posix_local" else s; print(sysconfig.get_paths(scheme=s)["include"])')
if ! command -v gcc >/dev/null || [ ! -f "$PYINC/Python.h" ]; then  # Triton's launcher build
  apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev python3-dev \
    >/dev/null && record compilers_headers ok || record compilers_headers FAIL
fi
python3 -c 'import fla' 2>/dev/null || python3 -m pip install -q "$FLA_SPEC" einops
python3 - <<'PY' > runs/tier1-kernels.txt 2>&1
import torch
import fla.utils
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("chunk_gated_delta_rule:", getattr(m.chunk_gated_delta_rule, "__module__", None))
print("fla device_platform:", fla.utils.device_platform)
q = torch.randn(1, 64, 32, 128, device="cuda", dtype=torch.bfloat16)
g = -torch.rand(1, 64, 32, device="cuda", dtype=torch.float32)
b = torch.rand(1, 64, 32, device="cuda", dtype=torch.bfloat16).sigmoid()
o, _ = m.chunk_gated_delta_rule(q, q, q, g=g, beta=b, use_qk_l2norm_in_kernel=True)
print("KERNEL_RAN" if o.is_cuda and torch.isfinite(o).all().item() else "KERNEL_BAD")
PY
cat runs/tier1-kernels.txt
if grep -q "chunk_gated_delta_rule: fla" runs/tier1-kernels.txt \
   && grep -q "fla device_platform: cuda" runs/tier1-kernels.txt \
   && grep -q "KERNEL_RAN" runs/tier1-kernels.txt; then
  record fla_active ok
else
  record fla_active "FAIL (tier 1 runs on whatever path transformers bound)"
fi
python3 -c "from huggingface_hub import snapshot_download as s; s('Cloudflare/clef-flash', revision='$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/float32/manifest.json'))['revision'])")')" \
  > /dev/null && record weights_prefetched ok || record weights_prefetched FAIL

# ---- 2. batching worker ----------------------------------------------------------------------
step "2. worker with BATCH_MAX=$BATCH"
if start_worker "$BATCH" runs/worker-tier1.log; then
  record worker_batched ok
  # Same-GPU tier 0 answers if this pod ran tier 0 (informational): runs/http-fla-<slug>.
  ALSO=()
  [ -d "runs/http-fla-$SLUG" ] && ALSO=(--also "runs/http-fla-$SLUG")
  python3 scripts/parity_http.py http://127.0.0.1:8000 "runs/http-batched-$SLUG" \
    --concurrency "$BATCH" "${ALSO[@]}" --json "runs/parity_batched_$SLUG.json" \
    > "runs/parity_batched_$SLUG.md" && record parity_batched PASS || record parity_batched FAIL
  head -16 "runs/parity_batched_$SLUG.md"
  if alive "order records" runs/worker-tier1.log && [ -f fixtures/clef-flash/float32-order/manifest.json ]; then
    python3 scripts/parity_http.py http://127.0.0.1:8000 "runs/http-batched-order-$SLUG" \
      --concurrency "$BATCH" --requests fixtures/requests-order/order.jsonl \
      --reference fixtures/clef-flash/float32-order \
      > "runs/parity_batched_order_$SLUG.md" && record parity_batched_order PASS \
      || record parity_batched_order FAIL
  fi
  save_root "runs/worker-tier1-after-parity.json"
  if alive "load test" runs/worker-tier1.log; then
    python3 scripts/loadtest.py http://127.0.0.1:8000 "runs/load-ours-batched-$SLUG" \
      --label "ours-fla-batch$BATCH-$SLUG" && record load_batched ok || record load_batched FAIL
  fi
  save_root "runs/worker-tier1-after-load.json"
else
  record worker_batched FAIL; tail -n 40 runs/worker-tier1.log
fi
stop_worker

# ---- 3. same pod, BATCH_MAX=1 ----------------------------------------------------------------
step "3. worker with BATCH_MAX=1 (tier 0 baseline on this GPU)"
if start_worker 1 runs/worker-tier1-single.log; then
  python3 scripts/loadtest.py http://127.0.0.1:8000 "runs/load-ours-single-$SLUG" \
    --label "ours-fla-single-$SLUG" && record load_single ok || record load_single FAIL
else
  record worker_single FAIL; tail -n 40 runs/worker-tier1-single.log
fi
stop_worker

# ---- 4. push ---------------------------------------------------------------------------------
step "4. push"
[ ${#STATUS[@]} -gt 0 ] && printf '%s\n' "${STATUS[@]}" | tee runs/pod_tier1_status.txt
BRANCH="m5/tier1-$SLUG-$(date -u +%Y%m%d-%H%M)"
if [ -z "${RUNPOD_POD_ID:-}" ] && [ "${FORCE_PUSH:-0}" != "1" ]; then
  echo "not on a Runpod pod: nothing committed or pushed (results are under runs/)"
else
  git config user.name "worker-decide pod"
  git config user.email "pod@worker-decide.invalid"
  git checkout -q -B "$BRANCH"
  for p in runs/pod_tier1* runs/tier1-* runs/worker-tier1* runs/http-batched-* runs/parity_batched* \
           runs/load-ours-batched-* runs/load-ours-single-*; do
    [ -e "$p" ] && git add -f "$p"
  done
  git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    if [ "$(stat -c %s "$f")" -gt 52428800 ]; then
      echo "not committing $f ($(du -h "$f" | cut -f1)), too large"; git reset -q -- "$f"
    fi
  done
  git commit -q --allow-empty -m "tier 1 (batching, BATCH_MAX=$BATCH) on $GPU" \
    -m "Produced by scripts/pod_tier1.sh at $(git rev-parse --short HEAD). $(printf '%s; ' "${STATUS[@]}")"
  if [ -n "${GH_TOKEN:-}" ]; then
    AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
    git -c http.extraheader="AUTHORIZATION: basic ${AUTH}" push -q "https://github.com/${GH_REPO}" "$BRANCH" \
      && echo "pushed $BRANCH" || echo "PUSH FAILED"
  else
    echo "GH_TOKEN unset: committed locally on $BRANCH, not pushed"
  fi
fi
echo "== done $(date -u +%FT%TZ) =="
if [ "${EXIT_WHEN_DONE:-0}" = "1" ]; then exit 0; fi
echo "holding the pod open; stop it from the Runpod console"
exec sleep infinity
