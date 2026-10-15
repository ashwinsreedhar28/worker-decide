#!/usr/bin/env bash
# Optimization tier 0 on a Runpod pod: flash-linear-attention kernels for Gated DeltaNet.
#
# The Oct 15 H100 profile: backbone 95-97% of request time; inside it the GPU idles ~44% across
# thousands of small kernels, consistent with transformers' pure-torch Gated DeltaNet fallback.
# transformers 5.10.2 falls back per piece: fla supplies chunk_gated_delta_rule and
# FusedRMSNormGated (the recurrence, the expensive part of a prefill); causal-conv1d only the
# short convolution, which stays on torch's conv1d here. So this installs fla-core alone (pure
# Triton, no CUDA build) into the worker image's environment and re-measures everything:
#
#   1. install fla-core (+ gcc for Triton's launcher); confirm transformers picked it up
#   2. our worker with an EMPTY Triton cache: time to ready, first vs steady request per shape
#   3. correctness gate on the 87 fixtures (vs fp32) and the order records; informational
#      comparison with the pure-torch bf16 run on the same GPU when there is one
#   4. load test (same workloads as the baseline)
#   5. restart with the cache now WARM: first requests again (what persisting the cache buys)
#   6. in-process profile with traces
#   7. on a Runpod pod only: commit runs/ to m5/tier0-fla-<gpu>-<date> and push
#
# Run from the worker image (POD_SCRIPT=scripts/pod_tier0.sh) or, on a pod that already ran
# pod_m4.sh, from its checkout:  cd /root/worker-decide && git pull && bash scripts/pod_tier0.sh
# Env: FLA_SPEC (default fla-core==0.5.2), GH_TOKEN, EXIT_WHEN_DONE.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
FLA_SPEC="${FLA_SPEC:-fla-core==0.5.2}"
export HF_HOME="${HF_HOME_POD:-/root/hf}"
export TRITON_CACHE_DIR=/root/triton-cache-tier0
mkdir -p runs "$HF_HOME"
LOG=runs/pod_tier0.log
exec > >(tee -a "$LOG") 2>&1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU" | sed -E 's/NVIDIA //; s/[^A-Za-z0-9]+/-/g' | tr 'A-Z' 'a-z' | sed 's/-$//')
echo "== pod_tier0.sh $(date -u +%FT%TZ) == commit $(git rev-parse --short HEAD)  gpu $GPU ($SLUG)"
STATUS=()
step() { echo; echo "==== $* ($(date -u +%T)) ===="; }
record() { STATUS+=("$1: $2"); echo "[status] $1: $2"; }

http_code() {  # url -> HTTP status, 0 if unreachable (the image has no curl)
  python3 -c 'import sys, urllib.request, urllib.error
try: print(urllib.request.urlopen(sys.argv[1], timeout=5).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception: print(0)' "$1"
}

wait_200() {  # url, timeout_s [, pid]
  local t0=$SECONDS code
  while true; do
    code=$(http_code "$1")
    [ "$code" = 200 ] && return 0
    [ "$code" = 500 ] && return 1
    [ -n "${3:-}" ] && ! kill -0 "$3" 2>/dev/null && return 1
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

start_worker() {  # log file; sets WPID and READY_S
  local t0=$SECONDS
  env -u POD_SCRIPT TRANSPORT=http PORT=8000 python3 -m worker > "$1" 2>&1 &
  WPID=$!
  if wait_200 http://127.0.0.1:8000/ping 900 $WPID; then READY_S=$(( SECONDS - t0 )); return 0; fi
  READY_S=""; return 1
}

stop_worker() { kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; wait_gpu_free || true; }

# ---- 1. install ------------------------------------------------------------------------------
step "1. install $FLA_SPEC"
if ! command -v gcc >/dev/null; then  # Triton builds a small C launcher on first use
  apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null \
    && record gcc ok || record gcc FAIL
fi
python3 -m pip install -q "$FLA_SPEC" einops && record fla_install ok || record fla_install FAIL
python3 - <<'PY' > runs/tier0-kernels.txt 2>&1
import importlib.metadata as md
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("fla-core", md.version("fla-core"), "| triton", md.version("triton"))
print("chunk_gated_delta_rule:", getattr(m.chunk_gated_delta_rule, "__module__", None))
print("FusedRMSNormGated:", m.FusedRMSNormGated)
print("causal_conv1d_fn:", m.causal_conv1d_fn, "| is_fast_path_available:", m.is_fast_path_available)
PY
cat runs/tier0-kernels.txt
if ! grep -q "chunk_gated_delta_rule: fla" runs/tier0-kernels.txt; then
  record fla_active FAIL
  echo "transformers did not pick up fla; nothing to measure"
  STEPS_DONE=0
else
  record fla_active ok
  STEPS_DONE=1
fi

if [ "$STEPS_DONE" = 1 ]; then
  # ---- 2. empty cache: ready time, first requests ----------------------------------------------
  step "2. our worker, empty Triton cache"
  rm -rf "$TRITON_CACHE_DIR"
  if start_worker runs/worker-tier0.log; then
    record ready_empty_cache "${READY_S}s"
    python3 scripts/first_requests.py http://127.0.0.1:8000 "runs/first-fla-$SLUG" \
      --label "fla, empty cache, $SLUG" && record first_empty ok || record first_empty FAIL

    # ---- 3. correctness ----------------------------------------------------------------------
    step "3. correctness gate"
    OURS_TORCH=runs/http-a40
    BF16_GPU=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['gpu'])")
    [ "$GPU" = "$BF16_GPU" ] && OURS_TORCH=fixtures/clef-flash/bfloat16
    python3 scripts/parity_http.py http://127.0.0.1:8000 "runs/http-fla-$SLUG" \
      --also "$OURS_TORCH" --json "runs/parity_fla_$SLUG.json" \
      > "runs/parity_fla_$SLUG.md" && record parity_fla PASS || record parity_fla FAIL
    head -16 "runs/parity_fla_$SLUG.md"
    if [ -f fixtures/clef-flash/float32-order/manifest.json ]; then
      python3 scripts/parity_http.py http://127.0.0.1:8000 "runs/http-fla-order-$SLUG" \
        --requests fixtures/requests-order/order.jsonl --reference fixtures/clef-flash/float32-order \
        > "runs/parity_fla_order_$SLUG.md" && record parity_fla_order PASS || record parity_fla_order FAIL
    fi
    python3 scripts/compare_runs.py fixtures/clef-flash/float32 "$OURS_TORCH" "runs/http-fla-$SLUG" \
      --labels "torch-gdn,fla-gdn" > "runs/compare_torch_vs_fla_$SLUG.md" || true

    # ---- 4. load ------------------------------------------------------------------------------
    step "4. load test"
    python3 scripts/loadtest.py http://127.0.0.1:8000 "runs/load-ours-fla-$SLUG" \
      --label "ours-fla-$SLUG" && record load_fla ok || record load_fla FAIL
  else
    record worker_start FAIL; tail -n 40 runs/worker-tier0.log
  fi
  stop_worker

  # ---- 5. warm cache ---------------------------------------------------------------------------
  step "5. restart with the cache warm"
  if start_worker runs/worker-tier0-warm.log; then
    record ready_warm_cache "${READY_S}s"
    python3 scripts/first_requests.py http://127.0.0.1:8000 "runs/first-fla-warmcache-$SLUG" \
      --label "fla, warm cache, $SLUG" && record first_warm ok || record first_warm FAIL
  else
    record worker_start_warm FAIL
  fi
  stop_worker
  du -sh "$TRITON_CACHE_DIR" | tee runs/tier0-triton-cache-size.txt

  # ---- 6. profile ------------------------------------------------------------------------------
  step "6. profile"
  python3 scripts/profile_engine.py "runs/profile-ours-fla-$SLUG" --reps 20 --warmup 3 --trace \
    && record profile_fla ok || record profile_fla FAIL
fi

# ---- 7. push ---------------------------------------------------------------------------------
step "7. push"
[ ${#STATUS[@]} -gt 0 ] && printf '%s\n' "${STATUS[@]}" | tee runs/pod_tier0_status.txt
BRANCH="m5/tier0-fla-$SLUG-$(date -u +%Y%m%d)"
if [ -z "${RUNPOD_POD_ID:-}" ] && [ "${FORCE_PUSH:-0}" != "1" ]; then
  echo "not on a Runpod pod: nothing committed or pushed (results are under runs/)"
else
  git config user.name "worker-decide pod"
  git config user.email "pod@worker-decide.invalid"
  git checkout -q -B "$BRANCH"
  for p in runs/pod_tier0* runs/tier0-* runs/first-fla-* runs/http-fla-* runs/parity_fla* \
           runs/compare_torch_vs_fla_* runs/load-ours-fla-* runs/profile-ours-fla-* runs/worker-tier0*; do
    [ -e "$p" ] && git add -f "$p"
  done
  git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    if [ "$(stat -c %s "$f")" -gt 52428800 ]; then
      echo "not committing $f ($(du -h "$f" | cut -f1)), too large"; git reset -q -- "$f"
    fi
  done
  git commit -q --allow-empty -m "tier 0 (fla-core GDN kernels) on $GPU" \
    -m "Produced by scripts/pod_tier0.sh at $(git rev-parse --short HEAD). $(printf '%s; ' "${STATUS[@]}")"
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
