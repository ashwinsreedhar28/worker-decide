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
#   7. on a Runpod pod only: commit runs/ to m5/tier0-fla-<gpu>-<date>-<hhmm> and push
#
# Run from the worker image: POD_SCRIPT=scripts/pod_tier0.sh (and REPO_REF=<branch> until it is
# on main). On a pod that already ran pod_m4.sh (a shallow clone on a local branch, so no
# `git pull`; REF is main once merged, m5/tier0-fla before), detached so it survives the shell:
#   cd /root/worker-decide && git fetch -q --depth 1 origin REF && git stash -q \
#     ; git checkout -q FETCH_HEAD
#   EXIT_WHEN_DONE=1 setsid nohup bash scripts/pod_tier0.sh > /dev/null 2>&1 &
# (the stash: pod_m4.sh's log changes after its commit and blocks the checkout; two lines: a
# trailing & on the first would background the cd too), then follow runs/pod_tier0.log.
# Commits only on a Runpod pod (RUNPOD_POD_ID set, or FORCE_PUSH=1, which also rewrites the
# checkout's git user and branch: never on a dev machine); pushes only if GH_TOKEN is set too.
# It does not stop the pod.
# Env: FLA_SPEC (default fla-core==0.5.2), GH_TOKEN, EXIT_WHEN_DONE, FORCE_PUSH.
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

start_worker() {  # log file, info file; sets WPID. WARMUP=0: no kernel compiles before ready,
                 # so first requests carry the JIT cost. The worker's own timeline (GET /) is
                 # saved: it has process start -> ready to the millisecond, unlike $SECONDS.
  env -u POD_SCRIPT TRANSPORT=http PORT=8000 WARMUP=0 python3 -m worker > "$1" 2>&1 &
  WPID=$!
  if wait_200 http://127.0.0.1:8000/ping 900 $WPID; then
    python3 -c 'import json, sys, urllib.request
print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8000/", timeout=10)), indent=1))' \
      > "$2" 2>/dev/null || true
    return 0
  fi
  return 1
}

ready_s() {  # info file (GET /) -> "<s>s", the worker's ready mark since process start
  python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1]))["engine"]["timeline"]["marks"]["ready"]) + "s")' "$1" 2>/dev/null \
    || echo unknown
}

alive() {  # the worker process still runs (against a dead one, parity_http waits 900 s)
  if kill -0 "$WPID" 2>/dev/null; then return 0; fi
  record worker_died "before $1"; tail -n 40 "$2"; return 1
}

stop_worker() {  # kill, wait up to 30 s, then -9
  kill "$WPID" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
  kill -9 "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; wait_gpu_free || true
}

# ---- 1. install ------------------------------------------------------------------------------
step "1. install $FLA_SPEC"
# Triton builds a C launcher that includes Python.h with gcc on first use. The worker image is
# CUDA "base" with python3-pip installed --no-install-recommends: no gcc and no Python.h
# (Oct 15 audit; SGLang's Triton worked only because its uv Python ships headers). Decide by
# the header, not by gcc: a pod that ran pod_m4.sh has gcc but still no headers.
PYINC=$(python3 -c 'import sysconfig; s = sysconfig.get_default_scheme(); s = "posix_prefix" if s == "posix_local" else s; print(sysconfig.get_paths(scheme=s)["include"])')
if ! command -v gcc >/dev/null || [ ! -f "$PYINC/Python.h" ]; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev python3-dev \
    >/dev/null && record compilers_headers ok || record compilers_headers FAIL
fi
python3 -m pip install -q "$FLA_SPEC" einops && record fla_install ok || record fla_install FAIL
# Not just "is it bound": fla catches a failed Triton probe at import and falls back to CPU, and
# transformers binds its functions anyway. So run the kernel on the GPU next to transformers'
# own torch fallback, on Qwen3.5-9B's Gated DeltaNet shapes (32 heads x 128 after the q/k
# repeat, bf16 inputs, float32 gate), and require it to run on CUDA and agree.
python3 - <<'PY' > runs/tier0-kernels.txt 2>&1
import importlib.metadata as md
import torch
import fla.utils
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("fla-core", md.version("fla-core"), "| triton", md.version("triton"))
print("chunk_gated_delta_rule:", getattr(m.chunk_gated_delta_rule, "__module__", None))
print("FusedRMSNormGated:", m.FusedRMSNormGated)
print("causal_conv1d_fn:", m.causal_conv1d_fn, "| is_fast_path_available:", m.is_fast_path_available)
print("fla device_platform:", fla.utils.device_platform)
torch.manual_seed(0)
B, T, H, D = 1, 300, 32, 128
dev = "cuda"
q, k, v = (torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
g = -torch.rand(B, T, H, device=dev, dtype=torch.float32)
beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
kw = dict(g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
o_fla, _ = m.chunk_gated_delta_rule(q, k, v, **kw)
o_ref, _ = m.torch_chunk_gated_delta_rule(q, k, v, **kw)
torch.cuda.synchronize()
diff = (o_fla.float() - o_ref.float()).abs().max().item()
scale = o_ref.float().abs().max().item()
print(f"kernel on {o_fla.device}: max |fla - torch| = {diff:.3e} (output scale {scale:.3e})")
print("KERNEL_OK" if o_fla.is_cuda and diff <= 0.05 * max(scale, 1e-6) else "KERNEL_MISMATCH")
PY
cat runs/tier0-kernels.txt
if grep -q "chunk_gated_delta_rule: fla" runs/tier0-kernels.txt \
   && grep -q "fla device_platform: cuda" runs/tier0-kernels.txt \
   && grep -q "KERNEL_OK" runs/tier0-kernels.txt; then
  record fla_active ok
  STEPS_DONE=1
else
  record fla_active FAIL
  echo "fla is not running on the GPU (see runs/tier0-kernels.txt); nothing to measure"
  STEPS_DONE=0
fi

if [ "$STEPS_DONE" = 1 ]; then
  # ---- 2. empty cache: ready time, first requests ----------------------------------------------
  # Weights first, so neither start pays a download (a fresh pod has an empty HF_HOME).
  python3 -c "from huggingface_hub import snapshot_download as s; s('Cloudflare/clef-flash', revision='$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/float32/manifest.json'))['revision'])")')" \
    > /dev/null && record weights_prefetched ok || record weights_prefetched FAIL
  step "2. our worker, empty Triton cache"
  rm -rf "$TRITON_CACHE_DIR"
  EMPTY_OK=0
  if start_worker runs/worker-tier0.log runs/worker-tier0-info.json; then
    EMPTY_OK=1
    record ready_empty_cache "$(ready_s runs/worker-tier0-info.json)"
    python3 scripts/first_requests.py http://127.0.0.1:8000 "runs/first-fla-$SLUG" \
      --label "fla, empty cache, $SLUG" && record first_empty ok || record first_empty FAIL

    # ---- 3. correctness ----------------------------------------------------------------------
    step "3. correctness gate"
    alive "step 3" runs/worker-tier0.log && {
    # Pure-torch baseline on the same GPU if we have one: a worker run for this GPU, else the
    # bf16 oracle fixture if it was made here (H100), else the A40 run (cross-GPU, info only).
    OURS_TORCH=runs/http-a40
    BF16_GPU=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['gpu'])")
    [ "$GPU" = "$BF16_GPU" ] && OURS_TORCH=fixtures/clef-flash/bfloat16
    for d in "runs/http-$SLUG" "runs/http-${SLUG%%-*}"; do [ -d "$d" ] && OURS_TORCH="$d" && break; done
    record torch_baseline "$OURS_TORCH"
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
    }

    # ---- 4. load ------------------------------------------------------------------------------
    step "4. load test"
    if alive "step 4" runs/worker-tier0.log; then
      python3 scripts/loadtest.py http://127.0.0.1:8000 "runs/load-ours-fla-$SLUG" \
        --label "ours-fla-$SLUG" && record load_fla ok || record load_fla FAIL
    fi
  else
    record worker_start FAIL; tail -n 40 runs/worker-tier0.log
  fi
  stop_worker

  # ---- 5. warm cache ---------------------------------------------------------------------------
  step "5. restart with the cache warm"
  if [ "$EMPTY_OK" != 1 ]; then
    record warm_cache "skipped (the empty-cache run did not start)"
  elif start_worker runs/worker-tier0-warm.log runs/worker-tier0-warm-info.json; then
    record ready_warm_cache "$(ready_s runs/worker-tier0-warm-info.json)"
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
BRANCH="m5/tier0-fla-$SLUG-$(date -u +%Y%m%d-%H%M)"
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
