#!/usr/bin/env bash
# Per-shape cost on a Runpod pod: is a request whose exact token count is new slower, and why?
#
# The tier 0 H100 run's parity log: text requests under 2,048 tokens took a median 166 ms when
# their exact token count was new and 54 ms when it had been seen; the pure-torch run of Oct 14
# shows the same split. scripts/shape_probe.py measures it directly and names the ops that grow.
# Each environment variant runs in its own process (these settings are fixed at CUDA init):
#
#   1. fla-core installed and bound on the GPU (as pod_tier0.sh step 1, check only)
#   2. probes, pulse record (843 tokens), 8 variants each:
#        default  (with a profiler trace of one new/repeat pair)
#        alloc    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (allocator growth?)
#        eager    CUDA_MODULE_LOADING=EAGER                          (lazy kernel loading?)
#        lt       TORCH_BLAS_PREFER_CUBLASLT=1                       (GEMM path / heuristics?)
#      and default on long_2k (2,486 tokens), to see how the cost scales with length
#   3. on a Runpod pod only: commit runs/ to m5/shapes-<gpu>-<date>-<hhmm> and push
#
# Run from the worker image: POD_SCRIPT=scripts/pod_shapes.sh (REPO_REF=<branch> until it is on
# main). On a pod that already has the repo (a shallow clone on a local branch):
#   cd /root/worker-decide && git fetch -q --depth 1 origin REF && git stash -q \
#     ; git checkout -q FETCH_HEAD
#   EXIT_WHEN_DONE=1 setsid nohup bash scripts/pod_shapes.sh > /dev/null 2>&1 &
# then follow runs/pod_shapes.log. Commits only on a Runpod pod (RUNPOD_POD_ID set, or
# FORCE_PUSH=1: never on a dev machine); pushes only if GH_TOKEN is set too. It does not stop
# the pod. Env: FLA_SPEC (default fla-core==0.5.2), VARIANTS (default 8), GH_TOKEN,
# EXIT_WHEN_DONE, FORCE_PUSH.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
FLA_SPEC="${FLA_SPEC:-fla-core==0.5.2}"
VARIANTS="${VARIANTS:-8}"
export HF_HOME="${HF_HOME_POD:-/root/hf}"
export TRITON_CACHE_DIR=/root/triton-cache-shapes  # shared by all probes: only the first compiles
mkdir -p runs "$HF_HOME"
LOG=runs/pod_shapes.log
exec > >(tee -a "$LOG") 2>&1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU" | sed -E 's/NVIDIA //; s/[^A-Za-z0-9]+/-/g' | tr 'A-Z' 'a-z' | sed 's/-$//')
echo "== pod_shapes.sh $(date -u +%FT%TZ) == commit $(git rev-parse --short HEAD)  gpu $GPU ($SLUG)"
STATUS=()
step() { echo; echo "==== $* ($(date -u +%T)) ===="; }
record() { STATUS+=("$1: $2"); echo "[status] $1: $2"; }

# ---- 1. fla ----------------------------------------------------------------------------------
step "1. fla-core"
PYINC=$(python3 -c 'import sysconfig; s = sysconfig.get_default_scheme(); s = "posix_prefix" if s == "posix_local" else s; print(sysconfig.get_paths(scheme=s)["include"])')
if ! command -v gcc >/dev/null || [ ! -f "$PYINC/Python.h" ]; then  # Triton's launcher build
  apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev python3-dev \
    >/dev/null && record compilers_headers ok || record compilers_headers FAIL
fi
python3 -c 'import fla' 2>/dev/null || python3 -m pip install -q "$FLA_SPEC" einops
# Bound by transformers and on the GPU; correctness against the torch fallback is pod_tier0's.
python3 - <<'PY' > runs/shapes-kernels.txt 2>&1
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
cat runs/shapes-kernels.txt
if grep -q "chunk_gated_delta_rule: fla" runs/shapes-kernels.txt \
   && grep -q "fla device_platform: cuda" runs/shapes-kernels.txt \
   && grep -q "KERNEL_RAN" runs/shapes-kernels.txt; then
  record fla_active ok
  FLA_OK=1
else
  record fla_active FAIL
  FLA_OK=0
fi

# ---- 2. probes -------------------------------------------------------------------------------
if [ "$FLA_OK" = 1 ]; then
  python3 -c "from huggingface_hub import snapshot_download as s; s('Cloudflare/clef-flash', revision='$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/float32/manifest.json'))['revision'])")')" \
    > /dev/null && record weights_prefetched ok || record weights_prefetched FAIL

  probe() {  # name, record, extra args...; env comes from the caller's prefix
    local name=$1 rec=$2; shift 2
    step "2. probe $name ($rec)"
    python3 scripts/shape_probe.py "runs/shapes-$name-$SLUG" --record "$rec" \
      --variants "$VARIANTS" --label "$name, $rec, $SLUG" "$@" \
      && record "probe_$name" ok || record "probe_$name" FAIL
  }
  probe default pulse --trace
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True probe alloc pulse
  CUDA_MODULE_LOADING=EAGER probe eager pulse
  TORCH_BLAS_PREFER_CUBLASLT=1 probe lt pulse
  probe long long_2k
fi

# ---- 3. push ---------------------------------------------------------------------------------
step "3. push"
[ ${#STATUS[@]} -gt 0 ] && printf '%s\n' "${STATUS[@]}" | tee runs/pod_shapes_status.txt
BRANCH="m5/shapes-$SLUG-$(date -u +%Y%m%d-%H%M)"
if [ -z "${RUNPOD_POD_ID:-}" ] && [ "${FORCE_PUSH:-0}" != "1" ]; then
  echo "not on a Runpod pod: nothing committed or pushed (results are under runs/)"
else
  git config user.name "worker-decide pod"
  git config user.email "pod@worker-decide.invalid"
  git checkout -q -B "$BRANCH"
  for p in runs/pod_shapes* runs/shapes-*; do
    [ -e "$p" ] && git add -f "$p"
  done
  git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    if [ "$(stat -c %s "$f")" -gt 52428800 ]; then
      echo "not committing $f ($(du -h "$f" | cut -f1)), too large"; git reset -q -- "$f"
    fi
  done
  git commit -q --allow-empty -m "shape probe on $GPU" \
    -m "Produced by scripts/pod_shapes.sh at $(git rev-parse --short HEAD). $(printf '%s; ' "${STATUS[@]}")"
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
