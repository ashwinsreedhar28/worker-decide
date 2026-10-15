#!/usr/bin/env bash
# Milestone 4 + 5a session on one Runpod GPU pod: SGLang vs our worker on the same GPU.
#
# Deploy a pod FROM THE WORKER IMAGE (ghcr.io/ashwinsreedhar28/worker-decide:clef-<sha7>) with
#   POD_SCRIPT=scripts/pod_m4.sh      (optional REPO_REF=<branch>, default main)
#   GH_TOKEN={{ RUNPOD_SECRET_GH_TOKEN }}
# container disk >= 80 GB (SGLang venv ~15 GB + weights 19 GB + our stack), no volume needed.
# The image's entry point clones the repo and runs this script (see pod_http_parity.sh).
#
# Steps (each logged; a failing step is recorded and the script goes on to the next):
#   1. order oracle: fixtures/requests-order/order.jsonl in fp32 (the truth for those records)
#      and bf16, with our stack                       -> fixtures/clef-flash/float32-order, runs/
#   2. profile our engine (bf16): encode / backbone / head / rest      -> runs/profile-ours-<gpu>
#   3. our worker over HTTP: load test; order records through the gates -> runs/load-ours-<gpu>
#   4. SGLang nightly with Clef's joint head (pinned wheel, own venv): correctness gate on the
#      87 fixtures and the order records, compare against our run and Workers AI, load test
#                                                                   -> runs/sglang-<gpu>, ...
#   5. on a Runpod pod only: commit runs/ and the order oracle to m4/pod-<gpu>-<yyyymmdd-hhmm>
#      (files over 50 MB left out) and push
#
# Accepted risk: the fp32 order oracle on a 48 GB A40 is untried (weights ~38 GB in fp32; the
# records are short). If it fails or OOMs, step 1 records FAIL and the order gates are skipped,
# not failed; everything else runs.
#
# Env: GH_TOKEN (push), STEPS (default "1 2 3 4"), SGLANG_SPEC (default the cookbook's pinned
# nightly), EXIT_WHEN_DONE (default 0 = hold the pod open).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
STEPS="${STEPS-1 2 3 4}"   # empty = no steps (not the default)
SGLANG_SPEC="${SGLANG_SPEC:-sglang==0.5.22.dev20261009+g37ae292e6f}"
SGLANG_INDEX="${SGLANG_INDEX:-https://docs.sglang.ai/whl/cu130/}"
REV=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/float32/manifest.json'))['revision'])")
export HF_HOME="${HF_HOME_POD:-/root/hf}"   # container disk; both engines read the same snapshot
mkdir -p runs "$HF_HOME"
LOG=runs/pod_m4.log
exec > >(tee -a "$LOG") 2>&1

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU" | sed -E 's/NVIDIA //; s/[^A-Za-z0-9]+/-/g' | tr 'A-Z' 'a-z' | sed 's/-$//')
echo "== pod_m4.sh $(date -u +%FT%TZ) =="
echo "commit $(git rev-parse --short HEAD)  image ${WORKER_DECIDE_COMMIT:-?}  gpu $GPU ($SLUG)  rev ${REV:0:7}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
STATUS=()
step() { echo; echo "==== $* ($(date -u +%T)) ===="; }
record() { STATUS+=("$1: $2"); echo "[status] $1: $2"; }
want() { [[ " $STEPS " == *" $1 "* ]]; }

http_code() {  # url -> HTTP status, 0 if unreachable (the image has no curl)
  python3 -c 'import sys, urllib.request, urllib.error
try: print(urllib.request.urlopen(sys.argv[1], timeout=5).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception: print(0)' "$1"
}

wait_200() {  # url, timeout_s [, pid]: exactly 200 (our /ping is 204 while loading); a 500 or
              # a dead pid ends the wait early
  local t0=$SECONDS code
  while true; do
    code=$(http_code "$1")
    [ "$code" = 200 ] && return 0
    [ "$code" = 500 ] && return 1
    [ -n "${3:-}" ] && ! kill -0 "$3" 2>/dev/null && return 1
    (( SECONDS - t0 > $2 )) && return 1
    sleep 3
  done
}

order_oracle_ok() {  # manifest.json is written last; a bare directory means the oracle died, and
                    # error records (e.g. an fp32 OOM mid-forward) mean it is not a truth
  python3 -c 'import json, sys
m = json.load(open("fixtures/clef-flash/float32-order/manifest.json"))
sys.exit(0 if m["counts"]["error"] == 0 and m["counts"]["ok"] > 0 else 1)' 2>/dev/null
}

wait_gpu_free() {  # until < 1 GiB is in use (a killed process can hold memory for a moment)
  local t0=$SECONDS used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 1024 ] && return 0
    (( SECONDS - t0 > 120 )) && { echo "GPU still has ${used} MiB in use"; return 1; }
    sleep 2
  done
}

# ---- 1. order oracle (fp32 truth + bf16) -----------------------------------------------------
if want 1; then
  step "1. order oracle"
  python3 -m deciders.clef.oracle --requests fixtures/requests-order/order.jsonl \
    --out fixtures/clef-flash/float32-order --dtype float32 --revision "$REV" \
    && record order_oracle_fp32 ok || record order_oracle_fp32 FAIL
  python3 -m deciders.clef.oracle --requests fixtures/requests-order/order.jsonl \
    --out runs/order-bf16-oracle-$SLUG --dtype bfloat16 --revision "$REV" \
    && record order_oracle_bf16 ok || record order_oracle_bf16 FAIL
  if order_oracle_ok; then
    python3 scripts/order_findings.py fixtures/clef-flash/float32-order runs/order_findings.json \
      > runs/order_findings.md && cat runs/order_findings.md || record order_findings FAIL
  fi
fi

# ---- 2. profile our engine -------------------------------------------------------------------
if want 2; then
  step "2. profile our engine (bf16)"
  python3 scripts/profile_engine.py "runs/profile-ours-$SLUG" --reps 20 --warmup 3 --trace \
    && record profile ok || record profile FAIL
fi

# ---- 3. our worker over HTTP -----------------------------------------------------------------
if want 3; then
  step "3. our worker: load test + order records"
  env -u POD_SCRIPT TRANSPORT=http PORT=8000 python3 -m worker > runs/worker.log 2>&1 &
  WPID=$!
  if wait_200 http://127.0.0.1:8000/ping 900 $WPID; then
    python3 scripts/loadtest.py http://127.0.0.1:8000 "runs/load-ours-$SLUG" --label "ours-$SLUG" \
      && record load_ours ok || record load_ours FAIL
    if order_oracle_ok; then
      python3 scripts/parity_http.py http://127.0.0.1:8000 "runs/http-order-$SLUG" \
        --requests fixtures/requests-order/order.jsonl --reference fixtures/clef-flash/float32-order \
        > "runs/parity_http_order_$SLUG.md" && record parity_ours_order PASS || record parity_ours_order FAIL
    fi
  else
    record worker_start FAIL; tail -n 40 runs/worker.log
  fi
  kill $WPID 2>/dev/null; wait $WPID 2>/dev/null
  wait_gpu_free || record gpu_free_after_worker FAIL
fi

# ---- 4. SGLang -------------------------------------------------------------------------------
if want 4; then
  step "4. SGLang ($SGLANG_SPEC)"
  # The worker image is CUDA "base" without compilers. Triton (SGLang's linear-attention and
  # triton attention kernels) builds a small C launcher with gcc on first use, and flashinfer
  # may JIT with nvcc (Oct 15 audit). Best effort; versions recorded.
  if ! command -v gcc >/dev/null; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev \
      >/dev/null && record compilers ok || record compilers FAIL
  fi
  # Decided on its own, not with gcc: on an A40 pod that ran pod_shapes.sh/pod_tier1.sh first
  # (gcc present, nvcc not), SGLang's JIT activation kernel for sm_86 failed with
  # "/usr/local/cuda/bin/nvcc: not found" (Oct 15).
  if [ ! -x /usr/local/cuda/bin/nvcc ]; then
    { apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-nvcc-13-0; } \
      >/dev/null 2>&1 && record nvcc ok || record nvcc "FAIL (SGLang JIT kernels will not build)"
  fi
  # Unconditional, so a rerun in the same container (compilers already there) gets it too.
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
  fi
  { gcc --version | head -1; command -v nvcc && nvcc --version | tail -2
    [ -x /opt/sgl/bin/ninja ] && echo "ninja $(/opt/sgl/bin/ninja --version) (venv)"; } \
    > runs/sglang-toolchain.txt 2>&1
  if [ ! -x /opt/sgl/bin/python ]; then
    python3 -m pip install -q uv   # the image sets PIP_BREAK_SYSTEM_PACKAGES=1
    uv venv /opt/sgl --python 3.10 \
      && uv pip install --python /opt/sgl/bin/python --prerelease=allow \
           --index-strategy unsafe-best-match --extra-index-url "$SGLANG_INDEX" "$SGLANG_SPEC" \
      && record sglang_install ok || record sglang_install FAIL
  fi
  uv pip freeze --python /opt/sgl/bin/python > runs/sglang-freeze.txt 2>/dev/null || true
  /opt/sgl/bin/python - <<'PY' > runs/sglang-head-check.txt 2>&1 || true
import importlib.metadata as md, pathlib, sglang
root = pathlib.Path(sglang.__file__).parent
print("sglang", md.version("sglang"), root)
for rel in ["srt/layers/joint_schema_head.py", "srt/entrypoints/systemone/joint_schema.py"]:
    print(rel, "PRESENT" if (root / rel).exists() else "ABSENT")
PY
  cat runs/sglang-head-check.txt
  SGL_PID=""
  start_sglang() {  # extra args...
    # The venv's bin on PATH: SGLang's JIT kernels run `ninja` (pip-installed into the venv) by
    # name. Oct 15 H100 run: both starts failed with FileNotFoundError 'ninja' without this.
    PATH="/opt/sgl/bin:$PATH" /opt/sgl/bin/python -m sglang.launch_server --model-path Cloudflare/clef-flash \
      --revision "$REV" --host 127.0.0.1 --port 30000 "$@" > runs/sglang.log 2>&1 &
    SGL_PID=$!
    wait_200 http://127.0.0.1:30000/health 1800 $SGL_PID
  }
  BACKEND=default
  if ! start_sglang; then
    echo "--- sglang default start failed; log tail:"; tail -n 60 runs/sglang.log
    cp runs/sglang.log runs/sglang-default-failed.log
    kill $SGL_PID 2>/dev/null; wait $SGL_PID 2>/dev/null
    wait_gpu_free || true
    BACKEND=triton
    start_sglang --attention-backend triton || { record sglang_start FAIL; tail -n 60 runs/sglang.log; }
  fi
  echo "sglang attention backend: $BACKEND" | tee runs/sglang-backend.txt
  if [ "$(http_code http://127.0.0.1:30000/health)" = 200 ]; then
    record sglang_start "ok ($BACKEND)"
    # Our own bf16 answers on this GPU, if we have them (fixtures/clef-flash/bfloat16 is the H100
    # oracle run; runs/http-a40 the A40 worker run); otherwise the A40 run, read as cross-GPU.
    OURS=runs/http-a40; OURS_LABEL=ours-A40
    BF16_GPU=$(python3 -c "import json;print(json.load(open('fixtures/clef-flash/bfloat16/manifest.json'))['gpu'])")
    if [ "$GPU" = "$BF16_GPU" ]; then OURS=fixtures/clef-flash/bfloat16; OURS_LABEL="ours-$SLUG"; fi
    python3 scripts/parity_sglang.py http://127.0.0.1:30000 "runs/sglang-$SLUG" \
      --also "$OURS" --json "runs/parity_sglang_$SLUG.json" \
      > "runs/parity_sglang_$SLUG.md" && record parity_sglang PASS || record parity_sglang FAIL
    head -20 "runs/parity_sglang_$SLUG.md"
    if order_oracle_ok; then
      python3 scripts/parity_sglang.py http://127.0.0.1:30000 "runs/sglang-order-$SLUG" \
        --requests fixtures/requests-order/order.jsonl --reference fixtures/clef-flash/float32-order \
        > "runs/parity_sglang_order_$SLUG.md" && record parity_sglang_order PASS || record parity_sglang_order FAIL
    fi
    python3 scripts/hosted_to_run.py fixtures/hosted/clef-flash-workers-ai-2026-10-13.jsonl \
      runs/hosted fixtures/clef-flash/float32 2>/dev/null
    python3 scripts/compare_runs.py fixtures/clef-flash/float32 "runs/sglang-$SLUG" runs/hosted \
      --labels "SGLang-$SLUG,WorkersAI" > "runs/compare_sglang_vs_hosted_$SLUG.md" || true
    python3 scripts/compare_runs.py fixtures/clef-flash/float32 "$OURS" "runs/sglang-$SLUG" \
      --labels "$OURS_LABEL,SGLang-$SLUG" > "runs/compare_ours_vs_sglang_$SLUG.md" || true
    python3 scripts/loadtest.py http://127.0.0.1:30000 "runs/load-sglang-$SLUG" --label "sglang-$SLUG" \
      --model Cloudflare/clef-flash && record load_sglang ok || record load_sglang FAIL
  fi
  kill $SGL_PID 2>/dev/null; wait $SGL_PID 2>/dev/null
fi

# ---- 5. push ---------------------------------------------------------------------------------
step "5. push"
[ ${#STATUS[@]} -gt 0 ] && printf '%s\n' "${STATUS[@]}" | tee runs/pod_m4_status.txt
BRANCH="m4/pod-$SLUG-$(date -u +%Y%m%d-%H%M)"   # with the time: a same-day rerun collided (Oct 15)
# Commit and push only from a Runpod pod (RUNPOD_POD_ID is set there): Oct 15, a local smoke
# test in a workspace with a GH_TOKEN pushed a junk branch. FORCE_PUSH=1 overrides.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ "${FORCE_PUSH:-0}" != "1" ]; then
  echo "not on a Runpod pod: nothing committed or pushed (results are under runs/)"
else
  git config user.name "worker-decide pod"
  git config user.email "pod@worker-decide.invalid"
  git checkout -q -B "$BRANCH"
  for p in runs/*.md runs/*.json runs/*.txt runs/*.log runs/profile-* runs/load-* runs/sglang-* \
           runs/http-order-* runs/order-* fixtures/clef-flash/float32-order; do
    [ -e "$p" ] && git add -f "$p"
  done
  # GitHub refuses files over 100 MB and the whole push with them; keep anything over 50 MB out.
  git diff --cached --name-only -z | while IFS= read -r -d '' f; do
    if [ "$(stat -c %s "$f")" -gt 52428800 ]; then
      echo "not committing $f ($(du -h "$f" | cut -f1)), too large"; git reset -q -- "$f"
    fi
  done
  git commit -q --allow-empty -m "m4 pod session on $GPU: SGLang vs ours, profile, load, order oracle" \
    -m "Produced by scripts/pod_m4.sh at $(git rev-parse --short HEAD). $(printf '%s; ' "${STATUS[@]}")"
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
