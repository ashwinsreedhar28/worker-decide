#!/usr/bin/env bash
# Self-driving oracle run for a Runpod GPU pod. No SSH needed: the pod's start command is
#
#   bash -c "git clone https://github.com/ashwinsreedhar28/worker-decide /root/worker-decide \
#            && cd /root/worker-decide && bash scripts/pod_oracle.sh"
#
# The clone goes to the container disk (/root), not the volume, so a second pod on the same
# volume clones fresh instead of failing on an existing directory.
#
# Pod env (set in the template; GH_TOKEN as a Runpod secret, never in the repo):
#   GH_TOKEN      fine-grained PAT, contents:read+write on this repo only   (required to push)
#   GH_REPO       default ashwinsreedhar28/worker-decide
#   HF_HOME       default /runpod-volume/hf, falls back to /workspace/hf (Runpod's default volume mount)
#   MODEL         default Cloudflare/clef-flash
#   REVISION      optional HF commit sha; default main, recorded in manifest.json
#   DTYPES        default "float32 bfloat16"; the first is written as the oracle (fixtures/),
#                 the rest as runs/<dtype>/ and compared against it
#   STAGE         default all; "dry" stops after the dry run (cheap smoke test)
#   EXIT_WHEN_DONE default 0; 1 = exit after pushing instead of holding the pod open
#
# Everything the run produces is committed to oracle/<model>-<rev7>-<yyyymmdd> and pushed.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Runpod mounts a network volume at /workspace by default; /runpod-volume only if the template says so.
if ! mkdir -p "$HF_HOME" 2>/dev/null; then
  export HF_HOME=/workspace/hf; mkdir -p "$HF_HOME"
  echo "using $HF_HOME (persists only if /workspace is the network volume)"
fi
MODEL="${MODEL:-Cloudflare/clef-flash}"
GH_REPO="${GH_REPO:-ashwinsreedhar28/worker-decide}"
DTYPES="${DTYPES:-float32 bfloat16}"
STAGE="${STAGE:-all}"
LOG=runs/pod_oracle.log
mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

echo "== pod_oracle.sh $(date -u +%FT%TZ) =="
echo "commit $(git rev-parse --short HEAD)  model $MODEL  revision ${REVISION:-main}  dtypes [$DTYPES]"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h "$HF_HOME" 2>/dev/null | tail -1 || true

# ---- environment (venv on the volume so a re-run skips the install) ----------------------
VENV="${VENV:-$(dirname "$HF_HOME")/venv-clef}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "venv unavailable; installing into the system python"; VENV=""; }
fi
# shellcheck disable=SC1091
[ -n "$VENV" ] && source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-oracle.txt
if ! python -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))"; then
  echo "default torch wheel can't init CUDA on this host; falling back to the cu126 wheel"
  python -m pip install -q --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 torchvision==0.26.0
  python -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
fi
python -c "import transformers; from transformers import Qwen3_5ForConditionalGeneration; print('transformers', transformers.__version__)"

# ---- weights -----------------------------------------------------------------------------
python - <<EOF
from huggingface_hub import snapshot_download, HfApi
rev = "${REVISION:-}" or None
info = HfApi().model_info("$MODEL", revision=rev)
print("hub sha", info.sha)
p = snapshot_download("$MODEL", revision=rev)
print("snapshot", p)
EOF

# ---- dry run: encoding + token counts, no weights ----------------------------------------
REQ="fixtures/requests/*.jsonl"
python -m deciders.clef.oracle --requests $REQ --out runs/dry --dry-run ${REVISION:+--revision $REVISION}
python scripts/check_dry_run.py runs/dry
[ "$STAGE" = "dry" ] && { echo "STAGE=dry, stopping"; exit 0; }

# ---- oracle + comparison dtypes ----------------------------------------------------------
set -- $DTYPES
ORACLE_DTYPE="$1"; shift
ORACLE_OUT="fixtures/clef-flash/$ORACLE_DTYPE"
python -m deciders.clef.oracle --requests $REQ --out "$ORACLE_OUT" --dtype "$ORACLE_DTYPE" ${REVISION:+--revision $REVISION}
for dt in "$@"; do   # comparison dtypes are references too (serving arms gate against their own dtype)
  python -m deciders.clef.oracle --requests $REQ --out "fixtures/clef-flash/$dt" --dtype "$dt" ${REVISION:+--revision $REVISION}
  python scripts/parity.py "$ORACLE_OUT" "fixtures/clef-flash/$dt" --json "runs/parity_${dt}.json" > "runs/parity_${dt}.md" || true
  head -12 "runs/parity_${dt}.md"
done
python scripts/parity.py "$ORACLE_OUT" "$ORACLE_OUT" > runs/parity_selfcheck.md

# ---- hosted drift (Pulse 50 only) --------------------------------------------------------
python scripts/hosted_to_run.py fixtures/hosted/clef-flash-workers-ai-2026-10-13.jsonl runs/hosted "$ORACLE_OUT"
python scripts/parity.py "$ORACLE_OUT" runs/hosted --json runs/parity_hosted.json > runs/parity_hosted.md || true
head -12 runs/parity_hosted.md

# ---- push results on a branch -----------------------------------------------------------
REV7=$(python -c "import json;print(json.load(open('$ORACLE_OUT/manifest.json'))['revision'][:7])")
BRANCH="oracle/clef-flash-${REV7}-$(date -u +%Y%m%d)"
git config user.name "worker-decide pod"
git config user.email "pod@worker-decide.invalid"
git checkout -q -b "$BRANCH"
git add "$ORACLE_OUT"
for dt in "$@"; do git add "fixtures/clef-flash/$dt"; done
git add -f runs/*.md runs/*.json runs/pod_oracle.log runs/dry runs/hosted   # runs/ is gitignored for local work
git commit -q -m "oracle: $MODEL $ORACLE_DTYPE on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), rev $REV7" \
  -m "Produced by scripts/pod_oracle.sh. manifest: $ORACLE_OUT/manifest.json. Comparison dtypes: $*."
if [ -z "${RUNPOD_POD_ID:-}" ] && [ "${FORCE_PUSH:-0}" != "1" ]; then
  echo "not on a Runpod pod: results committed locally, not pushed (FORCE_PUSH=1 overrides)"
elif [ -n "${GH_TOKEN:-}" ]; then
  git push -q "https://x-access-token:${GH_TOKEN}@github.com/${GH_REPO}" "$BRANCH"
  echo "pushed $BRANCH"
else
  echo "GH_TOKEN unset: results committed locally on $BRANCH, not pushed"
fi
echo "== done $(date -u +%FT%TZ) =="
# Hold the container open rather than exiting: Runpod restarts the command on exit and the
# restart re-runs everything. (And `[ x = 1 ] && sleep infinity` returns 1 under set -e, so
# the old line killed the script before `exit 0`.) Stop the pod once the branch has landed.
if [ "${EXIT_WHEN_DONE:-0}" = "1" ]; then
  exit 0
fi
echo "holding the pod open; stop it from the Runpod console"
exec sleep infinity
