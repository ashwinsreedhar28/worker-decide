#!/usr/bin/env bash
# Run several pod session scripts in order on one pod, each with its own log, commit and push.
#
#   POD_SCRIPT=scripts/pod_chain.sh  POD_CHAIN="pod_shapes.sh pod_tier1.sh"  (the default chain)
#
# or on a pod that has the repo: EXIT_WHEN_DONE=1 setsid nohup bash scripts/pod_chain.sh
# > /dev/null 2>&1 &  then follow runs/pod_chain.log and each script's own log. A script that
# fails does not stop the chain. Each script runs with EXIT_WHEN_DONE=1; the chain itself holds
# the pod open at the end unless EXIT_WHEN_DONE=1. It does not stop the pod.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
exec > >(tee -a runs/pod_chain.log) 2>&1
echo "== pod_chain.sh $(date -u +%FT%TZ) == commit $(git rev-parse --short HEAD) chain: ${POD_CHAIN:-pod_shapes.sh pod_tier1.sh}"
for s in ${POD_CHAIN:-pod_shapes.sh pod_tier1.sh}; do
  echo "---- $s ($(date -u +%T))"
  EXIT_WHEN_DONE=1 bash "scripts/$s"
  echo "---- $s exited $? ($(date -u +%T))"
done
echo "== chain done $(date -u +%FT%TZ) =="
if [ "${EXIT_WHEN_DONE:-0}" = "1" ]; then exit 0; fi
echo "holding the pod open; stop it from the Runpod console"
exec sleep infinity
