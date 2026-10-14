# worker-decide: Clef-flash decision worker for Runpod Serverless (queue or load-balancing).
#
# CUDA 13.0, the same as the oracle environment (torch 2.11.0+cu130) and worker-vllm's Hub
# config (allowedCudaVersions ["13.0"]); the endpoint must filter hosts to CUDA 13.0 or newer.
# Weights are not baked: they load from the network volume (HF_HOME below) or, without one,
# download to the container disk on first start (~18 GB bf16, needs containerDiskInGb >= 60).
#
#   docker build --platform linux/amd64 -t worker-decide .
#   docker run --gpus all -e TRANSPORT=http -p 8000:8000 -v hf:/runpod-volume worker-decide

ARG BASE_IMAGE=nvidia/cuda:13.0.1-base-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-oracle.txt pyproject.toml ./
COPY deciders ./deciders
COPY worker ./worker
COPY handler.py ./
RUN python3 -m pip install -r requirements-oracle.txt \
    && python3 -m pip install ".[worker]"
# Fail the build, not the first cold start, if the pins drifted.
RUN python3 -c "import torch, transformers; \
    assert torch.__version__ == '2.11.0+cu130', torch.__version__; \
    assert transformers.__version__ == '5.10.2', transformers.__version__; \
    import fastapi, uvicorn, runpod"

ARG BASE_PATH=/runpod-volume
ARG WORKER_DECIDE_COMMIT=""
ENV BASE_PATH=${BASE_PATH} \
    HF_HOME=${BASE_PATH}/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODEL_NAME=Cloudflare/clef-flash \
    MODEL_REVISION=8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 \
    DTYPE=bfloat16 \
    TRANSPORT=queue \
    PORT=8000 \
    PORT_HEALTH=8000 \
    WORKER_DECIDE_COMMIT=${WORKER_DECIDE_COMMIT}

# TRANSPORT=queue -> runpod.serverless handler; TRANSPORT=http -> uvicorn on $PORT.
# CMD, not ENTRYPOINT, so a pod template's start command can replace it (scripts/pod_http_parity.sh).
CMD ["python3", "-m", "worker"]
