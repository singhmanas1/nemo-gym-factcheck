#!/usr/bin/env bash
# GPU 1: policy Nemotron :8000
# GPU 2: checker/judge Nemotron :8001
# Use --enforce-eager and 8k context on 32GB cards (see README).
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set and no .hf_token file was found." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/cuda}"
export CUDA_PATH="$CUDA_HOME"
SERVE_VENV="${SERVE_VENV:-$FACTCHECK_ROOT/.venv-serve}"
export PATH="$SERVE_VENV/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${POLICY_MODEL:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}"
VLLM_BIN="${VLLM_BIN:-$SERVE_VENV/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "Missing $VLLM_BIN. Run scripts/setup_venvs.sh first." >&2
  exit 1
fi

echo "nvcc=$(command -v nvcc || true) ninja=$(command -v ninja || true) CUDA_HOME=$CUDA_HOME"

echo "Ensuring $MODEL is in the Hugging Face cache ..."
"$SERVE_VENV/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("POLICY_MODEL", "$MODEL")
token = os.environ["HF_TOKEN"]
try:
    path = snapshot_download(repo, token=token, local_files_only=True)
except Exception:
    path = snapshot_download(repo, token=token)
print("MODEL_PATH", path, flush=True)
PY

start_one() {
  local gpu="$1" port="$2" name="$3"
  local log="$FACTCHECK_ROOT/vllm_${name}.log"
  local pidfile="$FACTCHECK_ROOT/vllm_${name}.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Already running $name pid=$(cat "$pidfile") :$port"
    return 0
  fi
  rm -f "$pidfile"
  : >"$log"
  echo "Starting $name on GPU $gpu port $port"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA_HOME="$CUDA_HOME" \
    CUDA_PATH="$CUDA_PATH" \
    PATH="$PATH" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    HF_TOKEN="$HF_TOKEN" \
    HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
    nohup "$VLLM_BIN" serve "$MODEL" \
      --trust-remote-code \
      --mamba_ssm_cache_dtype float32 \
      --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN:-8192}" \
      --max-num-seqs "${MAX_NUM_SEQS:-8}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.70}" \
      --enforce-eager \
      --port "$port" \
      --host 0.0.0.0 \
      --served-model-name "$MODEL" \
      >>"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "pid=$(cat "$pidfile") log=$log"
}

start_one "${POLICY_GPU:-1}" "${POLICY_PORT:-8000}" policy
start_one "${CHECKER_GPU:-2}" "${CHECKER_PORT:-8001}" checker
echo
echo "Watch for Application startup complete:"
echo "  tail -f $FACTCHECK_ROOT/vllm_policy.log $FACTCHECK_ROOT/vllm_checker.log"
echo "Policy  http://127.0.0.1:${POLICY_PORT:-8000}/v1/models"
echo "Checker http://127.0.0.1:${CHECKER_PORT:-8001}/v1/models"
