#!/usr/bin/env bash
# GPU 1: policy Nemotron :8000
# GPU 2: checker/judge Nemotron :8001
#
# Default: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (H100 recipe).
# 9B-v2:   POLICY_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
# 32GB 9B: export MAX_MODEL_LEN=8192
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set and no .hf_token file was found." >&2
  exit 1
fi

resolve_cuda_home() {
  local candidate
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "$CUDA_HOME"
    return
  fi
  for candidate in \
    /usr/local/cuda \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.4 \
    /opt/pytorch/cuda \
    /usr/lib/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      echo "$candidate"
      return
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    dirname "$(dirname "$(command -v nvcc)")"
    return
  fi
  echo "${CUDA_HOME:-/usr/local/cuda}"
}

export CUDA_HOME="$(resolve_cuda_home)"
export CUDA_PATH="$CUDA_HOME"
SERVE_VENV="${SERVE_VENV:-$FACTCHECK_ROOT/.venv-serve}"
export PATH="$SERVE_VENV/bin:$CUDA_HOME/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NVCC_BIN="${CUDA_HOME}/bin/nvcc"
MODEL="${POLICY_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
VLLM_BIN="${VLLM_BIN:-$SERVE_VENV/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "Missing $VLLM_BIN. Run scripts/setup_venvs.sh first." >&2
  exit 1
fi

is_lightning() {
  [[ "$MODEL" == *Nemotron-3.5-Lightning* ]]
}

echo "nvcc=$(command -v nvcc || true) ninja=$(command -v ninja || true) CUDA_HOME=$CUDA_HOME"

echo "Ensuring $MODEL is in the Hugging Face cache ..."
POLICY_MODEL="$MODEL" "$SERVE_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["POLICY_MODEL"]
token = os.environ["HF_TOKEN"]
try:
    path = snapshot_download(repo, token=token, local_files_only=True)
except Exception:
    path = snapshot_download(repo, token=token)
print("MODEL_PATH", path, flush=True)
open("/tmp/nemotron_model_path.txt", "w").write(path)
PY

EXTRA_ARGS=()
LEN_ARGS=()
if is_lightning; then
  # Official H100 BF16 recipe wants --mamba-backend flashinfer, which JIT-compiles
  # via nvcc. This box has no /opt/pytorch/cuda/bin/nvcc, so default to Triton
  # (vLLM's built-in SSU backend). Set MAMBA_BACKEND=flashinfer once nvcc exists.
  TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
  if [[ -z "${MAMBA_BACKEND:-}" ]]; then
    if [[ -x "$NVCC_BIN" ]]; then
      MAMBA_BACKEND=flashinfer
    else
      MAMBA_BACKEND=triton
    fi
  fi
  if [[ "$MAMBA_BACKEND" == flashinfer ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
  else
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  fi
  EXTRA_ARGS+=(
    --enable-prefix-caching
    --mamba-backend "$MAMBA_BACKEND"
    --mamba-ssm-cache-dtype float16
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --reasoning-parser nemotron_v3
  )
  echo "Lightning mamba-backend=$MAMBA_BACKEND nvcc=$NVCC_BIN (exists=$([[ -x $NVCC_BIN ]] && echo yes || echo no))"
  # 256k is the HF H100 cap; 131k is enough for this eval and leaves KV headroom
  # for two replicas plus EmbeddingGemma on GPU 0.
  if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN=131072
  fi
  LEN_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  echo "Lightning 30B-A3B: tool-call-parser=$TOOL_CALL_PARSER reasoning-parser=nemotron_v3 max-model-len=$MAX_MODEL_LEN"
else
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  TOOL_PARSER_PLUGIN="${TOOL_PARSER_PLUGIN:-$FACTCHECK_ROOT/deploy/nemotron_json_tool_parser.py}"
  TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-nemotron_json}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
  if [[ ! -f "$TOOL_PARSER_PLUGIN" ]]; then
    echo "Missing tool parser plugin: $TOOL_PARSER_PLUGIN" >&2
    exit 1
  fi
  EXTRA_ARGS+=(
    --mamba_ssm_cache_dtype float32
    --enforce-eager
    --enable-auto-tool-choice
    --tool-parser-plugin "$TOOL_PARSER_PLUGIN"
    --tool-call-parser "$TOOL_CALL_PARSER"
  )
  if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    LEN_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    echo "max-model-len=$MAX_MODEL_LEN"
  else
    echo "max-model-len omitted (9B-v2 default 131072)"
  fi
  echo "Tool calling: --tool-call-parser $TOOL_CALL_PARSER plugin $TOOL_PARSER_PLUGIN"
fi

stop_one() {
  local name="$1"
  local pidfile="$FACTCHECK_ROOT/vllm_${name}.pid"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name pid=$pid"
    kill "$pid" || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}

if [[ "${STOP_EXISTING:-1}" == "1" ]]; then
  stop_one policy
  stop_one checker
fi

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
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    HF_TOKEN="$HF_TOKEN" \
    HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
    nohup "$VLLM_BIN" serve "$MODEL" \
      --trust-remote-code \
      --tensor-parallel-size 1 \
      "${LEN_ARGS[@]}" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      "${EXTRA_ARGS[@]}" \
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
echo "Leave embed :8002 running. Bounce Gym after the model name change."
