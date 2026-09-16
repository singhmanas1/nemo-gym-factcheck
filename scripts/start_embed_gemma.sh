#!/usr/bin/env bash
# EmbeddingGemma-300m on one GPU, OpenAI-compatible :8002
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set and no .hf_token file was found." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export EMBED_PORT="${EMBED_PORT:-8002}"
VENV="${EMBED_VENV:-$FACTCHECK_ROOT/.venv-embed}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV. Run scripts/setup_venvs.sh first." >&2
  exit 1
fi

"$VENV/bin/python" - <<'PY'
import os
import sys
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
try:
    me = api.whoami()
    print(f"HF user: {me.get('name')}", flush=True)
except Exception as exc:
    print(f"WARNING: could not resolve token identity: {exc}", flush=True)

candidates = [
    os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m"),
    "unsloth/embeddinggemma-300m",
]
last_err = None
path = None
used = None
for repo in candidates:
    print(f"Downloading {repo} ...", flush=True)
    try:
        try:
            path = snapshot_download(repo, token=token, local_files_only=True)
        except Exception:
            path = snapshot_download(repo, token=token)
        used = repo
        print(f"MODEL_PATH {path}", flush=True)
        break
    except GatedRepoError as exc:
        last_err = exc
        print(f"GATED: {repo}: {exc}", flush=True)
        continue
    except HfHubHTTPError as exc:
        last_err = exc
        print(f"HTTP error for {repo}: {exc}", flush=True)
        continue

if path is None:
    print(
        "Cannot download EmbeddingGemma-300m.\n"
        "Accept the license at https://huggingface.co/google/embeddinggemma-300m\n",
        file=sys.stderr,
    )
    raise SystemExit(2) from last_err

open("/tmp/embed_model_id.txt", "w").write(used)
open("/tmp/embed_model_path.txt", "w").write(path)
PY

export EMBED_MODEL="$(cat /tmp/embed_model_id.txt)"
export EMBED_MODEL_PATH="$(cat /tmp/embed_model_path.txt)"
LOG="${EMBED_LOG:-$FACTCHECK_ROOT/embed_gemma.log}"
PIDFILE="${EMBED_PID:-$FACTCHECK_ROOT/embed_gemma.pid}"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running pid=$(cat "$PIDFILE")  http://0.0.0.0:${EMBED_PORT}/healthz"
  exit 0
fi
echo "Serving $EMBED_MODEL from $EMBED_MODEL_PATH on GPU $CUDA_VISIBLE_DEVICES port $EMBED_PORT"
nohup "$VENV/bin/python" "$FACTCHECK_ROOT/deploy/embed_gemma_server.py" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "pid=$(cat "$PIDFILE")  log=$LOG"
echo "curl -s http://127.0.0.1:${EMBED_PORT}/healthz"
