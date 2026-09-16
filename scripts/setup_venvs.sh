#!/usr/bin/env bash
# Create Gym, embedding, and vLLM virtualenvs.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

echo "== Gym venv (Python 3.12+) =="
cd "$GYM_ROOT"
if [[ ! -x "$GYM_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12
fi
# shellcheck disable=SC1091
source "$GYM_ROOT/.venv/bin/activate"
uv sync --extra dev
uv pip install pymilvus tavily-python

echo "== Embedding venv =="
EMBED_VENV="${EMBED_VENV:-$FACTCHECK_ROOT/.venv-embed}"
if [[ ! -x "$EMBED_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$EMBED_VENV"
fi
uv pip install --python "$EMBED_VENV/bin/python" \
  "sentence-transformers" "fastapi" "uvicorn" "huggingface_hub"

echo "== vLLM venv =="
SERVE_VENV="${SERVE_VENV:-$FACTCHECK_ROOT/.venv-serve}"
if [[ ! -x "$SERVE_VENV/bin/python" ]]; then
  # Reuse a site-wide PyTorch (DLAMI / NGC) when present.
  if python3 -c "import torch" >/dev/null 2>&1; then
    uv venv --python 3.12 --system-site-packages "$SERVE_VENV"
  else
    uv venv --python 3.12 "$SERVE_VENV"
  fi
fi
uv pip install --python "$SERVE_VENV/bin/python" "vllm" "ninja" "huggingface_hub"

echo
echo "Done."
echo "  Gym:   $GYM_ROOT/.venv"
echo "  Embed: $EMBED_VENV"
echo "  vLLM:  $SERVE_VENV"
echo "Copy deploy/env.yaml.example -> $GYM_ROOT/env.yaml before ng_run."
