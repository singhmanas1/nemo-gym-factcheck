#!/usr/bin/env bash
# Shared paths for every launcher. Source this from other scripts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FACTCHECK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export GYM_ROOT="${GYM_ROOT:-$FACTCHECK_ROOT/gym}"
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$FACTCHECK_ROOT/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$FACTCHECK_ROOT/.hf_token")"
elif [[ -z "$HF_TOKEN" && -f "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HOME/.hf_token")"
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export HF_HUB_DISABLE_XET=1
