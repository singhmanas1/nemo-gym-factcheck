#!/usr/bin/env bash
# Create a public GitHub repo from this tree. Run once after: gh auth login
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh not on PATH. Install: https://cli.github.com/" >&2
  exit 1
fi

cd "$FACTCHECK_ROOT"
NAME="${GITHUB_REPO_NAME:-nemo-gym-factcheck}"
if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged in. On this machine run:" >&2
  echo "  gh auth login -h github.com -p https -w" >&2
  exit 1
fi

gh repo create "$NAME" \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "NeMo Gym fact-checking E2E harness (Nemotron 9B, EmbeddingGemma, Milvus/Tavily)"

gh repo view --web
