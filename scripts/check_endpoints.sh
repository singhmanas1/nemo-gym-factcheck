#!/usr/bin/env bash
# Probe local model servers (and optional Milvus URI).
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check() {
  local url="$1"
  echo -n "$url  "
  if curl -sS -m 5 -o /tmp/factcheck_curl_body -w "%{http_code}\n" "$url"; then
    head -c 200 /tmp/factcheck_curl_body; echo
  else
    echo "FAILED"
  fi
}

check "http://127.0.0.1:8000/v1/models"
check "http://127.0.0.1:8001/v1/models"
check "http://127.0.0.1:8002/healthz"

if [[ -n "${MILVUS_URI:-}" ]]; then
  echo "Milvus $MILVUS_URI"
  curl -sS -m 5 -v "$MILVUS_URI" || true
fi
