#!/usr/bin/env bash
# Stop background embed + vLLM processes started by this repo's scripts.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

stop_pidfile() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping pid $pid ($pidfile)"
    kill "$pid" || true
  fi
  rm -f "$pidfile"
}

stop_pidfile "$FACTCHECK_ROOT/embed_gemma.pid"
stop_pidfile "$FACTCHECK_ROOT/vllm_policy.pid"
stop_pidfile "$FACTCHECK_ROOT/vllm_checker.pid"
echo "Stopped local servers (Gym ng_run must be Ctrl+C'd in its terminal)."
