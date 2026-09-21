#!/usr/bin/env bash
# Stop policy :8000 and checker :8001 only. Does not touch embed :8002 or Gym.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

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

stop_one policy
stop_one checker
echo "Stopped policy/checker. Embed :8002 left running."
echo "Do not use scripts/stop_servers.sh unless you also want to kill embed."
