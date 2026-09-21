#!/usr/bin/env bash
# Stop leftover Gym/Ray on :11000. Does not touch vLLM :8000/:8001 or embed :8002.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "Freeing Gym head :11000 (leave vLLM/embed running) ..."
fuser -k 11000/tcp 2>/dev/null || true

if [[ -x "$GYM_ROOT/.venv/bin/ray" ]]; then
  "$GYM_ROOT/.venv/bin/ray" stop --force 2>/dev/null || true
else
  ray stop --force 2>/dev/null || true
fi

pkill -f '/nemo-gym-factcheck/gym/.venv' 2>/dev/null || true
# Per-server Gym venvs (policy/judge adapters, resource server, agent).
# Do NOT match .venv-serve / vllm on :8000 :8001 or embed :8002.
pkill -f '/nemo-gym-factcheck/gym/responses_api_models/.venv' 2>/dev/null || true
pkill -f '/nemo-gym-factcheck/gym/responses_api_models/vllm_model/.venv' 2>/dev/null || true
pkill -f '/nemo-gym-factcheck/gym/responses_api_agents/.venv' 2>/dev/null || true
pkill -f '/nemo-gym-factcheck/gym/responses_api_agents/simple_agent/.venv' 2>/dev/null || true
pkill -f '/nemo-gym-factcheck/gym/resources_servers/.venv' 2>/dev/null || true
pkill -f '/nemo-gym-factcheck/gym/resources_servers/fact_checking_reward_model_dev/.venv' 2>/dev/null || true
pkill -f 'nemo_gym' 2>/dev/null || true
pkill -f 'ng_run' 2>/dev/null || true
pkill -f 'fact_checking_reward_model_dev' 2>/dev/null || true
pkill -f 'responses_api_agents/simple_agent' 2>/dev/null || true
sleep 3

if ss -lptn 2>/dev/null | grep -q ':11000'; then
  echo "WARNING: :11000 still listening:" >&2
  ss -lptn | grep 11000 || true
  echo "Run: lsof -iTCP:11000 -sTCP:LISTEN" >&2
  exit 1
fi
echo "11000 is free."
echo "Check vLLM/embed still answer:"
echo "  curl -sS -m 3 http://127.0.0.1:8000/v1/models | head -c 80"
echo "  curl -sS -m 3 http://127.0.0.1:8002/healthz"
