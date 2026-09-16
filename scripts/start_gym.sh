#!/usr/bin/env bash
# Start NeMo Gym servers for fact-checking (Milvus backend by default).
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="${BACKEND:-milvus}"
# shellcheck disable=SC1091
source "$GYM_ROOT/.venv/bin/activate"

if [[ ! -f "$GYM_ROOT/env.yaml" ]]; then
  cp "$FACTCHECK_ROOT/deploy/env.yaml.example" "$GYM_ROOT/env.yaml"
  echo "Wrote $GYM_ROOT/env.yaml from example."
fi

if [[ "$BACKEND" == "milvus" ]]; then
  OVERRIDE="$GYM_ROOT/milvus_override.yaml"
  if [[ ! -f "$OVERRIDE" ]]; then
    cp "$FACTCHECK_ROOT/deploy/milvus_override.yaml.example" "$OVERRIDE"
    echo "Wrote $OVERRIDE — edit milvus_uri before relying on retrieval."
  fi
  CONFIG_EXTRA="resources_servers/fact_checking_reward_model_dev/configs/fact_checking_reward_model_dev.yaml,milvus_override.yaml"
elif [[ "$BACKEND" == "tavily" ]]; then
  if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    echo "TAVILY_API_KEY is required for BACKEND=tavily" >&2
    exit 1
  fi
  OVERRIDE="$GYM_ROOT/tavily_override.yaml"
  if [[ ! -f "$OVERRIDE" ]]; then
    cp "$FACTCHECK_ROOT/deploy/tavily_override.yaml.example" "$OVERRIDE"
  fi
  CONFIG_EXTRA="resources_servers/fact_checking_reward_model_tavily/configs/fact_checking_reward_model_tavily.yaml,tavily_override.yaml"
else
  echo "BACKEND must be milvus or tavily" >&2
  exit 1
fi

cd "$GYM_ROOT"
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml,\
${CONFIG_EXTRA}"

echo "ng_run backend=$BACKEND"
exec ng_run "+config_paths=[$config_paths]"
