#!/usr/bin/env bash
# Collect fact-check rollouts. Default input is the bundled example JSONL.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="${BACKEND:-milvus}"
INPUT="${INPUT_JSONL:-$FACTCHECK_ROOT/examples/factcheck_example.jsonl}"
OUTPUT="${OUTPUT_JSONL:-$FACTCHECK_ROOT/factcheck_output.jsonl}"

if [[ "$BACKEND" == "tavily" ]]; then
  AGENT="${AGENT_NAME:-fact_checking_reward_model_tavily_simple_agent}"
else
  AGENT="${AGENT_NAME:-fact_checking_reward_model_dev_simple_agent}"
fi

# shellcheck disable=SC1091
source "$GYM_ROOT/.venv/bin/activate"
cd "$GYM_ROOT"

echo "agent=$AGENT"
echo "input=$INPUT"
echo "output=$OUTPUT"
ng_collect_rollouts \
  +agent_name="$AGENT" \
  +input_jsonl_fpath="$INPUT" \
  +output_jsonl_fpath="$OUTPUT"
