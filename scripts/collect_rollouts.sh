#!/usr/bin/env bash
# Collect fact-check rollouts. Default input is the bundled example JSONL.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BACKEND="${BACKEND:-milvus}"

if [[ "$BACKEND" == "tavily" ]]; then
  AGENT="${AGENT_NAME:-fact_checking_reward_model_tavily_simple_agent}"
  DEFAULT_INPUT="$FACTCHECK_ROOT/examples/factcheck_example.jsonl"
  DEFAULT_OUTPUT="$FACTCHECK_ROOT/factcheck_output.jsonl"
elif [[ "$BACKEND" == "exa" ]]; then
  AGENT="${AGENT_NAME:-fact_checking_reward_model_exa_simple_agent}"
  DEFAULT_INPUT="$FACTCHECK_ROOT/data/rlhf24_final_audited_dataset.jsonl"
  DEFAULT_OUTPUT="$FACTCHECK_ROOT/factcheck_output_exa.jsonl"
elif [[ "$BACKEND" == "milvus" ]]; then
  AGENT="${AGENT_NAME:-fact_checking_reward_model_dev_simple_agent}"
  DEFAULT_INPUT="$FACTCHECK_ROOT/examples/factcheck_example.jsonl"
  DEFAULT_OUTPUT="$FACTCHECK_ROOT/factcheck_output.jsonl"
else
  echo "BACKEND must be milvus, exa, or tavily" >&2
  exit 1
fi

# Paths are from the repo root. ng_collect_rollouts runs with cwd gym/.
resolve_repo_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$FACTCHECK_ROOT/${p#./}"
  fi
}

INPUT="$(resolve_repo_path "${INPUT_JSONL:-$DEFAULT_INPUT}")"
OUTPUT="$(resolve_repo_path "${OUTPUT_JSONL:-$DEFAULT_OUTPUT}")"

# shellcheck disable=SC1091
source "$GYM_ROOT/.venv/bin/activate"
cd "$GYM_ROOT"

echo "agent=$AGENT"
echo "input=$INPUT"
echo "output=$OUTPUT"
COLLECT_ARGS=(
  +agent_name="$AGENT"
  +input_jsonl_fpath="$INPUT"
  +output_jsonl_fpath="$OUTPUT"
)
if [[ -n "${NUM_SAMPLES_IN_PARALLEL:-}" ]]; then
  echo "num_samples_in_parallel=$NUM_SAMPLES_IN_PARALLEL"
  COLLECT_ARGS+=(+num_samples_in_parallel="$NUM_SAMPLES_IN_PARALLEL")
fi
if [[ -n "${LIMIT:-}" ]]; then
  echo "limit=$LIMIT"
  COLLECT_ARGS+=(+limit="$LIMIT")
fi
ng_collect_rollouts "${COLLECT_ARGS[@]}"
