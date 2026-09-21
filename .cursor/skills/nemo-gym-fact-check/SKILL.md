---
name: nemo-gym-fact-check
description: Set up and run the NeMo Gym fact-checking evaluation pipeline with Milvus or Tavily retrieval. Use when running fact-checking evaluations, collecting factuality rollouts, serving Nemotron on 32GB GPUs, or configuring ng_run / ng_collect_rollouts.
---

# NeMo Gym fact-checking

Follow `README.md` in the repo root. Use `scripts/` instead of ad-hoc `vllm serve` flags.

## Pipeline

```
Prompt → policy :8000 → checker :8001 ──search_wiki──► embed :8002 ──► Milvus :19530
                                                              or Tavily
Policy writes [Factual Errors] → verify() → factuality_f1_score
```

- Same Lightning weights on two GPUs (`:8000` policy / `:8001` YES/NO matcher). Default: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (`qwen3_coder`, `--reasoning-parser nemotron_v3`). 9B-v2: `POLICY_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2`. Not `Nano-8B-v2` (HF 404).
- `search_wiki` is still Milvus (or Tavily): one HTTP call per **unique** query. Policy cap: `max_search_calls: 3`, duplicate queries skipped, then `tool_choice: none` writes the tagged verdict.
- F1 vs gold `expected_errors`, not vs FineWeb. Empty gold: blank error box → F1 1. Filled gold: YES/NO matcher vs the whole error box; `num_errors` is **line-count** (no count LLM).
- Collect **appends**. Use a new `OUTPUT_JSONL`. Sidecar `*.metrics.jsonl` has per-sample F1 plus `t_*` step times. Python-only changes: bounce Gym (`stop_gym.sh` / `start_gym.sh`), not vLLM.

## 32GB GPUs (9B only)

Lightning 30B needs large cards (this harness defaults to H100, `--max-model-len 131072`). For 9B on 32GB use `scripts/start_vllm_policy_checker.sh` with `POLICY_MODEL=...9B-v2` and `MAX_MODEL_LEN=8192` (`0.70` util, `--enforce-eager`). Empty `curl :8000/v1/models` → read `vllm_policy.log`.

## Milvus

Gym talks to **19530**. ClusterIPs are not public. Tunnel or LoadBalancer; set `milvus_uri` in `gym/milvus_override.yaml` (copy from `deploy/milvus_override.yaml.example`). Do not commit that file (local URI). Embeddings go to local `:8002` via `milvus_embedding_base_url`. Keep `summarize_retrieval_results: false` for raw FineWeb text.

## Commands (repo root)

```bash
export HF_TOKEN=hf_...
./scripts/setup_venvs.sh
cp deploy/env.yaml.example gym/env.yaml
cp deploy/milvus_override.yaml.example gym/milvus_override.yaml
# edit milvus_uri
./scripts/start_embed_gemma.sh
./scripts/start_vllm_policy_checker.sh
./scripts/check_endpoints.sh
./scripts/start_gym.sh
# other terminal:
./scripts/collect_rollouts.sh
```

Mixed empty+filled slice (skip Chris Rock crossword):

```bash
python3 scripts/extract_mixed_gold_slice.py data/rlhf24_final_audited_dataset.jsonl data/rlhf24_mixed10.jsonl
NUM_SAMPLES_IN_PARALLEL=2 INPUT_JSONL=data/rlhf24_mixed10.jsonl OUTPUT_JSONL=./factcheck_output_mixed10.jsonl ./scripts/collect_rollouts.sh
python3 scripts/summarize_slice_timings.py ./factcheck_output_mixed10.jsonl
```

Do not iterate on a single crossword row. Tavily: `BACKEND=tavily ./scripts/start_gym.sh` with `TAVILY_API_KEY`.

## Do not

- Serve 32k context on 32GB for 9B hybrid.
- Treat ClusterIP as a public URL.
- Commit `.hf_token`, `gym/env.yaml`, `gym/milvus_override.yaml`, or `factcheck_output*.jsonl`.
- Start a second vLLM if `scripts/start_*.sh` reports already running.
- Bounce vLLM for Gym Python-only changes; bounce Gym only (`./scripts/stop_gym.sh`).
