---
name: nemo-gym-fact-check
description: Set up and run the NeMo Gym fact-checking evaluation pipeline with Milvus or Tavily retrieval. Use when running fact-checking evaluations, collecting factuality rollouts, serving Nemotron on 32GB GPUs, or configuring ng_run / ng_collect_rollouts.
---

# NeMo Gym fact-checking

Follow the repo README first: `README.md` in the repository root. Use the scripts in `scripts/` instead of ad-hoc `vllm serve` flags.

## Pipeline

```
Prompt → policy vLLM :8000 → checker :8001 ──search_wiki──► embed :8002 ──► Milvus :19530
                                                                      or Tavily
Checker emits [Factual Errors] → verify() → factuality_f1_score
```

- No separate atomization step. The checker chooses search queries.
- One Milvus/Tavily call **per search query**, not per sample.
- Policy and checker are the same 9B weights on **two GPUs**. Do not use `NVIDIA-Nemotron-Nano-8B-v2` (HF 404). Use `nvidia/NVIDIA-Nemotron-Nano-9B-v2`.

## 32GB GPUs (RTX PRO 4500 class)

Weights (~16.6 GiB) fit. Default vLLM 32k ctx + 0.90 util + CUDA graphs **OOM** during Mamba warmup.

Always launch with `scripts/start_vllm_policy_checker.sh` (`8192` / `0.70` / `--enforce-eager` / `VLLM_USE_FLASHINFER_SAMPLER=0`). Empty `curl :8000/v1/models` means the engine is not up; read `vllm_policy.log`.

`--enforce-eager` increases decode latency; that is the cost of fitting 9B hybrid on 32GB.

## Milvus connectivity

`10.185.x.x` ClusterIPs are **not** internet or typical AWS VPC addresses. If `curl -m 5 http://<ip>:19530` times out from the GPU box and from a laptop, the IP is cluster-internal.

Ask for LoadBalancer EXTERNAL-IP, VPN, or:

```bash
ssh -N -L 19530:<CLUSTER-IP>:19530 user@milvus-node
```

Set `milvus_uri: "http://127.0.0.1:19530"`. Do not add `10.185.0.0/16 → igw` on the GPU VPC. Do not open inbound 19530 on the GPU instance SG to “fix” a client timeout (outbound is already all-traffic).

Gym talks to **19530**, not 9091.

This snapshot already supports `milvus_embedding_base_url` so Gym can POST to the local embed server on `:8002` instead of loading SentenceTransformer inside Gym.

## Commands (from repo root)

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

Tavily (no Milvus): `BACKEND=tavily ./scripts/start_gym.sh` with `TAVILY_API_KEY`.

Convert audited JSONL: `python3 scripts/convert_audited_jsonl.py audited.jsonl`.

## Hugging Face

Accept licenses for Nemotron-9B-v2 and `google/embeddinggemma-300m`. Fine-grained tokens need gated-repo access. Prefer `HF_HUB_DISABLE_XET=1` if downloads segfault after 403s.

## Do not

- Serve 32k context on 32GB for this 9B hybrid.
- Treat ClusterIP as a public URL.
- Commit `.hf_token`, `gym/env.yaml`, or rollout outputs with secrets.
- Start a second vLLM if `scripts/start_*.sh` reports already running.
