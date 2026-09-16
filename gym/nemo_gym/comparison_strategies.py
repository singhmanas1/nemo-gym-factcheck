# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comparison strategies for multi-generation reward computation.
"""
import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Protocol, Set, Tuple, runtime_checkable

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from nemo_gym.server_utils import ServerClient, raise_for_status


@runtime_checkable
class ComparisonStrategy(Protocol):
    """Protocol for comparison strategies that compute rewards from multiple generations."""
    
    agent_names: List[str]
    num_generations_per_prompt: int
    policy_model_server_name: str
    
    async def compare(
        self,
        conversation_history: List[Dict[str, str]],
        responses: List[str],
        server_client: ServerClient,
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        """Compare N responses and return (rewards, metrics)."""
        ...


class GenRMStrategyConfig(BaseModel):
    """Configuration for GenRM comparison strategy."""
    agent_names: List[str] = Field(default_factory=lambda: ["genrm_simple_agent"])
    num_generations_per_prompt: int = 16
    genrm_compare_server_name: str = "genrm_compare"
    policy_model_server_name: str = "policy_model"


class GenRMStrategy:
    """GenRM comparison strategy using pairwise comparisons."""
    
    def __init__(self, config: GenRMStrategyConfig):
        self.config = config
        self.agent_names = config.agent_names
        self.num_generations_per_prompt = config.num_generations_per_prompt
        self.policy_model_server_name = config.policy_model_server_name
    
    async def compare(
        self,
        conversation_history: List[Dict[str, str]],
        response_objs: List[Dict],
        server_client: ServerClient,
        principle: Optional[str] = None,
    ) -> Tuple[List[float], Dict[str, float]]:
        """Call genrm_compare server to get rewards for each response.
        
        Args:
            conversation_history: The conversation context
            response_objs: List of raw Response API objects
            server_client: The server client for making requests
            principle: Optional principle for principle-based GenRM comparison
            
        Returns:
            Tuple of (rewards, metrics) from GenRM comparison
        """
        payload = {
            "conversation_history": conversation_history,
            "response_objs": response_objs,
        }
        
        if principle is not None:
            payload["principle"] = principle
        
        res = await server_client.post(
            server_name=self.config.genrm_compare_server_name,
            url_path="/compare",
            json=payload,
        )
        await raise_for_status(res)
        result = await res.json()
        
        rewards = result.get("rewards", [0.0] * len(response_objs))
        metrics = result.get("metrics", {})
        
        return rewards, metrics


def get_prompt_key(example: Dict) -> str:
    """Get stable key for grouping examples by prompt and principle.
    
    Examples with the same conversation history but different principles
    should be in separate groups, so we include principle in the hash.
    """
    if "prompt_id" in example:
        # If prompt_id exists, combine it with principle for uniqueness
        prompt_id = str(example["prompt_id"])
        principle = example.get("principle")
        if principle is not None:
            return f"{prompt_id}::{principle}"
        return prompt_id
    
    # Hash both conversation history and principle together
    conv = extract_conversation_history(example)
    principle = example.get("principle")
    key_data = {
        "conversation": conv,
        "principle": principle,
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def extract_conversation_history(example: Dict) -> List[Dict]:
    """Extract conversation history from example.
    
    Gym examples store history in responses_create_params.input
    """
    responses_create_params = example.get("responses_create_params")
    if responses_create_params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    if "input" not in responses_create_params:
        raise ValueError(f"responses_create_params missing 'input': {list(responses_create_params.keys())}")
    return responses_create_params["input"]


def extract_generated_text(gen_result: Dict) -> str:
    """Extract generated text from generation result."""
    if not isinstance(gen_result, dict):
        raise ValueError(f"Expected dict, got {type(gen_result)}")
    if "output" in gen_result:
        output = gen_result["output"]
        if isinstance(output, list) and output:
            return output[0].get("content", "")
        if isinstance(output, str):
            return output
    if "response" in gen_result:
        return gen_result["response"]
    raise ValueError(f"Cannot extract generated text from: {list(gen_result.keys())}")


async def generate_response(example: Dict, server_client: ServerClient, model_server: str) -> Dict:
    """Generate a single response using the policy model."""
    params = example.get("responses_create_params")
    if params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    res = await server_client.post(server_name=model_server, url_path="/v1/responses", json=params)
    await raise_for_status(res)
    result = await res.json()
    # Preserve hidden dataset metadata (notably prompt_type) even when the
    # policy Responses API implementation does not echo request metadata.
    request_metadata = params.get("metadata") or {}
    if request_metadata:
        response_metadata = result.get("metadata") or {}
        if not isinstance(response_metadata, dict):
            response_metadata = {}
        result["metadata"] = {**response_metadata, **request_metadata}
    return result


def run_examples_with_comparison_strategy(
    *,
    examples: List[Dict],
    rollout_helper: Any,
    head_server_config: Any,
    strategy: ComparisonStrategy,
) -> Iterator[asyncio.Future]:
    """Run grouped policy generations using Gym's comparison strategy.

    Gym 0.2 exposed this through RolloutCollectionHelper.run_examples(). Gym
    0.3 removed that argument, so this compatibility entry point keeps the
    proven grouping implementation in Gym without routing cohorts through
    per-sample /verify calls.
    """
    server_client = rollout_helper.setup_server_client(head_server_config)
    strategy_agent_names = set(strategy.agent_names)
    loop = asyncio.get_running_loop()
    result_futures = [loop.create_future() for _ in examples]

    strategy_samples: List[Tuple[int, Dict]] = []
    standard_samples: List[Tuple[int, Dict]] = []
    for idx, example in enumerate(examples):
        agent_ref = example.get("agent_ref", {})
        agent_name = agent_ref.get("name", "") if isinstance(agent_ref, dict) else ""
        target = strategy_samples if agent_name in strategy_agent_names else standard_samples
        target.append((idx, example))

    async def _set_exception_for_unfinished(error: BaseException) -> None:
        for future in result_futures:
            if not future.done():
                future.set_exception(error)

    async def _run_standard(idx: int, example: Dict) -> None:
        response = await server_client.post(
            server_name=example["agent_ref"]["name"],
            url_path="/run",
            json=example,
        )
        await raise_for_status(response)
        result_futures[idx].set_result(await response.json())

    async def _run_grouped() -> None:
        num_generations = strategy.num_generations_per_prompt
        prompt_buffers: Dict[str, List[Tuple[int, Dict, Dict]]] = defaultdict(list)
        compared: Set[str] = set()
        compare_tasks: List[asyncio.Task] = []
        lock = asyncio.Lock()

        async def _compare_group(group: List[Tuple[int, Dict, Dict]]) -> None:
            first_example = group[0][1]
            rewards, metrics = await strategy.compare(
                extract_conversation_history(first_example),
                [generated for _, _, generated in group],
                server_client,
                principle=first_example.get("principle"),
            )
            if len(rewards) != len(group):
                raise RuntimeError(
                    f"Comparison returned {len(rewards)} rewards for {len(group)} generations"
                )
            component_metrics = {f"genrm_{key}": value for key, value in metrics.items()}
            if "severity_reward_mean" in metrics:
                component_metrics["fact_checker_reward_mean"] = metrics[
                    "severity_reward_mean"
                ]
            for reward, (idx, _, generated) in zip(rewards, group):
                result_futures[idx].set_result(
                    {
                        "response": generated,
                        "reward": reward,
                        **component_metrics,
                    }
                )

        async def _generate(idx: int, example: Dict) -> None:
            generated = await generate_response(
                example, server_client, strategy.policy_model_server_name
            )
            prompt_key = get_prompt_key(example)
            async with lock:
                group = prompt_buffers[prompt_key]
                group.append((idx, example, generated))
                if len(group) == num_generations and prompt_key not in compared:
                    compared.add(prompt_key)
                    compare_tasks.append(asyncio.create_task(_compare_group(list(group))))

        await asyncio.gather(
            *(_generate(idx, example) for idx, example in strategy_samples)
        )
        incomplete = {
            key: len(group)
            for key, group in prompt_buffers.items()
            if len(group) != num_generations
        }
        if incomplete:
            raise RuntimeError(
                f"Expected {num_generations} generations per prompt; incomplete cohorts: {incomplete}"
            )
        if compare_tasks:
            await asyncio.gather(*compare_tasks)

    async def _orchestrate() -> None:
        try:
            await asyncio.gather(
                *(_run_standard(idx, example) for idx, example in standard_samples),
                _run_grouped(),
            )
        except BaseException as error:
            await _set_exception_for_unfinished(error)

    asyncio.create_task(_orchestrate())

    async def _result_at(idx: int) -> Tuple[Dict, Dict]:
        return examples[idx], await result_futures[idx]

    return tqdm.as_completed(
        [asyncio.create_task(_result_at(idx)) for idx in range(len(examples))],
        desc="Collecting rollouts",
        miniters=10,
        total=len(examples),
    )
