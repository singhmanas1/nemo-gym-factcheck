# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GenRM Pairwise Comparison Resources Server.

Compares multiple candidate responses using a GenRM model via pairwise comparisons.
The GenRM wrapper injects candidate responses into the model's custom chat template.

Input:
- conversation_history: List of user/assistant messages
- responses: List of N candidate response strings to compare

Output:
- Per-response rewards after pairwise aggregation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status
from fact_checker_prompts import (
    FACT_CHECKING_RM_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
)
from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
)

logger = logging.getLogger(__name__)


class GenRMCompareConfig(BaseResourcesServerConfig):
    """Configuration for the GenRM compare server.

    Attributes:
        genrm_model_server: Target model server (GenRM model with custom chat template)
        genrm_responses_create_params: Base create params for GenRM calls
        comparison_strategy: "all_pairs" or "circular"
        num_judges_per_comparison: Number of judge passes per pair (majority voting)
        aggregator_method: Method for aggregating scores
        reasoning_bonus: Bonus for shortest reasoning content among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for applying bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty
        default_score: Default neutral score when parsing fails
        default_ranking: Default neutral ranking when parsing fails
        debug_logging: Enable verbose logging for debugging
        genrm_parse_retries: Number of retries on parse failures
        genrm_parse_retry_sleep_s: Sleep duration between parse retries
        use_principle: Enable principle-based comparison
        default_principle: Default principle when none provided in request
        reward_combination: How normalized GenRM and factuality are combined
        nonlinear_reward_alpha: GenRM exponent/weight for nonlinear combinations
    """

    name: str = "genrm_compare"
    genrm_model_server: ModelServerRef
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    # Gym scores rollouts through /verify one at a time. Buffer one complete
    # prompt cohort before calculating relative GenRM rewards.
    num_rollouts_per_prompt: int = 1

    # Comparison strategy
    comparison_strategy: str = "circular"  # "all_pairs" or "circular"
    num_judges_per_comparison: int = 1

    # Principle-based GenRM settings
    use_principle: bool = False
    default_principle: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
        "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
        "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
        "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
        "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
        "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
        "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
        "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
        "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
        "identify any missing important information in the assistants' answers that would be beneficial to include "
        "when responding to the user prompt."
    )

    # Aggregator settings (only "simple_tiebreaker" is currently implemented)
    aggregator_method: str = "simple_tiebreaker"

    # Length bonus config (only for simple_tiebreaker)
    reasoning_bonus: float = 0.0
    answer_bonus: float = 0.0
    top_percentile: float = 0.2
    group_reasoning_length_penalty_coeff: float = 0.0
    group_answer_length_penalty_coeff: float = 0.0

    # Default neutral scores when parsing fails
    default_score: float = 3.0
    default_ranking: float = 3.5

    # Debug logging
    debug_logging: bool = False

    # Retry config for parse failures
    genrm_parse_retries: int = 3
    genrm_parse_retry_sleep_s: float = 0.2

    # Optional per-candidate factuality reward.  These are off by default so
    # existing GenRM-only runs preserve their exact reward behavior.  The
    # standalone normalization switch supports checker-free ablations of joint
    # runs, whose GenRM component is normalized before reward combination.
    normalize_genrm_rewards: bool = False
    enable_fact_checker: bool = False
    checker_agent_server: Optional[AgentServerRef] = None
    checker_responses_create_params: Optional[
        NeMoGymResponseCreateParamsNonStreaming
    ] = None
    checker_prompt_template: str = FACT_CHECKING_RM_PROMPT_TEMPLATE
    fact_checker_weight: float = 1.0
    genrm_score_weight: float = 1.0
    reward_combination: Literal[
        "weighted_sum",
        "min",
        "geometric_mean",
        "harmonic_mean",
        "genrm_only",
    ] = "weighted_sum"
    nonlinear_reward_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    fact_checker_prompt_type_metadata_key: Optional[str] = None
    fact_checker_prompt_type: str = "factual"
    artifact_log_path: Optional[str] = None


class GenRMCompareRequest(BaseModel):
    """Request payload for GenRM pairwise comparison."""

    conversation_history: List[Dict[str, str]]  # User/assistant messages before the responses
    response_objs: List[Dict[str, Any]]  # Raw Response API objects from policy model
    principle: Optional[str] = None  # Principle for principle-based GenRM (e.g., "The response should be helpful")


class GenRMCompareVerifyRequest(BaseVerifyRequest):
    """A single Gym rollout, optionally carrying a comparison principle."""

    principle: Optional[str] = None


class GenRMCompareVerifyResponse(BaseVerifyResponse):
    """A Gym rollout reward with cohort-level reward component metrics."""

    genrm_genrm_reward_mean: Optional[float] = None
    genrm_combined_reward_mean: Optional[float] = None
    fact_checker_reward_mean: Optional[float] = None
    genrm_fact_checker_applied: Optional[float] = None
    genrm_fact_checker_failure_count: Optional[float] = None


class GenRMCompareResponse(BaseModel):
    """Response payload with per-response rewards."""

    rewards: List[float]  # One reward per response, in same order as input
    comparison_results: Optional[List[Dict[str, Any]]] = None  # Detailed pairwise results
    metrics: Optional[Dict[str, float]] = None  # Aggregation metrics
    fact_checker_results: Optional[List[Dict[str, Any]]] = None


class GenRMCompareResourcesServer(SimpleResourcesServer):
    """Resources server for GenRM pairwise comparison of multiple responses."""

    config: GenRMCompareConfig
    _artifact_log_file: Any = None
    _artifact_log_lock: Optional[threading.Lock] = None
    _cohort_lock: Optional[asyncio.Lock] = None
    _cohort_buffers: Dict[
        str, List[Tuple[GenRMCompareVerifyRequest, asyncio.Future]]
    ] = defaultdict(list)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {
                str(key): GenRMCompareResourcesServer._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [GenRMCompareResourcesServer._jsonable(item) for item in value]
        return value

    @classmethod
    def _prompt_key(cls, input_messages: Any, principle: Optional[str]) -> str:
        key_data = {
            "input": cls._jsonable(list(input_messages) if input_messages else []),
            "principle": principle,
        }
        serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def _response_with_request_metadata(
        cls, body: GenRMCompareVerifyRequest
    ) -> Dict[str, Any]:
        """Preserve hidden dataset metadata even if vLLM does not echo it."""
        response_obj = cls._jsonable(body.response)
        if not isinstance(response_obj, dict):
            raise TypeError("The policy response must serialize to an object")

        request_metadata = cls._jsonable(
            getattr(body.responses_create_params, "metadata", None)
        )
        if request_metadata:
            if not isinstance(request_metadata, dict):
                raise TypeError("responses_create_params.metadata must be an object")
            response_metadata = response_obj.get("metadata") or {}
            if not isinstance(response_metadata, dict):
                response_metadata = {}
            response_obj["metadata"] = {**response_metadata, **request_metadata}
        return response_obj

    @staticmethod
    def _input_to_conversation_history(input_messages: Any) -> List[Dict[str, str]]:
        conversation_history: List[Dict[str, str]] = []
        for message in list(input_messages) if input_messages else []:
            data = (
                message.model_dump(mode="json")
                if isinstance(message, BaseModel)
                else message
            )
            if not isinstance(data, dict):
                continue
            content = data.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"input_text", "output_text"}
                )
            conversation_history.append(
                {
                    "role": str(data.get("role", "user")),
                    "content": str(content),
                }
            )
        return conversation_history

    async def verify(
        self, body: GenRMCompareVerifyRequest
    ) -> GenRMCompareVerifyResponse:
        """Buffer a prompt cohort, then return its joint GenRM/checker rewards."""
        cohort_size = int(self.config.num_rollouts_per_prompt)
        principle = getattr(body, "principle", None)
        if cohort_size <= 1:
            comparison = await self.compare(
                GenRMCompareRequest(
                    conversation_history=self._input_to_conversation_history(
                        getattr(body.responses_create_params, "input", None)
                    ),
                    response_objs=[self._response_with_request_metadata(body)],
                    principle=principle,
                )
            )
            return self._build_verify_response(
                body, comparison.rewards[0], comparison.metrics
            )

        input_messages = getattr(body.responses_create_params, "input", None) or []
        prompt_key = self._prompt_key(input_messages, principle)
        reward_future = asyncio.get_running_loop().create_future()
        if self._cohort_lock is None:
            self._cohort_lock = asyncio.Lock()

        cohort: Optional[
            List[Tuple[GenRMCompareVerifyRequest, asyncio.Future]]
        ] = None
        async with self._cohort_lock:
            buffer = self._cohort_buffers[prompt_key]
            buffer.append((body, reward_future))
            if len(buffer) > cohort_size:
                self._cohort_buffers.pop(prompt_key, None)
                raise RuntimeError(
                    f"Received more than {cohort_size} rollouts for one prompt cohort"
                )
            if len(buffer) == cohort_size:
                cohort = self._cohort_buffers.pop(prompt_key)

        if cohort is not None:
            try:
                first_body = cohort[0][0]
                comparison = await self.compare(
                    GenRMCompareRequest(
                        conversation_history=self._input_to_conversation_history(
                            getattr(first_body.responses_create_params, "input", None)
                        ),
                        response_objs=[
                            self._response_with_request_metadata(cohort_body)
                            for cohort_body, _ in cohort
                        ],
                        principle=getattr(first_body, "principle", None),
                    )
                )
                if len(comparison.rewards) != len(cohort):
                    raise RuntimeError(
                        "The comparison server returned a reward count that does "
                        "not match the rollout cohort"
                    )
                for (_, future), reward in zip(cohort, comparison.rewards):
                    if not future.done():
                        future.set_result((float(reward), comparison.metrics))
            except Exception as error:
                for _, future in cohort:
                    if not future.done():
                        future.set_exception(error)

        reward, metrics = await asyncio.shield(reward_future)
        return self._build_verify_response(body, reward, metrics)

    @staticmethod
    def _build_verify_response(
        body: GenRMCompareVerifyRequest,
        reward: float,
        metrics: Optional[Dict[str, float]],
    ) -> GenRMCompareVerifyResponse:
        metrics = metrics or {}
        return GenRMCompareVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
            genrm_genrm_reward_mean=metrics.get("genrm_reward_mean"),
            genrm_combined_reward_mean=metrics.get("combined_reward_mean"),
            fact_checker_reward_mean=metrics.get("severity_reward_mean"),
            genrm_fact_checker_applied=metrics.get("fact_checker_applied"),
            genrm_fact_checker_failure_count=metrics.get(
                "fact_checker_failure_count"
            ),
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        self._artifact_log_lock = threading.Lock()
        artifact_log_path = self.config.artifact_log_path or os.environ.get(
            "GENRM_COMPARE_ARTIFACT_LOG_PATH"
        )
        if artifact_log_path:
            artifact_path = os.path.abspath(artifact_log_path)
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            self._artifact_log_file = open(artifact_path, "a", buffering=1)
            logger.info("[GenRM] Fact-check trace logging enabled: %s", artifact_path)
        app.post("/compare")(self.compare)
        return app

    def _log_reward_trace(self, record: Dict[str, Any]) -> None:
        # The environment fallback is intentional. Gym resources servers run in
        # child processes, and older launcher/config plumbing can omit optional
        # resource-server fields even though the parent resolved config has them.
        artifact_log_path = self.config.artifact_log_path or os.environ.get(
            "GENRM_COMPARE_ARTIFACT_LOG_PATH"
        )
        if not artifact_log_path:
            logger.warning("[GenRM] Reward trace logging is disabled")
            return
        artifact_path = os.path.abspath(artifact_log_path)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        if self._artifact_log_lock is None:
            self._artifact_log_lock = threading.Lock()
        with self._artifact_log_lock:
            # Open for each completed comparison so traces survive subprocess
            # teardown and do not depend on a long-lived inherited file handle.
            with open(artifact_path, "a", encoding="utf-8") as artifact_file:
                artifact_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _format_input_conversation(conversation_history: List[Dict[str, str]]) -> str:
        return "".join(
            f"[Begin of {message.get('role', 'user')} Message]\n"
            f"{message.get('content', '')}\n"
            f"[End of {message.get('role', 'user')} Message]\n"
            for message in conversation_history
        ).strip()

    @staticmethod
    def _extract_factual_severity(checker_output: str) -> float:
        matches = re.findall(
            r"\[Beginning of Factual Severity\](.*?)\[End of Factual Severity\]",
            checker_output,
            re.DOTALL,
        )
        # Reasoning models sometimes quote the output template (for example,
        # with a placeholder severity of "X") before emitting their answer.
        # The final valid tagged block is the model's verdict.
        for match in reversed(matches):
            try:
                severity = float(match.strip())
            except (TypeError, ValueError):
                continue
            if 1.0 <= severity <= 5.0:
                return severity
        return -1.0

    @staticmethod
    def _normalized_factuality_reward(checker_result: Dict[str, Any]) -> float:
        """Return higher-is-better factuality f on the complete [0, 1] scale."""
        if "factuality_reward" in checker_result:
            factuality = float(checker_result["factuality_reward"])
        else:
            severity = float(checker_result.get("predicted_severity", -1.0))
            factuality = (5.0 - severity) / 4.0 if 1.0 <= severity <= 5.0 else 0.0
        return min(1.0, max(0.0, factuality))

    async def _run_fact_check(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the tool-using checker once for one candidate response."""
        cfg = self.config
        if cfg.checker_agent_server is None or cfg.checker_responses_create_params is None:
            raise RuntimeError(
                "enable_fact_checker requires checker_agent_server and "
                "checker_responses_create_params."
            )

        response_text = extract_output_text(response_obj)
        checker_prompt = cfg.checker_prompt_template.format(
            input_conversation=self._format_input_conversation(conversation_history),
            response=response_text,
        )
        params = cfg.checker_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=checker_prompt)]
        params.tools = SEARCH_WIKI_TOOL
        params.parallel_tool_calls = True
        params.tool_choice = "auto"
        started = time.monotonic()
        try:
            response = await self.server_client.post(
                server_name=cfg.checker_agent_server.name,
                url_path="/v1/responses",
                json=params,
            )
            status_code = getattr(response, "status", None)
            await raise_for_status(response)
            payload = await response.json()
            parsed = NeMoGymResponse.model_validate(payload)
            checker_output = extract_output_text(parsed.model_dump(mode="json"))
            severity = self._extract_factual_severity(checker_output)
            if severity < 1.0:
                logger.warning(
                    "[GenRM] Fact checker returned no parseable severity: %s",
                    checker_output[-2000:],
                )
            return {
                "checker_prompt": checker_prompt,
                "checker_response_text": checker_output,
                "predicted_severity": severity,
                "severity_reward": (6.0 - severity) / 5.0 if severity >= 1.0 else 0.0,
                # The checker emits error severity (1 is best, 5 is worst).
                # Convert it to a true higher-is-better factuality score whose
                # endpoints span [0, 1].  Equivalently, FactChecker=6-severity
                # and f=(FactChecker-1)/4.
                "factuality_reward": (5.0 - severity) / 4.0
                if severity >= 1.0
                else 0.0,
                "checker_debug": {
                    "status_code": status_code,
                    "raw_response": payload,
                    "checker_wall_ms": (time.monotonic() - started) * 1000,
                },
            }
        except Exception as error:
            logger.exception("[GenRM] Fact-checker request failed")
            return {
                "checker_prompt": checker_prompt,
                "checker_response_text": repr(error),
                "predicted_severity": -1.0,
                "severity_reward": 0.0,
                "factuality_reward": 0.0,
                "checker_debug": {
                    "error_type": type(error).__name__,
                    "error_repr": repr(error),
                    "checker_wall_ms": (time.monotonic() - started) * 1000,
                },
            }

    def _combine_rewards(
        self,
        genrm_rewards: List[float],
        checker_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Normalize component scores and apply the configured aggregation."""
        cfg = self.config
        combined: List[float] = []
        if checker_results is not None and len(checker_results) != len(genrm_rewards):
            raise ValueError("checker_results must align with genrm_rewards")
        aligned_checker_results = checker_results or [None] * len(genrm_rewards)
        for genrm_reward, checker_result in zip(genrm_rewards, aligned_checker_results):
            # Pairwise GenRM scores are nominally in [1, 5]; normalize before
            # combining with the fact-checker reward, which is already [0, 1].
            normalized_genrm = min(1.0, max(0.0, (float(genrm_reward) - 1.0) / 4.0))
            if checker_result is None:
                combined.append(cfg.genrm_score_weight * normalized_genrm)
                continue
            # Keep the established severity_reward scale for existing weighted
            # sum/min experiments.  The new nonlinear modes use factuality f,
            # normalized over the complete [0, 1] range.
            severity_reward = min(
                1.0, max(0.0, float(checker_result["severity_reward"]))
            )
            factuality_reward = self._normalized_factuality_reward(checker_result)
            if cfg.reward_combination == "min":
                combined.append(min(severity_reward, normalized_genrm))
            elif cfg.reward_combination == "geometric_mean":
                aggregate = (
                    normalized_genrm**cfg.nonlinear_reward_alpha
                    * factuality_reward ** (1.0 - cfg.nonlinear_reward_alpha)
                )
                combined.append(aggregate)
            elif cfg.reward_combination == "harmonic_mean":
                if cfg.nonlinear_reward_alpha == 0.0:
                    aggregate = factuality_reward
                elif cfg.nonlinear_reward_alpha == 1.0:
                    aggregate = normalized_genrm
                elif normalized_genrm == 0.0 or factuality_reward == 0.0:
                    aggregate = 0.0
                else:
                    aggregate = 1.0 / (
                        cfg.nonlinear_reward_alpha / normalized_genrm
                        + (1.0 - cfg.nonlinear_reward_alpha) / factuality_reward
                    )
                combined.append(aggregate)
            elif cfg.reward_combination == "genrm_only":
                combined.append(cfg.genrm_score_weight * normalized_genrm)
            else:
                combined.append(
                    cfg.fact_checker_weight * severity_reward
                    + cfg.genrm_score_weight * normalized_genrm
                )
        return combined

    def _prompt_type(self, response_objs: List[Dict[str, Any]]) -> Optional[str]:
        """Read and validate the hidden dataset prompt-type tag."""
        metadata_key = self.config.fact_checker_prompt_type_metadata_key
        if metadata_key is None:
            return None

        prompt_types: set[str] = set()
        for response_obj in response_objs:
            metadata = response_obj.get("metadata") or {}
            prompt_type = metadata.get(metadata_key)
            if not isinstance(prompt_type, str) or not prompt_type:
                raise ValueError(
                    "Fact-checker prompt gating requires response metadata "
                    f"{metadata_key!r}, but it was missing from a policy response."
                )
            prompt_types.add(prompt_type)

        if len(prompt_types) != 1:
            raise ValueError(
                "All responses in one comparison group must have the same prompt type; "
                f"got {sorted(prompt_types)}."
            )
        return next(iter(prompt_types))

    def _should_run_fact_checker(self, prompt_type: Optional[str]) -> bool:
        """Gate fact checking to the configured factual prompt type."""
        cfg = self.config
        if not cfg.enable_fact_checker:
            return False
        if cfg.fact_checker_prompt_type_metadata_key is None:
            return True
        if prompt_type not in {cfg.fact_checker_prompt_type, "general"}:
            raise ValueError(
                "Unknown prompt type for fact-checker gating: "
                f"{prompt_type!r}; expected {cfg.fact_checker_prompt_type!r} or 'general'."
            )
        return prompt_type == cfg.fact_checker_prompt_type

    async def compare(self, body: GenRMCompareRequest) -> GenRMCompareResponse:
        """Compare multiple responses using GenRM pairwise comparisons.

        Args:
            body: Request with conversation_history and response_objs

        Returns:
            GenRMCompareResponse with per-response rewards
        """
        cfg = self.config
        response_objs = body.response_objs
        conversation_history = body.conversation_history
        num_responses = len(response_objs)
        prompt_type = self._prompt_type(response_objs)
        run_fact_checker = self._should_run_fact_checker(prompt_type)

        if cfg.debug_logging:
            logger.info(f"[GenRM] Compare request: {num_responses} responses")

        # Single response case - no comparison is possible. Normalize the
        # neutral GenRM score and add factuality only for tagged prompts.
        if num_responses < 2:
            checker_results = None
            if run_fact_checker:
                checker_results = await asyncio.gather(
                    *(self._run_fact_check(conversation_history, obj) for obj in response_objs)
                )
            if cfg.enable_fact_checker or cfg.normalize_genrm_rewards:
                rewards = self._combine_rewards([cfg.default_score], checker_results)
                return GenRMCompareResponse(
                    rewards=rewards,
                    comparison_results=None,
                    metrics={
                        "genrm_reward_mean": cfg.default_score,
                        "combined_reward_mean": sum(rewards) / len(rewards),
                        "fact_checker_applied": float(run_fact_checker),
                    },
                    fact_checker_results=checker_results,
                )
            return GenRMCompareResponse(
                rewards=[cfg.default_score],
                comparison_results=None,
                metrics=None,
            )

        # Generate comparison pairs
        try:
            comparison_pairs = generate_comparison_pairs(
                cfg.comparison_strategy, num_responses
            )
            if cfg.debug_logging:
                logger.info(f"[GenRM] Strategy '{cfg.comparison_strategy}': {len(comparison_pairs)} pairs")
        except ValueError as e:
            raise ValueError(f"Configuration error: {e}")

        # Build comparison tasks - one task per (pair, judge) combination
        # Multiple judges per pair enables majority voting for more robust scores
        comparison_tasks = []
        comparison_metadata = []

        for judge_idx in range(cfg.num_judges_per_comparison):
            for i, j in comparison_pairs:
                task = self._run_single_comparison(
                    conversation_history,
                    response_objs[i],
                    response_objs[j],
                    pair_idx=(i, j),
                    principle=body.principle,
                )
                comparison_tasks.append(task)
                comparison_metadata.append((i, j, judge_idx))

        # Jiaqi comparison and factuality scoring use separate model servers.
        # Schedule both batches before awaiting either one so factual cohorts do
        # not leave the checker/judge nodes idle while Jiaqi is running (or vice
        # versa).  Calls within each batch are concurrent as well.
        comparison_future = asyncio.gather(*comparison_tasks)
        checker_results: Optional[List[Dict[str, Any]]] = None
        if cfg.enable_fact_checker and run_fact_checker:
            checker_future = asyncio.gather(
                *(self._run_fact_check(conversation_history, obj) for obj in response_objs)
            )
            comparison_results, checker_results = await asyncio.gather(
                comparison_future, checker_future
            )
        else:
            comparison_results = await comparison_future

        # Aggregate pairwise scores into per-response rewards
        rewards, metrics, base_rewards, bonuses = aggregate_scores(
            comparison_results=comparison_results,
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method=cfg.aggregator_method,
            default_score=cfg.default_score,
            reasoning_bonus=cfg.reasoning_bonus,
            answer_bonus=cfg.answer_bonus,
            top_percentile=cfg.top_percentile,
            group_reasoning_length_penalty_coeff=cfg.group_reasoning_length_penalty_coeff,
            group_answer_length_penalty_coeff=cfg.group_answer_length_penalty_coeff,
        )

        # Format detailed results
        detailed_results = [
            {
                "response_i": i,
                "response_j": j,
                "judge_idx": judge_idx,
                "score_1": score_1,
                "score_2": score_2,
                "ranking": ranking,
            }
            for (score_1, score_2, ranking), (i, j, judge_idx) in zip(
                comparison_results, comparison_metadata
            )
        ]

        if cfg.debug_logging:
            logger.info(f"[GenRM] Final rewards: {[f'{r:.4f}' for r in rewards]}")

        final_rewards = rewards
        if cfg.enable_fact_checker or cfg.normalize_genrm_rewards:
            final_rewards = self._combine_rewards(rewards, checker_results)
            metrics = dict(metrics or {})
            metrics["genrm_reward_mean"] = sum(rewards) / len(rewards)
            metrics["combined_reward_mean"] = sum(final_rewards) / len(final_rewards)
            metrics["fact_checker_applied"] = float(run_fact_checker)
            if checker_results is not None:
                metrics["severity_reward_mean"] = sum(
                    result["severity_reward"] for result in checker_results
                ) / len(checker_results)
                metrics["factuality_reward_mean"] = sum(
                    self._normalized_factuality_reward(result)
                    for result in checker_results
                ) / len(checker_results)
                metrics["fact_checker_failure_count"] = float(
                    sum(result["predicted_severity"] < 1.0 for result in checker_results)
                )

        self._log_reward_trace(
            {
                "timestamp": time.time(),
                "conversation_history": conversation_history,
                "prompt_type": prompt_type,
                "genrm_rewards": rewards,
                "normalized_genrm_rewards": [
                    min(1.0, max(0.0, (float(reward) - 1.0) / 4.0))
                    for reward in rewards
                ],
                "factuality_rewards": (
                    [
                        self._normalized_factuality_reward(result)
                        for result in checker_results
                    ]
                    if checker_results is not None
                    else None
                ),
                "fact_checker_results": checker_results,
                "fact_checker_applied": run_fact_checker,
                "reward_combination": cfg.reward_combination,
                "nonlinear_reward_alpha": cfg.nonlinear_reward_alpha,
                "rewards": final_rewards,
            }
        )

        return GenRMCompareResponse(
            rewards=final_rewards,
            comparison_results=detailed_results,
            metrics=metrics,
            fact_checker_results=checker_results,
        )

    async def _run_single_comparison(
        self,
        conversation_history: List[Dict[str, str]],
        response_obj_1: Dict[str, Any],
        response_obj_2: Dict[str, Any],
        pair_idx: Tuple[int, int] = (0, 0),
        principle: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Run a single pairwise comparison via GenRM.

        Args:
            conversation_history: The conversation context
            response_obj_1: First Response API object
            response_obj_2: Second Response API object
            pair_idx: Tuple of (i, j) for logging
            principle: Optional principle for principle-based comparison

        Returns:
            Tuple of (score_1, score_2, ranking)
        """
        cfg = self.config

        # Extract final answer from Response API objects (GenRM only takes the final answer, not reasoning)
        response_1 = extract_output_text(response_obj_1)
        response_2 = extract_output_text(response_obj_2)

        # Keep standard OpenAI roles in input. The genrm_model wrapper consumes
        # this metadata and injects response_1/response_2 into Jiaqi's custom
        # chat template before forwarding the request to vLLM.
        messages: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                type="message",
            )
            for msg in conversation_history
        ]
        metadata = {"response_1": response_1, "response_2": response_2}
        if cfg.use_principle:
            metadata["principle"] = principle if principle else cfg.default_principle

        # Build the request params
        responses_create_params = cfg.genrm_responses_create_params.model_copy(deep=True)
        responses_create_params.input = messages
        responses_create_params.metadata = metadata

        try:
            # Retry logic for parse failures (not connection errors, which are handled elsewhere)
            max_attempts = max(1, int(cfg.genrm_parse_retries) + 1)

            for attempt_idx in range(max_attempts):
                # Call the GenRM model via /v1/responses endpoint
                response = await self.server_client.post(
                    server_name=cfg.genrm_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                raw_response = await response.json()

                # Extract output_text from GenRM response (skip reasoning, only parse the final JSON scores)
                genrm_answer = extract_output_text(raw_response)

                try:
                    score_1, score_2, ranking = parse_genrm_output(
                        genrm_answer,
                        cfg.default_score,
                        cfg.default_ranking,
                        raise_on_fail=True,
                    )
                    return score_1, score_2, ranking

                except GenRMOutputParseError:
                    if attempt_idx < max_attempts - 1:
                        await asyncio.sleep(float(cfg.genrm_parse_retry_sleep_s))
                        continue

                    # Give up: fall back to defaults
                    logger.warning(
                        f"[GenRM] Parse failed for pair {pair_idx} after {max_attempts} attempts; "
                        f"falling back to defaults."
                    )
                    return cfg.default_score, cfg.default_score, cfg.default_ranking

            return cfg.default_score, cfg.default_score, cfg.default_ranking

        except Exception as e:
            logger.error(f"[GenRM] Error in comparison for pair {pair_idx}: {e}")
            return cfg.default_score, cfg.default_score, cfg.default_ranking


if __name__ == "__main__":
    GenRMCompareResourcesServer.run_webserver()
