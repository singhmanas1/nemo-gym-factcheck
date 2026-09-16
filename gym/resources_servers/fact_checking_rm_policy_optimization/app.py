from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from typing import Any, Literal, Optional

from aiohttp import ClientResponseError
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.config_types import AgentServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status
from resources_servers.fact_checking_reward_model_dev.app import (
    FactCheckingRewardModelDevConfig,
    FactCheckingRewardModelDevResourcesServer,
)
from resources_servers.fact_checking_rm_policy_optimization.prompts import (
    FACT_CHECKING_RM_PROMPT_TEMPLATE,
    POINTWISE_GENRM_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
)
from resources_servers.veriscore_rm_policy_optimization.utils import (
    error_debug,
    extract_text_from_response,
    format_input_conversation,
    response_debug,
)


class FactCheckingRMPolicyOptimizationConfig(FactCheckingRewardModelDevConfig):
    name: str = "fact_checking_rm_policy_optimization"
    checker_agent_server: Optional[AgentServerRef] = None
    checker_responses_create_params: Optional[
        NeMoGymResponseCreateParamsNonStreaming
    ] = None
    genrm_agent_server: AgentServerRef
    genrm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    prompt_template: str = Field(default=FACT_CHECKING_RM_PROMPT_TEMPLATE)
    genrm_prompt_template: str = Field(default=POINTWISE_GENRM_PROMPT_TEMPLATE)
    reward_for_parse_failure: float = Field(default=0.0)
    unlabeled_severity_weight: float = Field(
        default=1.0,
        description=(
            "Weight for the fact checker's predicted-severity reward "
            "(6 - predicted_severity) / 5."
        ),
    )
    genrm_score_weight: float = Field(
        default=1.0,
        description="Weight for the pointwise GenRM quality score normalized to [0, 1].",
    )
    reward_combination: Literal["weighted_sum", "min", "genrm_only"] = Field(
        default="weighted_sum",
        description=(
            "How to combine normalized severity and GenRM rewards. 'weighted_sum' "
            "uses their configured weights; 'min' uses min(severity_reward, "
            "genrm_score_reward) without weights; 'genrm_only' uses only GenRM."
        ),
    )
    missing_genrm_score_reward: float = Field(
        default=0.0,
        description="Normalized GenRM reward used when no valid 1-5 score is parsed.",
    )
    artifact_log_path: Optional[str] = Field(
        default=None,
        description=(
            "If set, append one JSON record per scored rollout containing the raw "
            "fact-checker trace, raw pointwise GenRM output, and reward components."
        ),
    )


class FactCheckingRMPolicyOptimizationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    response_to_check: Optional[str] = Field(
        default=None,
        description=(
            "Optional fixed response for eval. When set, this is scored instead of the "
            "policy rollout response."
        ),
    )


class FactCheckingRMPolicyOptimizationVerifyRequest(
    FactCheckingRMPolicyOptimizationRunRequest, BaseVerifyRequest
):
    pass


class FactCheckingRMPolicyOptimizationVerifyResponse(BaseVerifyResponse):
    checker_prompt: str
    checker_response_text: str
    predicted_severity: float
    severity_reward: float
    num_errors: int
    checker_response_debug: dict[str, Any]
    genrm_prompt: str
    genrm_response_text: str
    genrm_quality_score: Optional[float]
    genrm_score_reward: float
    genrm_response_debug: dict[str, Any]


class FactCheckingRMPolicyOptimizationResourcesServer(FactCheckingRewardModelDevResourcesServer):
    config: FactCheckingRMPolicyOptimizationConfig
    _artifact_log_file: Any = None
    _artifact_log_lock: Optional[threading.Lock] = None

    def setup_webserver(self):
        app = super().setup_webserver()
        self._artifact_log_lock = threading.Lock()
        if self.config.artifact_log_path:
            artifact_path = os.path.abspath(self.config.artifact_log_path)
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            self._artifact_log_file = open(artifact_path, "a", buffering=1)
            print(f"[joint-rm] Reward trace logging enabled -> {artifact_path}", flush=True)
        return app

    def _log_reward_trace(self, record: dict[str, Any]) -> None:
        if self._artifact_log_file is None or self._artifact_log_lock is None:
            return
        with self._artifact_log_lock:
            self._artifact_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_predicted_errors_block(checker_output: str) -> Optional[str]:
        match = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            checker_output,
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_genrm_quality_score(genrm_output: str) -> Optional[float]:
        match = re.search(
            r"\[Beginning of Quality Score\](.*?)\[End of Quality Score\]",
            genrm_output,
            re.DOTALL,
        )
        if not match:
            return None
        try:
            score = float(match.group(1).strip())
        except (TypeError, ValueError):
            return None
        return score if 1.0 <= score <= 5.0 else None

    def _genrm_score_reward(self, quality_score: Optional[float]) -> float:
        if quality_score is None:
            return self.config.missing_genrm_score_reward
        return (quality_score - 1.0) / 4.0

    async def _call_agent(
        self,
        *,
        agent_server: AgentServerRef,
        request_params: NeMoGymResponseCreateParamsNonStreaming,
        prompt: str,
        debug_prefix: str,
        use_search_tool: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        start = time.monotonic()
        params = request_params.model_copy(deep=True)
        if use_search_tool:
            params.tools = SEARCH_WIKI_TOOL
            params.parallel_tool_calls = True
            params.tool_choice = "auto"
        else:
            params.tools = []
            params.parallel_tool_calls = False
            params.tool_choice = "none"
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        try:
            response_obj = await self.server_client.post(
                server_name=agent_server.name,
                url_path="/v1/responses",
                json=params,
            )
            status_code = getattr(response_obj, "status", None)
            await raise_for_status(response_obj)
            payload = await response_obj.json()
            response = NeMoGymResponse.model_validate(payload)
            debug = response_debug(response)
            debug["status_code"] = status_code
            debug[f"{debug_prefix}_wall_ms"] = (time.monotonic() - start) * 1000
            return self._extract_text_from_response(response), debug
        except ClientResponseError as error:
            response_content = getattr(error, "response_content", b"")
            if isinstance(response_content, bytes):
                response_content = response_content.decode(errors="replace")
            debug = error_debug(
                error,
                response_payload=response_content or locals().get("payload"),
                status_code=getattr(error, "status", None),
            )
        except Exception as error:
            debug = error_debug(
                error,
                response_payload=locals().get("payload"),
                status_code=locals().get("status_code"),
            )
        debug[f"{debug_prefix}_wall_ms"] = (time.monotonic() - start) * 1000
        raw = debug.get("raw_response")
        return raw if isinstance(raw, str) else repr(debug), debug

    async def _call_checker_rm(self, checker_prompt: str) -> tuple[str, dict[str, Any]]:
        if (
            self.config.checker_agent_server is None
            or self.config.checker_responses_create_params is None
        ):
            raise RuntimeError(
                "The checker agent is required unless reward_combination='genrm_only'."
            )
        return await self._call_agent(
            agent_server=self.config.checker_agent_server,
            request_params=self.config.checker_responses_create_params,
            prompt=checker_prompt,
            debug_prefix="checker",
            use_search_tool=True,
        )

    async def _call_genrm(self, genrm_prompt: str) -> tuple[str, dict[str, Any]]:
        return await self._call_agent(
            agent_server=self.config.genrm_agent_server,
            request_params=self.config.genrm_responses_create_params,
            prompt=genrm_prompt,
            debug_prefix="genrm",
        )

    async def _score_checker_output(
        self,
        checker_output: str,
    ) -> tuple[float, float, int]:
        predicted_block = self._extract_predicted_errors_block(checker_output)
        predicted_severity = self._extract_factual_severity(checker_output)
        num_errors = 0
        severity_reward = 0.0

        if predicted_block is None:
            return predicted_severity, severity_reward, num_errors

        num_errors = len([line for line in predicted_block.splitlines() if line.strip()])
        if predicted_severity >= 1.0:
            severity_reward = max(0.0, (6.0 - predicted_severity) / 5.0)

        return predicted_severity, severity_reward, num_errors

    def _compute_reward(
        self,
        severity_reward: float,
        genrm_score_reward: float,
    ) -> float:
        if self.config.reward_combination == "genrm_only":
            return self.config.genrm_score_weight * genrm_score_reward
        if self.config.reward_combination == "min":
            return min(severity_reward, genrm_score_reward)

        fact_checker_reward = self.config.unlabeled_severity_weight * severity_reward
        return fact_checker_reward + self.config.genrm_score_weight * genrm_score_reward

    async def verify(
        self, body: FactCheckingRMPolicyOptimizationVerifyRequest
    ) -> FactCheckingRMPolicyOptimizationVerifyResponse:
        policy_response_text = (
            body.response_to_check.strip()
            if isinstance(body.response_to_check, str) and body.response_to_check.strip()
            else extract_text_from_response(body.response)
        )
        input_conversation = format_input_conversation(body.responses_create_params.input or [])

        if not policy_response_text.strip() or not input_conversation.strip():
            return FactCheckingRMPolicyOptimizationVerifyResponse(
                **body.model_dump(),
                reward=self.config.reward_for_parse_failure,
                checker_prompt="",
                checker_response_text="Bad input/response",
                predicted_severity=-1.0,
                severity_reward=0.0,
                num_errors=0,
                checker_response_debug={},
                genrm_prompt="",
                genrm_response_text="Bad input/response",
                genrm_quality_score=None,
                genrm_score_reward=self.config.missing_genrm_score_reward,
                genrm_response_debug={},
            )

        checker_prompt = self.config.prompt_template.format(
            input_conversation=input_conversation,
            response=policy_response_text,
        )
        genrm_prompt = self.config.genrm_prompt_template.format(
            input_conversation=input_conversation,
            response=policy_response_text,
        )
        if self.config.reward_combination == "genrm_only":
            checker_output, checker_debug = "Disabled (GenRM-only reward)", {}
            predicted_severity, severity_reward, num_errors = -1.0, 0.0, 0
            genrm_output, genrm_debug = await self._call_genrm(genrm_prompt)
        else:
            (checker_output, checker_debug), (genrm_output, genrm_debug) = await asyncio.gather(
                self._call_checker_rm(checker_prompt), self._call_genrm(genrm_prompt)
            )
            predicted_severity, severity_reward, num_errors = await self._score_checker_output(
                checker_output=checker_output
            )
        genrm_quality_score = self._extract_genrm_quality_score(genrm_output)
        genrm_score_reward = self._genrm_score_reward(genrm_quality_score)
        reward = self._compute_reward(
            severity_reward,
            genrm_score_reward,
        )

        self._log_reward_trace(
            {
                "timestamp": time.time(),
                "policy_response_text": policy_response_text,
                "input_conversation": input_conversation,
                "checker_prompt": checker_prompt,
                "checker_response_text": checker_output,
                "checker_response_debug": checker_debug,
                "predicted_severity": predicted_severity,
                "severity_reward": severity_reward,
                "num_errors": num_errors,
                "genrm_prompt": genrm_prompt,
                "genrm_response_text": genrm_output,
                "genrm_response_debug": genrm_debug,
                "genrm_quality_score": genrm_quality_score,
                "genrm_score_reward": genrm_score_reward,
                "reward_combination": self.config.reward_combination,
                "reward": float(reward),
            }
        )

        return FactCheckingRMPolicyOptimizationVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            checker_prompt=checker_prompt,
            checker_response_text=checker_output,
            predicted_severity=predicted_severity,
            severity_reward=severity_reward,
            num_errors=num_errors,
            checker_response_debug=checker_debug,
            genrm_prompt=genrm_prompt,
            genrm_response_text=genrm_output,
            genrm_quality_score=genrm_quality_score,
            genrm_score_reward=genrm_score_reward,
            genrm_response_debug=genrm_debug,
        )


if __name__ == "__main__":
    FactCheckingRMPolicyOptimizationResourcesServer.run_webserver()
