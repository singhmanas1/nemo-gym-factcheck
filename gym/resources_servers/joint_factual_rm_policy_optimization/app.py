import glob
import re
import subprocess
import time
import urllib.request
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status
from aiohttp import ClientResponseError
from resources_servers.fact_checker_policy_optimization.app import (
    SearchWikiRequest,
    SearchWikiResponse,
)
from resources_servers.factual_reward_model_with_wiki.app import (
    GenerativeRewardModelWithWikiResourcesServer,
)
from resources_servers.joint_factual_rm_policy_optimization.prompts import (
    JOINT_FACTUAL_RM_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
    WIKI_SUMMARY_PROMPT_TEMPLATE,
)


class JointFactualRMPolicyOptimizationConfig(BaseResourcesServerConfig):
    name: str = "joint_factual_rm_policy_optimization"
    rm_agent_server: AgentServerRef
    rm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    kiwix_zim_dir: Optional[str] = None
    kiwix_port: int = 8082
    kiwix_serve_path: Optional[str] = None
    kiwix_max_chars: int = 8000
    kiwix_top_k: int = 3
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    use_wiki_search_tool: bool = True
    wiki_summary_prompt_template: str = Field(
        default=WIKI_SUMMARY_PROMPT_TEMPLATE,
        description="Prompt template used to summarize Wikipedia evidence before returning it to the RM.",
    )
    prompt_template: str = Field(
        default=JOINT_FACTUAL_RM_PROMPT_TEMPLATE,
        description="Prompt used to ask the trained joint RM to assess a policy response.",
    )
    factuality_weight: float = Field(
        default=0.5,
        description="Weight for the factuality component of the reward (factual=1, hallucinated=0).",
    )
    quality_weight: float = Field(
        default=0.5,
        description="Weight for the quality component of the reward (normalized from 1-5 scale to 0-1).",
    )
    reward_for_parse_failure: float = Field(
        default=0.0,
        description="Reward assigned when the policy response or input conversation cannot be parsed.",
    )


class JointFactualRMPolicyOptimizationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class JointFactualRMPolicyOptimizationVerifyRequest(
    JointFactualRMPolicyOptimizationRunRequest, BaseVerifyRequest
):
    pass


class JointFactualRMPolicyOptimizationVerifyResponse(BaseVerifyResponse):
    rm_response_text: str
    rm_prompt: str
    factuality_prediction: str
    quality_score: Optional[float]
    hallucinated: Optional[bool]
    rm_num_errors: int
    rm_response_debug: dict[str, Any]


class JointFactualRMPolicyOptimizationResourcesServer(SimpleResourcesServer):
    config: JointFactualRMPolicyOptimizationConfig
    _kiwix_process: Optional[subprocess.Popen] = None

    def _kiwix_url(self) -> str:
        return GenerativeRewardModelWithWikiResourcesServer._kiwix_url(self)

    def _start_kiwix(self) -> None:
        return GenerativeRewardModelWithWikiResourcesServer._start_kiwix(self)

    def _stop_kiwix(self) -> None:
        return GenerativeRewardModelWithWikiResourcesServer._stop_kiwix(self)

    def setup_webserver(self) -> FastAPI:
        self._start_kiwix()
        app = super().setup_webserver()
        app.post("/search_wiki")(self.search_wiki)
        return app

    @staticmethod
    def _extract_text_from_response(response: NeMoGymResponse) -> str:
        return GenerativeRewardModelWithWikiResourcesServer._extract_text_from_response(response)

    @staticmethod
    def _format_input_conversation(messages: list[Any]) -> str:
        chunks: list[str] = []
        for message in messages:
            role = (
                message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            )
            content = (
                message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            )
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            chunks.append(f"[Begin of {role} Message]\n{content}\n[End of {role} Message]\n")
        return "".join(chunks).strip()

    @staticmethod
    def _extract_factuality_prediction(rm_response_text: str) -> str:
        block = (
            rm_response_text.split("[Beginning of Factuality Prediction]")[-1]
            .split("[End of Factuality Prediction]")[0]
            .strip()
            .upper()
        )
        if "YES" in block and "NO" not in block:
            return "YES"
        if "NO" in block and "YES" not in block:
            return "NO"
        last_line = (
            rm_response_text.strip().splitlines()[-1].upper()
            if rm_response_text.strip()
            else ""
        )
        if "YES" in last_line and "NO" not in last_line:
            return "YES"
        if "NO" in last_line and "YES" not in last_line:
            return "NO"
        return ""

    @staticmethod
    def _extract_quality_score(rm_response_text: str) -> Optional[float]:
        m = re.search(
            r"\[Beginning of Quality Score\](.*?)\[End of Quality Score\]",
            rm_response_text,
            re.DOTALL,
        )
        if not m:
            return None
        try:
            score = float(m.group(1).strip())
            return score if 1.0 <= score <= 5.0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_error_lines(rm_response_text: str) -> list[str]:
        match = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            rm_response_text,
            flags=re.DOTALL,
        )
        if not match:
            return []
        block = match.group(1).strip()
        return [line.strip() for line in block.splitlines() if line.strip()]

    @staticmethod
    def _build_rm_response_debug(response: Optional[NeMoGymResponse]) -> dict[str, Any]:
        if response is None:
            return {}
        raw_response = response.model_dump(mode="json")
        output = raw_response.get("output", [])
        output_types = [item.get("type") for item in output if isinstance(item, dict)]
        return {
            "id": raw_response.get("id"),
            "model": raw_response.get("model"),
            "status": raw_response.get("status"),
            "output_types": output_types,
            "raw_response": raw_response,
        }

    @staticmethod
    def _build_rm_error_debug(
        error: Exception,
        response_payload: Any = None,
        status_code: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "error_type": type(error).__name__,
            "error_repr": repr(error),
            "status_code": status_code,
            "raw_response": response_payload,
        }

    def _kiwix_search(self, query: str) -> list[dict]:
        return GenerativeRewardModelWithWikiResourcesServer._kiwix_search(self, query)

    def _kiwix_fetch_article(self, path: str) -> str:
        return GenerativeRewardModelWithWikiResourcesServer._kiwix_fetch_article(self, path)

    async def _summarize_wiki_content(self, query: str, wiki_content: str) -> str:
        if not self.config.judge_model_server or not self.config.judge_responses_create_params:
            return wiki_content
        judge_prompt = self.config.wiki_summary_prompt_template.format(
            query=query,
            wiki_content=wiki_content,
        )
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_prompt)]
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        summary = self._extract_text_from_response(judge_response_obj)
        return summary or wiki_content

    async def search_wiki(self, body: SearchWikiRequest) -> SearchWikiResponse:
        try:
            results = self._kiwix_search(body.query)
            if not results:
                return SearchWikiResponse(content=f"No Wikipedia results found for: {body.query}")
            parts = []
            for result in results[: self.config.kiwix_top_k]:
                article = self._kiwix_fetch_article(result["url"])
                parts.append(f"=== Wikipedia: {result['title']} ===\n\n{article}")
            wiki_content = "\n\n".join(parts)
            summarized_content = await self._summarize_wiki_content(
                query=body.query,
                wiki_content=wiki_content,
            )
            return SearchWikiResponse(content=summarized_content)
        except Exception as e:
            return SearchWikiResponse(content=f"Wiki search error: {e}. Query was: {body.query}")

    def _compute_reward(self, factuality_prediction: str, quality_score: Optional[float]) -> float:
        factuality_reward = 1.0 if factuality_prediction == "YES" else 0.0
        quality_reward = (quality_score - 1.0) / 4.0 if quality_score is not None else 0.0
        return (
            self.config.factuality_weight * factuality_reward
            + self.config.quality_weight * quality_reward
        )

    async def verify(
        self, body: JointFactualRMPolicyOptimizationVerifyRequest
    ) -> JointFactualRMPolicyOptimizationVerifyResponse:
        policy_response_text = self._extract_text_from_response(body.response)
        input_conversation = self._format_input_conversation(
            body.responses_create_params.input or []
        )

        if not policy_response_text.strip() or not input_conversation.strip():
            return JointFactualRMPolicyOptimizationVerifyResponse(
                **body.model_dump(),
                reward=self.config.reward_for_parse_failure,
                rm_response_text="Bad input/response",
                factuality_prediction="",
                quality_score=None,
                hallucinated=None,
                rm_num_errors=0,
                rm_prompt="",
                rm_response_debug={},
            )

        rm_prompt = self.config.prompt_template.format(
            input_conversation=input_conversation,
            response=policy_response_text,
        )

        rm_params = self.config.rm_responses_create_params.model_copy(deep=True)
        rm_params.tools = SEARCH_WIKI_TOOL if self.config.use_wiki_search_tool else []
        rm_params.parallel_tool_calls = self.config.use_wiki_search_tool
        rm_params.tool_choice = "auto"
        rm_params.input = [NeMoGymEasyInputMessage(role="user", content=rm_prompt)]

        rm_response_text = ""
        rm_response_debug: dict[str, Any]
        try:
            rm_response_obj = await self.server_client.post(
                server_name=self.config.rm_agent_server.name,
                url_path="/v1/responses",
                json=rm_params,
            )
            status_code = getattr(rm_response_obj, "status", None)
            await raise_for_status(rm_response_obj)
            rm_response_payload = await rm_response_obj.json()
            rm_response = NeMoGymResponse.model_validate(rm_response_payload)
            rm_response_text = self._extract_text_from_response(rm_response)
            rm_response_debug = self._build_rm_response_debug(rm_response)
            rm_response_debug["status_code"] = status_code
        except ClientResponseError as e:
            response_content = getattr(e, "response_content", b"")
            if isinstance(response_content, bytes):
                response_content = response_content.decode(errors="replace")
            rm_response_payload = response_content or locals().get("rm_response_payload")
            status_code = getattr(e, "status", None)
            rm_response_debug = self._build_rm_error_debug(
                e, response_payload=rm_response_payload, status_code=status_code
            )
            rm_response_text = (
                rm_response_payload if isinstance(rm_response_payload, str) else repr(e)
            )
        except Exception as e:
            rm_response_payload = locals().get("rm_response_payload")
            status_code = locals().get("status_code")
            rm_response_debug = self._build_rm_error_debug(
                e, response_payload=rm_response_payload, status_code=status_code
            )
            rm_response_text = (
                rm_response_payload if isinstance(rm_response_payload, str) else repr(e)
            )

        factuality_prediction = self._extract_factuality_prediction(rm_response_text)
        quality_score = self._extract_quality_score(rm_response_text)
        error_lines = self._extract_error_lines(rm_response_text)

        hallucinated = (factuality_prediction == "NO") if factuality_prediction else None
        reward = self._compute_reward(factuality_prediction, quality_score)

        return JointFactualRMPolicyOptimizationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            rm_response_text=rm_response_text,
            factuality_prediction=factuality_prediction,
            quality_score=quality_score,
            hallucinated=hallucinated,
            rm_num_errors=len(error_lines),
            rm_prompt=rm_prompt,
            rm_response_debug=rm_response_debug,
        )


if __name__ == "__main__":
    JointFactualRMPolicyOptimizationResourcesServer.run_webserver()
