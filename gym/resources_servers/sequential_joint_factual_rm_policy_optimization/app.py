import re
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
from resources_servers.joint_factual_rm_policy_optimization.app import (
    JointFactualRMPolicyOptimizationResourcesServer,
    JointFactualRMPolicyOptimizationVerifyResponse,
)
from resources_servers.joint_factual_rm_policy_optimization.prompts import (
    JOINT_FACTUAL_RM_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
    WIKI_SUMMARY_PROMPT_TEMPLATE,
)

STEP2_QUALITY_PROMPT_TEMPLATE = """\
{rm_prompt}

[Factuality Assessment]
We have determined that the response above is {factuality_verdict}.
{errors_section}
Given this factuality assessment, please now provide only the quality score.

Your output must follow this exact format:

[Beginning of Quality Score]
X
[End of Quality Score]\
"""

class SequentialJointFactualRMPolicyOptimizationConfig(BaseResourcesServerConfig):
    name: str = "sequential_joint_factual_rm_policy_optimization"
    # Step 1: RM agent (handles wiki tool calls + factuality generation)
    rm_agent_server: AgentServerRef
    rm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Step 2: direct model call (no tools, fresh user prompt with factuality verdict)
    rm_model_server: ModelServerRef
    rm_step2_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    step2_prompt_template: str = Field(default=STEP2_QUALITY_PROMPT_TEMPLATE)
    kiwix_zim_dir: Optional[str] = None
    kiwix_port: int = 8082
    kiwix_serve_path: Optional[str] = None
    kiwix_max_chars: int = 8000
    kiwix_top_k: int = 3
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    use_wiki_search_tool: bool = True
    wiki_summary_prompt_template: str = Field(default=WIKI_SUMMARY_PROMPT_TEMPLATE)
    prompt_template: str = Field(default=JOINT_FACTUAL_RM_PROMPT_TEMPLATE)
    factuality_weight: float = Field(default=0.5)
    quality_weight: float = Field(default=0.5)
    reward_for_parse_failure: float = Field(default=0.0)


class SequentialJointFactualRMPolicyOptimizationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SequentialJointFactualRMPolicyOptimizationVerifyRequest(
    SequentialJointFactualRMPolicyOptimizationRunRequest, BaseVerifyRequest
):
    pass


class SequentialJointFactualRMPolicyOptimizationResourcesServer(SimpleResourcesServer):
    config: SequentialJointFactualRMPolicyOptimizationConfig
    _kiwix_process = None

    # Reuse kiwix and wiki helpers from the joint RM server
    _kiwix_url = JointFactualRMPolicyOptimizationResourcesServer._kiwix_url
    _start_kiwix = JointFactualRMPolicyOptimizationResourcesServer._start_kiwix
    _stop_kiwix = JointFactualRMPolicyOptimizationResourcesServer._stop_kiwix
    _kiwix_search = JointFactualRMPolicyOptimizationResourcesServer._kiwix_search
    _kiwix_fetch_article = JointFactualRMPolicyOptimizationResourcesServer._kiwix_fetch_article
    _extract_text_from_response = JointFactualRMPolicyOptimizationResourcesServer._extract_text_from_response
    _format_input_conversation = JointFactualRMPolicyOptimizationResourcesServer._format_input_conversation
    _extract_factuality_prediction = JointFactualRMPolicyOptimizationResourcesServer._extract_factuality_prediction
    _extract_quality_score = JointFactualRMPolicyOptimizationResourcesServer._extract_quality_score
    _extract_error_lines = JointFactualRMPolicyOptimizationResourcesServer._extract_error_lines
    _build_rm_response_debug = JointFactualRMPolicyOptimizationResourcesServer._build_rm_response_debug
    _build_rm_error_debug = JointFactualRMPolicyOptimizationResourcesServer._build_rm_error_debug
    _summarize_wiki_content = JointFactualRMPolicyOptimizationResourcesServer._summarize_wiki_content
    _compute_reward = JointFactualRMPolicyOptimizationResourcesServer._compute_reward

    def setup_webserver(self) -> FastAPI:
        self._start_kiwix()
        app = super().setup_webserver()
        app.post("/search_wiki")(self.search_wiki)
        return app

    async def search_wiki(self, body: SearchWikiRequest) -> SearchWikiResponse:
        return await JointFactualRMPolicyOptimizationResourcesServer.search_wiki(self, body)

    async def verify(
        self, body: SequentialJointFactualRMPolicyOptimizationVerifyRequest
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

        # ── Step 1: factuality (RM agent with wiki tools) ──────────────────
        rm_params_step1 = self.config.rm_responses_create_params.model_copy(deep=True)
        rm_params_step1.tools = SEARCH_WIKI_TOOL if self.config.use_wiki_search_tool else []
        rm_params_step1.parallel_tool_calls = self.config.use_wiki_search_tool
        rm_params_step1.tool_choice = "auto"
        rm_params_step1.input = [NeMoGymEasyInputMessage(role="user", content=rm_prompt)]

        step1_text = ""
        rm_response_debug: dict[str, Any] = {}
        try:
            resp1 = await self.server_client.post(
                server_name=self.config.rm_agent_server.name,
                url_path="/v1/responses",
                json=rm_params_step1,
            )
            status_code = getattr(resp1, "status", None)
            await raise_for_status(resp1)
            resp1_payload = await resp1.json()
            resp1_obj = NeMoGymResponse.model_validate(resp1_payload)
            step1_text = self._extract_text_from_response(resp1_obj)
            rm_response_debug = self._build_rm_response_debug(resp1_obj)
            rm_response_debug["status_code"] = status_code
        except ClientResponseError as e:
            content = getattr(e, "response_content", b"")
            if isinstance(content, bytes):
                content = content.decode(errors="replace")
            rm_response_debug = self._build_rm_error_debug(e, response_payload=content, status_code=getattr(e, "status", None))
            step1_text = content if isinstance(content, str) else repr(e)
        except Exception as e:
            rm_response_debug = self._build_rm_error_debug(e)
            step1_text = repr(e)

        factuality_prediction = self._extract_factuality_prediction(step1_text)
        step1_error_lines = self._extract_error_lines(step1_text)

        # ── Step 2: quality (fresh user prompt containing the factuality verdict) ─
        factuality_verdict = (
            "factually correct" if factuality_prediction == "YES"
            else "not factually correct" if factuality_prediction == "NO"
            else "of unknown factuality"
        )
        if step1_error_lines:
            errors_section = "The following factual errors were identified:\n" + "\n".join(
                f"- {e}" for e in step1_error_lines
            )
        else:
            errors_section = "No factual errors were identified."
        step2_prompt = self.config.step2_prompt_template.format(
            rm_prompt=rm_prompt,
            factuality_verdict=factuality_verdict,
            errors_section=errors_section,
        )
        step2_text = ""
        try:
            rm_params_step2 = self.config.rm_step2_responses_create_params.model_copy(deep=True)
            rm_params_step2.tools = []
            rm_params_step2.tool_choice = "none"
            rm_params_step2.input = [
                NeMoGymEasyInputMessage(role="user", content=step2_prompt),
            ]
            resp2 = await self.server_client.post(
                server_name=self.config.rm_model_server.name,
                url_path="/v1/responses",
                json=rm_params_step2,
            )
            await raise_for_status(resp2)
            resp2_obj = NeMoGymResponse.model_validate(await resp2.json())
            step2_text = self._extract_text_from_response(resp2_obj)
        except Exception as e:
            step2_text = repr(e)

        # Combine for quality/error extraction
        full_rm_text = step1_text + "\n" + step2_text
        quality_score = self._extract_quality_score(full_rm_text)
        error_lines = self._extract_error_lines(full_rm_text)
        hallucinated = (factuality_prediction == "NO") if factuality_prediction else None
        reward = self._compute_reward(factuality_prediction, quality_score)

        return JointFactualRMPolicyOptimizationVerifyResponse(
            **body.model_dump(),
            reward=reward,
            rm_response_text=full_rm_text,
            factuality_prediction=factuality_prediction,
            quality_score=quality_score,
            hallucinated=hallucinated,
            rm_num_errors=len(error_lines),
            rm_prompt=rm_prompt,
            rm_response_debug=rm_response_debug,
        )


if __name__ == "__main__":
    SequentialJointFactualRMPolicyOptimizationResourcesServer.run_webserver()
