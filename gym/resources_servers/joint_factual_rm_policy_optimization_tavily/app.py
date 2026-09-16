import asyncio
import json
import os
import re
import threading
import time
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
from resources_servers.joint_factual_rm_policy_optimization_tavily.prompts import (
    JOINT_FACTUAL_RM_PROMPT_TEMPLATE,
    SEARCH_WEB_TOOL,
)

_MAX_CONTENT_CHARS = 1000  # max chars per Tavily result snippet


class SearchWikiRequest(BaseModel):
    query: str


class SearchWikiResponse(BaseModel):
    content: str


class JointFactualRMPolicyOptimizationTavilyConfig(BaseResourcesServerConfig):
    name: str = "joint_factual_rm_policy_optimization_tavily"
    rm_agent_server: AgentServerRef
    rm_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    tavily_api_key: str
    tavily_max_results: int = 5
    tavily_cache_file: str = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/abukharin/fact-rm/fact-checker/evals/factscore/.cache/factscore/tavily_cache_global.json"
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


class JointFactualRMPolicyOptimizationTavilyRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class JointFactualRMPolicyOptimizationTavilyVerifyRequest(
    JointFactualRMPolicyOptimizationTavilyRunRequest, BaseVerifyRequest
):
    pass


class JointFactualRMPolicyOptimizationTavilyVerifyResponse(BaseVerifyResponse):
    rm_response_text: str
    rm_prompt: str
    factuality_prediction: str
    quality_score: Optional[float]
    hallucinated: Optional[bool]
    rm_num_errors: int
    rm_response_debug: dict[str, Any]


class JointFactualRMPolicyOptimizationTavilyResourcesServer(SimpleResourcesServer):
    config: JointFactualRMPolicyOptimizationTavilyConfig
    _tavily_cache: dict = {}
    _tavily_cache_lock: threading.Lock = None
    _tavily_semaphore: asyncio.Semaphore = None

    def setup_webserver(self) -> FastAPI:
        self._tavily_cache_lock = threading.Lock()
        self._tavily_cache = self._load_tavily_cache()
        self._tavily_semaphore = asyncio.Semaphore(20)
        app = super().setup_webserver()
        app.post("/search_wiki")(self.search_wiki)
        return app

    def _load_tavily_cache(self) -> dict:
        if os.path.exists(self.config.tavily_cache_file):
            try:
                with open(self.config.tavily_cache_file) as f:
                    cache = json.load(f)
                print(f"Loaded {len(cache)} Tavily cache entries from {self.config.tavily_cache_file}", flush=True)
                return cache
            except Exception as e:
                print(f"Warning: could not load Tavily cache: {e}", flush=True)
        return {}

    def _save_tavily_cache(self) -> None:
        import tempfile
        cache_file = self.config.tavily_cache_file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with self._tavily_cache_lock:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_file))
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(self._tavily_cache, f)
                os.replace(tmp_path, cache_file)
            except Exception:
                os.unlink(tmp_path)
                raise

    def _tavily_search(self, query: str) -> list[dict]:
        """Search Tavily, return [{title, url, text}, ...] with cache and retry on rate limits."""
        with self._tavily_cache_lock:
            if query in self._tavily_cache:
                return self._tavily_cache[query]

        from tavily import TavilyClient
        client = TavilyClient(api_key=self.config.tavily_api_key)
        max_attempts = 4
        wait = 1.0
        for attempt in range(max_attempts):
            try:
                response = client.search(
                    query=query,
                    max_results=self.config.tavily_max_results,
                    search_depth="basic",
                )
                results = [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "text": r.get("content", "") or ""}
                    for r in response.get("results", [])
                ]
                with self._tavily_cache_lock:
                    self._tavily_cache[query] = results
                self._save_tavily_cache()
                return results
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "rate" in err_str or "429" in err_str or "too many" in err_str
                if is_rate_limit and attempt < max_attempts - 1:
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise
        return []

    async def search_wiki(self, body: SearchWikiRequest) -> SearchWikiResponse:
        """Search the web via Tavily, return concatenated snippets (with cache)."""
        try:
            async with self._tavily_semaphore:
                results = await asyncio.to_thread(self._tavily_search, body.query)
            if not results:
                return SearchWikiResponse(content=f"No web results found for: {body.query}")
            parts = []
            for r in results:
                text = r.get("text") or r.get("content", "")
                if len(text) > _MAX_CONTENT_CHARS:
                    text = text[:_MAX_CONTENT_CHARS] + "..."
                parts.append(f"[{r['title']}]\n{text}")
            return SearchWikiResponse(content="\n\n".join(parts))
        except Exception as e:
            return SearchWikiResponse(content=f"Web search error: {e}. Query was: {body.query}")

    @staticmethod
    def _extract_text_from_response(response: NeMoGymResponse) -> str:
        for output in reversed(response.output):
            if getattr(output, "type", None) == "message" and getattr(output, "role", None) == "assistant":
                content = getattr(output, "content", None)
                texts: list[str] = []
                if isinstance(content, list):
                    for c in content:
                        text = getattr(c, "text", None)
                        if isinstance(text, str):
                            texts.append(text)
                elif isinstance(content, str):
                    texts = [content]
                if texts:
                    full_text = "\n".join(texts).strip()
                    return full_text.split("</think>")[-1].strip()
        return ""

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

    def _compute_reward(self, factuality_prediction: str, quality_score: Optional[float]) -> float:
        factuality_reward = 1.0 if factuality_prediction == "YES" else 0.0
        quality_reward = (quality_score - 1.0) / 4.0 if quality_score is not None else 0.0
        return (
            self.config.factuality_weight * factuality_reward
            + self.config.quality_weight * quality_reward
        )

    async def verify(
        self, body: JointFactualRMPolicyOptimizationTavilyVerifyRequest
    ) -> JointFactualRMPolicyOptimizationTavilyVerifyResponse:
        policy_response_text = self._extract_text_from_response(body.response)
        input_conversation = self._format_input_conversation(
            body.responses_create_params.input or []
        )

        if not policy_response_text.strip() or not input_conversation.strip():
            return JointFactualRMPolicyOptimizationTavilyVerifyResponse(
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
        rm_params.tools = SEARCH_WEB_TOOL
        rm_params.parallel_tool_calls = True
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

        return JointFactualRMPolicyOptimizationTavilyVerifyResponse(
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
    JointFactualRMPolicyOptimizationTavilyResourcesServer.run_webserver()
