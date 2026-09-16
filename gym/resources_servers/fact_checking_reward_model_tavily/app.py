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
import asyncio
import json
import os
import re
import threading
import time
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

_MAX_CONTENT_CHARS = 1000  # max chars per Tavily result snippet


class FactCheckingRewardModelTavilyConfig(BaseResourcesServerConfig):
    tavily_api_key: str
    tavily_max_results: int = 5
    tavily_cache_file: str = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/abukharin/fact-rm/fact-checker/evals/factscore/.cache/factscore/tavily_cache_global.json"
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    rubric_parallel_evaluation: bool = True
    rubric_yes_label: str = "[[YES]]"
    rubric_no_label: str = "[[NO]]"
    factuality_weight: float = Field(
        default=1.0,
        description="Multiplicative weight applied to the factuality component of the reward.",
    )
    quality_weight: float = Field(
        default=1.0,
        description="Multiplicative weight applied to the quality component of the reward.",
    )

    error_judging_judge_prompt_template: str = Field(
    default="""You are evaluating whether a model accurately identifies factual errors in a conversation.

MODEL'S IDENTIFIED FACTUAL ERRORS:
{predicted_errors}

GROUND-TRUTH FACTUAL ERROR:
{expected_error}

Does the model's identify the factual error correctly in its factual errors?
The identified factual error must be almost exactly the same as the ground truth error, with only small paraphrases allowed. Note that broad statements such as "Incorrect or misleading facts" should not be counted as matching the ground truth error. The level of precision in the predicted error must be the same as the level of precision in the ground truth error.
Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line.""",
        description="Template for the judge evaluation prompt"
    )

    num_errors_judge_prompt_template: str = Field(
        default="""You are counting how many factual errors attempts to identify in a conversation. The ground-truth factual errors are provided, and should be used as a reference for what constitutes an identified factual error. Note you are not evaluating correctness, but rather the number of errors identified. If a model idenitifies an error that is not in the ground-truth factual errors, still count it as an error.

If the model puts multiple errors in a single line, count each error separately. Use the ground-truth factual errors as a reference for what constitutes an identified factual error.

MODEL'S IDENTIFIED FACTUAL ERRORS:
{predicted_errors}

GROUND-TRUTH FACTUAL ERRORS:
{expected_errors}

How many factual errors does the model identify?
Analyze carefully, then respond with the number of errors identified in between <num_errors> tags (e.g. <num_errors>3</num_errors>).""",
        description="Template for the judge evaluation prompt"
    )


class RubricEvaluation(BaseModel):
    expected_error: str
    judge_prompt: str
    judge_response: str
    verdict: str
    score: float


class SearchWikiRequest(BaseModel):
    query: str


class SearchWikiResponse(BaseModel):
    content: str


class FactCheckingRewardModelRunRequest(BaseRunRequest):
    id: Union[int, str]
    expected_errors: List[str]
    is_factual: bool
    loss_type: str
    ground_truth_quality: int


class FactCheckingRewardModelVerifyRequest(FactCheckingRewardModelRunRequest, BaseVerifyRequest):
    pass


class FactCheckingRewardModelVerifyResponse(BaseVerifyResponse):
    reward: float
    factuality_accuracy: float
    factuality_f1_score: float
    num_errors: int
    quality_score: Optional[float]
    quality_reward: float


class FactCheckingRewardModelTavilyResourcesServer(SimpleResourcesServer):
    config: FactCheckingRewardModelTavilyConfig
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
        # Check cache first (shared with factscorer — keyed by query, values have "text" field)
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
                # Cache only on success; do NOT cache failures
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
    def _extract_verdict(response_text: str, yes_label: str, no_label: str) -> str:
        yes_pos = response_text.rfind(yes_label)
        no_pos = response_text.rfind(no_label)
        if yes_pos < 0 and no_pos < 0:
            last_line = response_text.strip().split("\n")[-1].upper() if response_text.strip() else ""
            if "YES" in last_line:
                return "YES"
            return "NO"
        return "YES" if yes_pos > no_pos else "NO"

    @staticmethod
    def _aggregate_scores(scores: list[float], num_predicted_errors: int, num_ground_truth_errors: int) -> float:
        num_predicted_errors = max(0, int(num_predicted_errors))
        num_ground_truth_errors = max(0, int(num_ground_truth_errors))

        if num_predicted_errors == 0 and num_ground_truth_errors == 0:
            return 1.0

        if not scores or num_predicted_errors == 0 or num_ground_truth_errors == 0:
            return 0.0

        tp = max(0.0, float(sum(scores)))
        tp = min(tp, float(num_predicted_errors), float(num_ground_truth_errors))

        precision = tp / num_predicted_errors if num_predicted_errors > 0 else 0.0
        recall = tp / num_ground_truth_errors if num_ground_truth_errors > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _extract_quality_score(output: str) -> Optional[float]:
        m = re.search(
            r"\[Beginning of Quality Score\](.*?)\[End of Quality Score\]",
            output,
            re.DOTALL,
        )
        if not m:
            return None
        try:
            score = float(m.group(1).strip())
            return score if 1.0 <= score <= 5.0 else None
        except (ValueError, TypeError):
            return None

    async def search_wiki(self, body: SearchWikiRequest) -> SearchWikiResponse:
        """Search the web via Tavily (exposed as search_wiki for model compatibility), return concatenated snippets (with cache)."""
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

    async def _evaluate_single_error(
        self, expected_error: str, predicted_factual_errors: str
    ) -> RubricEvaluation:
        judge_prompt = self.config.error_judging_judge_prompt_template.format(expected_error=expected_error, predicted_errors=predicted_factual_errors)
        msgs: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt)
        ]
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = msgs
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        judge_response = self._extract_text_from_response(judge_response_obj)
        verdict = self._extract_verdict(judge_response, self.config.rubric_yes_label, self.config.rubric_no_label)
        score = 1.0 if verdict == "YES" else 0.0
        return RubricEvaluation(expected_error=expected_error, judge_prompt=judge_prompt, judge_response=judge_response, verdict=verdict, score=score)

    async def _evaluate_num_errors(
        self, expected_errors: str, predicted_factual_errors: str
    ) -> int:
        judge_prompt = self.config.num_errors_judge_prompt_template.format(expected_errors=expected_errors, predicted_errors=predicted_factual_errors)
        msgs: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt)
        ]
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = msgs
        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response_obj = NeMoGymResponse.model_validate(await response_obj.json())
        judge_response = self._extract_text_from_response(judge_response_obj)
        try:
            num_errors = int(judge_response.split("<num_errors>")[-1].split("</num_errors>")[0].strip())
        except Exception:
            num_errors = 20
        return num_errors

    async def verify(
        self, body: FactCheckingRewardModelVerifyRequest
    ) -> FactCheckingRewardModelVerifyResponse:
        output = self._extract_text_from_response(body.response)

        expected_errors = [x for x in body.expected_errors if x.strip()]
        is_factual = body.is_factual
        ground_truth_quality = body.ground_truth_quality

        factuality_prediction = output.split("[Beginning of Factuality Prediction]")[-1].split("[End of Factuality Prediction]")[0].strip()
        if factuality_prediction == "YES" and is_factual:
            classification_accuracy = 1.0
        elif factuality_prediction == "NO" and not is_factual:
            classification_accuracy = 1.0
        elif factuality_prediction == "NO" and is_factual:
            classification_accuracy = 0.0
        elif factuality_prediction == "YES" and not is_factual:
            classification_accuracy = 0.0
        else:
            classification_accuracy = 0.0

        predicted_quality = self._extract_quality_score(output)

        if predicted_quality is None:
            quality_reward = -5
        else:
            quality_reward = -1 * abs(ground_truth_quality - predicted_quality)

        m = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            output,
            re.DOTALL,
        )
        if not is_factual:
            if not m:
                return FactCheckingRewardModelVerifyResponse(
                    **body.model_dump(),
                    factuality_accuracy=0,
                    factuality_f1_score=0.0,
                    num_errors=0,
                    reward=self.config.quality_weight * quality_reward,
                    quality_score=predicted_quality,
                    quality_reward=quality_reward,
                )

            predicted_factual_errors = m.group(1).strip()
            if self.config.rubric_parallel_evaluation and len(expected_errors) > 1:
                import asyncio

                evaluations = await asyncio.gather(
                    *[self._evaluate_single_error(err, predicted_factual_errors) for err in expected_errors]
                )
            else:
                evaluations = []
                for err in expected_errors:
                    evaluations.append(await self._evaluate_single_error(err, predicted_factual_errors))

            scores = [e.score for e in evaluations]
            num_errors = await self._evaluate_num_errors(
                "\n\n".join(expected_errors), predicted_factual_errors
            )
            classification_accuracy = classification_accuracy if num_errors > 0 else 0.0

            f1_score = self._aggregate_scores(
                scores=scores,
                num_predicted_errors=num_errors,
                num_ground_truth_errors=len(expected_errors),
            )
        else:
            f1_score = 1.0
            if not m:
                return FactCheckingRewardModelVerifyResponse(
                    **body.model_dump(),
                    reward=self.config.quality_weight * quality_reward,
                    factuality_accuracy=0.0,
                    factuality_f1_score=0.0,
                    num_errors=0,
                    quality_score=predicted_quality,
                    quality_reward=quality_reward,
                )
            predicted_factual_errors = m.group(1).strip()
            if not predicted_factual_errors.strip():
                num_errors = 0
            else:
                num_errors = await self._evaluate_num_errors(
                    "\n\n".join(expected_errors), predicted_factual_errors
                )
            f1_score = 1.0 if num_errors == 0 else 0.0

            classification_accuracy = classification_accuracy if num_errors == 0 else 0.0

        base_reward = self.config.factuality_weight * classification_accuracy * (1 + f1_score)
        reward = base_reward + self.config.quality_weight * quality_reward
        return FactCheckingRewardModelVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            factuality_accuracy=classification_accuracy,
            factuality_f1_score=f1_score,
            num_errors=num_errors,
            quality_score=predicted_quality,
            quality_reward=quality_reward,
        )


if __name__ == "__main__":
    FactCheckingRewardModelTavilyResourcesServer.run_webserver()
