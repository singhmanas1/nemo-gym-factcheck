# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import glob
import re
import subprocess
import time
import urllib.request
from html.parser import HTMLParser
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


class _TextExtractor(HTMLParser):
    """Lightweight HTML-to-text: strips tags, collapses whitespace."""
    _SKIP = frozenset(["script", "style", "noscript", "head", "nav", "footer", "header"])

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        lines = [" ".join(l.split()) for l in "".join(self._parts).splitlines()]
        return "\n".join(l for l in lines if l)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


class GenerativeRewardModelWithWikiResourcesServerConfig(BaseResourcesServerConfig):
    kiwix_zim_dir: Optional[str] = None
    kiwix_port: int = 8080
    kiwix_serve_path: str
    kiwix_max_chars: int = 8000  # max article text returned to model
    kiwix_top_k: int = 3  # number of articles to fetch per search
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    rubric_parallel_evaluation: bool = True
    rubric_yes_label: str = "[[YES]]"
    rubric_no_label: str = "[[NO]]"
    wiki_summary_prompt_template: str = Field(
        default="""You are preparing Wikipedia evidence for a factuality judge.

You will receive multiple retrieved Wikipedia articles. Your job is to turn them into a compact evidence packet that helps a downstream judge verify claims related to the search query.

Rules:
1. Use ONLY the provided Wikipedia content.
2. Do NOT add outside knowledge, assumptions, or speculation.
3. Extract only facts that are directly useful for checking claims related to the query.
4. Prefer concrete, atomic facts: names, dates, places, titles, numbers, definitions, and explicit relationships.
5. Remove irrelevant details, repetition, stylistic filler, and unnecessary background.
6. Combine evidence across articles only when it improves verification.
7. If the articles disagree, are ambiguous, or fail to answer the query, state that clearly.
8. Do NOT write a general summary of the articles.
9. Keep the output short and evidence-focused.

Inputs:

[Search Query]
{query}

[Wikipedia Articles]
{wiki_content}

Return output in exactly this format:

[Query-Relevant Evidence]
- [<article title>] <atomic fact>
- [<article title>] <atomic fact>
- [<article title>] <atomic fact>

[Supported By Retrieved Evidence]
- <claim or point that the retrieved evidence clearly supports>

[Not Established By Retrieved Evidence]
- <claim or point that the retrieved evidence does not establish, leaves ambiguous, or contradicts>

Requirements:
- Include 1 to 5 evidence bullets, ordered from most useful to least useful.
- Each bullet must contain only one atomic fact.
- Always include the article title.
- Prefer evidence that directly answers the query over broad background facts.
- If the articles do not provide useful evidence, say so explicitly.

If there is no useful evidence, return:

[Query-Relevant Evidence]
- none

[Supported By Retrieved Evidence]
- The retrieved Wikipedia articles do not clearly answer the query.

[Not Established By Retrieved Evidence]
- The retrieved content is too weak, broad, conflicting, or unrelated to verify the query.
""",
        description="Prompt template used to summarize Wikipedia evidence before returning it to the model.",
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


class GenerativeRewardModelWithWikiRunRequest(BaseRunRequest):
    id: Union[int, str]  # dataset may use int (e.g. question_id) or str (e.g. UUID-style)
    expected_errors: List[str]
    is_factual: bool
    loss_type: str


class GenerativeRewardModelWithWikiVerifyRequest(GenerativeRewardModelWithWikiRunRequest, BaseVerifyRequest):
    pass


class GenerativeRewardModelWithWikiVerifyResponse(BaseVerifyResponse):
    reward: float
    accuracy: float
    f1_score: float
    num_errors: int


class GenerativeRewardModelWithWikiResourcesServer(SimpleResourcesServer):
    config: GenerativeRewardModelWithWikiResourcesServerConfig
    _kiwix_process: Optional[subprocess.Popen] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _start_kiwix(self) -> None:
        if not self.config.kiwix_zim_dir:
            return
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.config.kiwix_port}/", timeout=1
            )
            print(
                f"Kiwix already running on port {self.config.kiwix_port}, reusing it.",
                flush=True,
            )
            return
        except Exception:
            pass
        zims = glob.glob(f"{self.config.kiwix_zim_dir.rstrip('/')}/*.zim")
        if not zims:
            print(f"Warning: no .zim files in {self.config.kiwix_zim_dir}, not starting Kiwix", flush=True)
            return
        port = self.config.kiwix_port
        kiwix_bin = self.config.kiwix_serve_path or "kiwix-serve"
        cmd = [kiwix_bin, "--port", str(port)] + zims
        print(f"Starting Kiwix on port {port} ({len(zims)} ZIM(s))...", flush=True)
        self._kiwix_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=False,
        )
        atexit.register(self._stop_kiwix)

        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                print("Kiwix server is ready.", flush=True)
                return
            except Exception:
                if self._kiwix_process.poll() is not None:
                    stderr = self._kiwix_process.stderr.read().decode() if self._kiwix_process.stderr else ""
                    raise RuntimeError(f"kiwix-serve exited early: {stderr}")
                time.sleep(1)
        raise RuntimeError("Kiwix server did not become ready in time")

    def _stop_kiwix(self) -> None:
        if self._kiwix_process is not None and self._kiwix_process.poll() is None:
            self._kiwix_process.terminate()
            self._kiwix_process.wait(timeout=5)
        self._kiwix_process = None

    def setup_webserver(self) -> FastAPI:
        self._start_kiwix()
        app = super().setup_webserver()
        app.post("/search_wiki")(self.search_wiki)
        return app

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
        """
        Compute F1 from judged matches.

        - `scores` are per-ground-truth match scores in [0, 1] (YES=1, NO=0).
        - `num_predicted_errors` is the number of predicted error lines.
        - `num_ground_truth_errors` is the number of expected errors.
        """
        num_predicted_errors = max(0, int(num_predicted_errors))
        num_ground_truth_errors = max(0, int(num_ground_truth_errors))

        # Perfect when both sides contain no errors.
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

    def _kiwix_url(self) -> str:
        return f"http://127.0.0.1:{self.config.kiwix_port}"

    def _kiwix_search(self, query: str) -> list[dict]:
        """Search Kiwix, return [{title, url, word_count}, ...]."""
        resp = urllib.request.urlopen(
            f"{self._kiwix_url()}/search?pattern={urllib.request.quote(query)}&pageLength={self.config.kiwix_top_k}",
            timeout=10,
        )
        html = resp.read().decode()
        results = []
        for m in re.finditer(
            r'<li>\s*<a\s+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>.*?<div class="informations">([\d,]+)\s*words</div>',
            html, re.DOTALL,
        ):
            results.append({
                "title": re.sub(r"<[^>]+>", "", m.group(2)).strip(),
                "url": m.group(1).strip(),
                "word_count": int(m.group(3).replace(",", "")),
            })
        return results

    def _kiwix_fetch_article(self, path: str) -> str:
        """Fetch a full article from Kiwix and return clean text, truncated."""
        url = f"{self._kiwix_url()}{path}" if path.startswith("/") else f"{self._kiwix_url()}/{path}"
        resp = urllib.request.urlopen(url, timeout=10)
        text = _html_to_text(resp.read().decode())
        max_c = self.config.kiwix_max_chars
        if len(text) > max_c:
            trunc = text[:max_c]
            last_dot = trunc.rfind(".")
            if last_dot > max_c // 2:
                trunc = trunc[: last_dot + 1]
            text = trunc + "\n\n[...article truncated...]"
        return text

    async def _summarize_wiki_content(
        self,
        query: str,
        wiki_content: str,
    ) -> str:
        if not self.config.judge_model_server or not self.config.judge_responses_create_params:
            return wiki_content

        judge_prompt = self.config.wiki_summary_prompt_template.format(
            query=query,
            wiki_content=wiki_content,
        )
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
        summary = self._extract_text_from_response(judge_response_obj)
        return summary or wiki_content

    async def search_wiki(self, body: SearchWikiRequest) -> SearchWikiResponse:
        """Search Kiwix for the query, fetch top article(s), return clean text."""
        try:
            results = self._kiwix_search(body.query)
            if not results:
                return SearchWikiResponse(content=f"No Wikipedia results found for: {body.query}")
            parts = []
            for r in results[: self.config.kiwix_top_k]:
                article = self._kiwix_fetch_article(r["url"])
                parts.append(f"=== Wikipedia: {r['title']} ===\n\n{article}")
            wiki_content = "\n\n".join(parts)
            summarized_content = await self._summarize_wiki_content(
                query=body.query,
                wiki_content=wiki_content,
            )
            return SearchWikiResponse(content=summarized_content)
        except Exception as e:
            return SearchWikiResponse(content=f"Wiki search error: {e}. Query was: {body.query}")
    
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

    async def verify(self, body: GenerativeRewardModelWithWikiVerifyRequest) -> GenerativeRewardModelWithWikiVerifyResponse:
        output = self._extract_text_from_response(body.response)

        expected_errors = [x for x in body.expected_errors if x.strip()]
        is_factual = body.is_factual

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


        # Require a non-empty factual-errors block between the markers.
        m = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            output,
            re.DOTALL,
        )
        if not is_factual:
            if not m:
                return GenerativeRewardModelWithWikiVerifyResponse(
                    **body.model_dump(),
                    accuracy=0,
                    f1_score=0.0,
                    num_errors=0,
                    reward=0,
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
                return GenerativeRewardModelWithWikiVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    accuracy=0.0,
                    f1_score=0.0,
                    num_errors=0,
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

        reward = classification_accuracy * (1 + f1_score)
        return GenerativeRewardModelWithWikiVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            accuracy=classification_accuracy,
            f1_score=f1_score,
            num_errors=num_errors,
        )



if __name__ == "__main__":
    GenerativeRewardModelWithWikiResourcesServer.run_webserver()

