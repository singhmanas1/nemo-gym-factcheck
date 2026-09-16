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
Fact Hallucination Detection Resources Server.

Trains a policy model to generate factually accurate responses by using a
trained Generative Reward Model (GenRM) as the judge.  The judge receives
the policy response together with retrieved evidence and identifies any
factual errors.  In binary reward mode the policy receives 1.0 when the
judge finds zero errors and 0.0 otherwise.

The retrieval backend is abstracted behind ``SearchBackend`` so that Kiwix
can be swapped for Google, Bing, or any other source without touching the
core reward logic.
"""
from __future__ import annotations

import abc
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


# ---------------------------------------------------------------------------
# Search backend abstraction
# ---------------------------------------------------------------------------

class SearchBackend(abc.ABC):
    """Interface for pluggable retrieval backends."""

    @abc.abstractmethod
    def start(self) -> None:
        """Perform any setup required before the first search (start processes, etc.)."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear down resources acquired in ``start``."""

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 3) -> str:
        """Return evidence text for *query*.

        The returned string is plain text ready to be injected into a judge
        prompt.  Implementations decide how many results to fetch and how to
        format them.
        """


# ---------------------------------------------------------------------------
# Kiwix implementation
# ---------------------------------------------------------------------------

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
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    return p.get_text()


class KiwixSearchBackend(SearchBackend):
    """Retrieval backend backed by a local kiwix-serve process."""

    def __init__(
        self,
        zim_dir: Optional[str],
        port: int,
        serve_path: str,
        max_chars: int,
    ):
        self._zim_dir = zim_dir
        self._port = port
        self._serve_path = serve_path
        self._max_chars = max_chars
        self._process: Optional[subprocess.Popen] = None

    @property
    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self._zim_dir:
            return
        zims = glob.glob(f"{self._zim_dir.rstrip('/')}/*.zim")
        if not zims:
            print(f"Warning: no .zim files in {self._zim_dir}, not starting Kiwix", flush=True)
            return
        kiwix_bin = self._serve_path or "kiwix-serve"
        cmd = [kiwix_bin, "--port", str(self._port)] + zims
        print(f"Starting Kiwix on port {self._port} ({len(zims)} ZIM(s))...", flush=True)
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=False,
        )
        atexit.register(self.stop)

        for _ in range(30):
            try:
                urllib.request.urlopen(f"{self._base_url}/", timeout=1)
                print("Kiwix server is ready.", flush=True)
                return
            except Exception:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    raise RuntimeError(f"kiwix-serve exited early: {stderr}")
                time.sleep(1)
        raise RuntimeError("Kiwix server did not become ready in time")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)
        self._process = None

    # -- search --------------------------------------------------------------

    def _raw_search(self, query: str, top_k: int) -> list[dict]:
        resp = urllib.request.urlopen(
            f"{self._base_url}/search?pattern={urllib.request.quote(query)}&pageLength={top_k}",
            timeout=10,
        )
        html = resp.read().decode()
        results = []
        for m in re.finditer(
            r'<li>\s*<a\s+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>.*?'
            r'<div class="informations">([\d,]+)\s*words</div>',
            html,
            re.DOTALL,
        ):
            results.append({
                "title": re.sub(r"<[^>]+>", "", m.group(2)).strip(),
                "url": m.group(1).strip(),
                "word_count": int(m.group(3).replace(",", "")),
            })
        return results

    def _fetch_article(self, path: str) -> str:
        url = f"{self._base_url}{path}" if path.startswith("/") else f"{self._base_url}/{path}"
        resp = urllib.request.urlopen(url, timeout=10)
        text = _html_to_text(resp.read().decode())
        if len(text) > self._max_chars:
            trunc = text[: self._max_chars]
            last_dot = trunc.rfind(".")
            if last_dot > self._max_chars // 2:
                trunc = trunc[: last_dot + 1]
            text = trunc + "\n\n[...article truncated...]"
        return text

    def search(self, query: str, top_k: int = 3) -> str:
        try:
            results = self._raw_search(query, top_k)
            if not results:
                return f"No results found for: {query}"
            parts = []
            for r in results[:top_k]:
                article = self._fetch_article(r["url"])
                parts.append(f"=== Wikipedia: {r['title']} ===\n\n{article}")
            return "\n\n".join(parts)
        except Exception as e:
            return f"Search error: {e}. Query was: {query}"


# ---------------------------------------------------------------------------
# Server config, request/response models
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_PROMPT = """\
You are a factual accuracy judge. You are given a QUESTION that was posed to
an AI assistant, the assistant's RESPONSE, and EVIDENCE retrieved from
Wikipedia.

Your task: identify every factual error in the RESPONSE using the EVIDENCE.
List each error between [Beginning of Factual Errors] and
[End of Factual Errors] markers.  If the response is fully correct, leave the
block empty or do not include it.

QUESTION:
{question}

RESPONSE:
{response}

EVIDENCE:
{evidence}

Analyze carefully, then list any factual errors found."""


class FactHallucinationDetectionConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_template: str = Field(
        default=_DEFAULT_JUDGE_PROMPT,
        description="Prompt sent to the GenRM judge. Placeholders: {question}, {response}, {evidence}.",
    )

    reward_mode: str = Field(
        default="binary",
        description=(
            "How to compute the reward from the judge output. "
            "'binary': 1.0 if zero factual errors detected, 0.0 otherwise."
        ),
    )

    # Kiwix / search-backend settings
    search_backend_type: str = Field(
        default="kiwix",
        description="Which search backend to use. Currently supported: 'kiwix'.",
    )
    kiwix_zim_dir: Optional[str] = None
    kiwix_port: int = 8083
    kiwix_serve_path: str = "kiwix-serve"
    kiwix_max_chars: int = 8000
    kiwix_top_k: int = 3


class FactHallucinationDetectionRunRequest(BaseRunRequest):
    id: Union[int, str]
    question: str
    search_queries: Optional[List[str]] = None


class FactHallucinationDetectionVerifyRequest(
    FactHallucinationDetectionRunRequest, BaseVerifyRequest
):
    pass


class FactHallucinationDetectionVerifyResponse(BaseVerifyResponse):
    reward: float
    judge_output: Optional[str] = None
    evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FactHallucinationDetectionResourcesServer(SimpleResourcesServer):
    config: FactHallucinationDetectionConfig
    _search_backend: Optional[SearchBackend] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_search_backend(self) -> SearchBackend:
        cfg = self.config
        if cfg.search_backend_type == "kiwix":
            return KiwixSearchBackend(
                zim_dir=cfg.kiwix_zim_dir,
                port=cfg.kiwix_port,
                serve_path=cfg.kiwix_serve_path,
                max_chars=cfg.kiwix_max_chars,
            )
        raise ValueError(f"Unknown search_backend_type: {cfg.search_backend_type}")

    def setup_webserver(self) -> FastAPI:
        self._search_backend = self._build_search_backend()
        self._search_backend.start()
        app = super().setup_webserver()
        return app

    # -- helpers -------------------------------------------------------------

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

    def _retrieve_evidence(self, body: FactHallucinationDetectionRunRequest) -> str:
        queries = body.search_queries if body.search_queries else [body.question]
        parts = []
        for q in queries:
            parts.append(self._search_backend.search(q, top_k=self.config.kiwix_top_k))
        return "\n\n".join(parts)

    @staticmethod
    def _has_factual_errors(judge_text: str) -> bool:
        """Return True if the judge listed any errors between the markers."""
        m = re.search(
            r"\[Beginning of Factual Errors\](.*?)\[End of Factual Errors\]",
            judge_text,
            re.DOTALL,
        )
        if not m:
            return False
        return bool(m.group(1).strip())

    def _compute_reward(self, judge_text: str) -> float:
        if self.config.reward_mode == "binary":
            return 0.0 if self._has_factual_errors(judge_text) else 1.0
        raise ValueError(f"Unknown reward_mode: {self.config.reward_mode}")

    # -- verify --------------------------------------------------------------

    async def verify(
        self, body: FactHallucinationDetectionVerifyRequest
    ) -> FactHallucinationDetectionVerifyResponse:
        policy_output = self._extract_text_from_response(body.response)
        evidence = self._retrieve_evidence(body)

        judge_prompt = self.config.judge_prompt_template.format(
            question=body.question,
            response=policy_output,
            evidence=evidence,
        )

        msgs: List[NeMoGymEasyInputMessage] = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]
        request_params = self.config.judge_responses_create_params.model_copy(deep=True)
        request_params.input = msgs

        response_obj = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        judge_response = NeMoGymResponse.model_validate(await response_obj.json())
        judge_text = self._extract_text_from_response(judge_response)

        reward = self._compute_reward(judge_text)

        return FactHallucinationDetectionVerifyResponse(
            **body.model_dump(),
            reward=reward,
            judge_output=judge_text,
            evidence=evidence,
        )


if __name__ == "__main__":
    FactHallucinationDetectionResourcesServer.run_webserver()
