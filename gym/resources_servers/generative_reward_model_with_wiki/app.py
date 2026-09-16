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
from pydantic import BaseModel

from score_parser import extract_scores

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
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
    C_1: int = 1
    C_2: int = 1
    kiwix_zim_dir: Optional[str] = None
    kiwix_port: int = 8080
    kiwix_serve_path: str
    kiwix_max_chars: int = 8000  # max article text returned to model
    kiwix_top_k: int = 3  # number of articles to fetch per search


class SearchWikiRequest(BaseModel):
    query: str


class SearchWikiResponse(BaseModel):
    content: str


class GenerativeRewardModelWithWikiRunRequest(BaseRunRequest):
    id: Union[int, str]  # dataset may use int (e.g. question_id) or str (e.g. UUID-style)
    ground_truth_ranking: Optional[float] = None
    ground_truth_score_1: float
    ground_truth_score_2: Optional[float] = None
    num_responses: int = 1


class GenerativeRewardModelWithWikiVerifyRequest(GenerativeRewardModelWithWikiRunRequest, BaseVerifyRequest):
    pass


class GenerativeRewardModelWithWikiVerifyResponse(BaseVerifyResponse):
    predicted_score_1: float
    predicted_score_2: Optional[float] = None
    ground_truth_score_1: float
    ground_truth_score_2: Optional[float] = None
    ground_truth_ranking: Optional[float] = None
    predicted_ranking: Optional[float] = None
    format_correct: bool
    C_1: Optional[int] = None
    C_2: Optional[int] = None
    num_responses: Optional[int] = None


class GenerativeRewardModelWithWikiResourcesServer(SimpleResourcesServer):
    config: GenerativeRewardModelWithWikiResourcesServerConfig
    _kiwix_process: Optional[subprocess.Popen] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _start_kiwix(self) -> None:
        if not self.config.kiwix_zim_dir:
            return
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
            return SearchWikiResponse(content="\n\n".join(parts))
        except Exception as e:
            return SearchWikiResponse(content=f"Wiki search error: {e}. Query was: {body.query}")

    async def verify(self, body: GenerativeRewardModelWithWikiVerifyRequest) -> GenerativeRewardModelWithWikiVerifyResponse:
        # Get the final text response from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                # Extract text from the nested content structure
                final_response_text = last_output.content[0].text

        output = final_response_text.split("</think>")[-1].strip()
        individual_scores, ranking_score, format_correct = extract_scores(output, body.num_responses)
        C_1, C_2 = self.config.C_1, self.config.C_2
        # Penalize wrong format (C_1), score errors, and ranking error (C_2)
        reward = -1 * C_1 * (0 if format_correct else 1)
        reward += -1 * abs(body.ground_truth_score_1 - individual_scores[0])

        if body.num_responses == 2:
            reward += -1 * abs(body.ground_truth_score_2 - individual_scores[1])
            reward += -1 * C_2 * abs(body.ground_truth_ranking - ranking_score)

        predicted_score_2 = individual_scores[1] if body.num_responses == 2 else None
        return GenerativeRewardModelWithWikiVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            predicted_score_1=individual_scores[0],
            predicted_score_2=predicted_score_2,
            predicted_ranking=ranking_score if body.num_responses == 2 else None,
            format_correct=format_correct,
            C_1=C_1,
            C_2=C_2,
        )


if __name__ == "__main__":
    GenerativeRewardModelWithWikiResourcesServer.run_webserver()

