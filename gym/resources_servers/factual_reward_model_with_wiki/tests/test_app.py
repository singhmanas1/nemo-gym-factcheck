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
import sys
import types
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("yappi", types.SimpleNamespace())

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.factual_reward_model_with_wiki.app import (
    GenerativeRewardModelWithWikiResourcesServer,
    GenerativeRewardModelWithWikiResourcesServerConfig,
    GenerativeRewardModelWithWikiVerifyRequest,
    SearchWikiRequest,
)


def _make_response(text: str, response_id: str = "resp_test") -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": response_id,
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [
                {
                    "id": f"{response_id}_msg",
                    "content": [
                        {"annotations": [], "text": text, "type": "output_text"}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _make_server() -> GenerativeRewardModelWithWikiResourcesServer:
    config = GenerativeRewardModelWithWikiResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="factual_reward_model_with_wiki",
        kiwix_serve_path="kiwix-serve",
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_responses_create_params={"input": [], "max_output_tokens": 512},
    )
    return GenerativeRewardModelWithWikiResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_request(
    response_text: str,
    *,
    expected_errors: list[str],
    is_factual: bool,
) -> GenerativeRewardModelWithWikiVerifyRequest:
    return GenerativeRewardModelWithWikiVerifyRequest(
        id=1,
        expected_errors=expected_errors,
        is_factual=is_factual,
        loss_type="classification",
        responses_create_params={"input": []},
        response=_make_response(response_text),
    )


class _DummyHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestGenerativeRewardModelWithWikiApp:
    def test_search_wiki_summarizes_results(self) -> None:
        server = _make_server()
        server._kiwix_search = MagicMock(
            return_value=[{"title": "Hamlet", "url": "/wiki/Hamlet", "word_count": 100}]
        )
        server._kiwix_fetch_article = MagicMock(
            return_value="Hamlet is a tragedy written by William Shakespeare."
        )
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response(
                    "[Query-Relevant Evidence]\n- [Hamlet] Hamlet is a tragedy by William Shakespeare.",
                    response_id="summary_resp",
                ).model_dump()
            )
        )

        result = asyncio.run(server.search_wiki(SearchWikiRequest(query="Who wrote Hamlet?")))

        assert "[Query-Relevant Evidence]" in result.content
        assert server.server_client.post.await_count == 1

    def test_verify_non_factual_response_returns_partial_reward(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            side_effect=[
                _DummyHTTPResponse(
                    _make_response("[[YES]]", response_id="judge_match").model_dump()
                ),
                _DummyHTTPResponse(
                    _make_response("<num_errors>1</num_errors>", response_id="judge_count").model_dump()
                ),
            ]
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    """
[Beginning of Factual Errors]
Hamlet was written by Charles Dickens.
[End of Factual Errors]
""",
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    is_factual=False,
                )
            )
        )

        assert result.accuracy == 1.0
        assert result.f1_score == 1.0
        assert result.num_errors == 1
        assert result.reward == 1.0

    def test_verify_factual_response_requires_zero_errors(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response("<num_errors>0</num_errors>", response_id="judge_count").model_dump()
            )
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    """
[Beginning of Factual Errors]
[End of Factual Errors]
""",
                    expected_errors=[""],
                    is_factual=True,
                )
            )
        )

        assert result.accuracy == 1.0
        assert result.f1_score == 1.0
        assert result.num_errors == 0
        assert result.reward == 1.0

    def test_verify_missing_error_block_returns_zero_reward_for_non_factual(self) -> None:
        server = _make_server()

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    "No factual error block here.",
                    expected_errors=["A factual error."],
                    is_factual=False,
                )
            )
        )

        assert result.accuracy == 0
        assert result.f1_score == 0.0
        assert result.num_errors == 0
        assert result.reward == 0
