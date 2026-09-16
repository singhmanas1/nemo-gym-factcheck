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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.fact_hallucination_detection.app import (
    FactHallucinationDetectionConfig,
    FactHallucinationDetectionResourcesServer,
    FactHallucinationDetectionVerifyRequest,
    KiwixSearchBackend,
    SearchBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response_output(text: str) -> list:
    return [
        {
            "id": "msg_test",
            "content": [{"annotations": [], "text": text, "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]


def _make_nemo_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=_make_response_output(text),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(
    response_text: str,
    question: str = "When was the Eiffel Tower built?",
    search_queries: list = None,
) -> FactHallucinationDetectionVerifyRequest:
    response = _make_nemo_response(response_text)
    return FactHallucinationDetectionVerifyRequest(
        id=1,
        question=question,
        search_queries=search_queries,
        responses_create_params={"input": [{"role": "user", "content": question}]},
        response=response,
    )


def _make_server(search_backend: SearchBackend = None) -> FactHallucinationDetectionResourcesServer:
    config = FactHallucinationDetectionConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={"type": "responses_api_models", "name": "genrm_model"},
        judge_responses_create_params={"input": [], "max_output_tokens": 4096},
    )
    server = FactHallucinationDetectionResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )
    if search_backend is None:
        search_backend = _StubSearchBackend()
    server._search_backend = search_backend
    return server


class _StubSearchBackend(SearchBackend):
    """Returns canned evidence for any query."""

    def __init__(self, evidence: str = "The Eiffel Tower was completed in 1889."):
        self._evidence = evidence

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def search(self, query: str, top_k: int = 3) -> str:
        return self._evidence


# ---------------------------------------------------------------------------
# Tests: _extract_text_from_response
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_basic(self):
        text = FactHallucinationDetectionResourcesServer._extract_text_from_response(
            _make_nemo_response("Hello world")
        )
        assert text == "Hello world"

    def test_strips_think_tags(self):
        text = FactHallucinationDetectionResourcesServer._extract_text_from_response(
            _make_nemo_response("<think>reasoning</think>The answer is 42.")
        )
        assert text == "The answer is 42."

    def test_empty_response(self):
        resp = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        text = FactHallucinationDetectionResourcesServer._extract_text_from_response(resp)
        assert text == ""


# ---------------------------------------------------------------------------
# Tests: _has_factual_errors
# ---------------------------------------------------------------------------

class TestHasFactualErrors:
    def test_no_markers(self):
        assert not FactHallucinationDetectionResourcesServer._has_factual_errors(
            "The response is correct."
        )

    def test_empty_block(self):
        text = "[Beginning of Factual Errors]\n\n[End of Factual Errors]"
        assert not FactHallucinationDetectionResourcesServer._has_factual_errors(text)

    def test_whitespace_only_block(self):
        text = "[Beginning of Factual Errors]   \n  \n[End of Factual Errors]"
        assert not FactHallucinationDetectionResourcesServer._has_factual_errors(text)

    def test_errors_present(self):
        text = (
            "[Beginning of Factual Errors]\n"
            "1. The Eiffel Tower was not built in 1901.\n"
            "[End of Factual Errors]"
        )
        assert FactHallucinationDetectionResourcesServer._has_factual_errors(text)


# ---------------------------------------------------------------------------
# Tests: _compute_reward (binary mode)
# ---------------------------------------------------------------------------

class TestComputeReward:
    def _server(self):
        return _make_server()

    def test_no_errors_returns_1(self):
        server = self._server()
        assert server._compute_reward("The response looks good.") == 1.0

    def test_empty_error_block_returns_1(self):
        server = self._server()
        text = "[Beginning of Factual Errors]\n[End of Factual Errors]"
        assert server._compute_reward(text) == 1.0

    def test_errors_present_returns_0(self):
        server = self._server()
        text = (
            "[Beginning of Factual Errors]\n"
            "The year is wrong.\n"
            "[End of Factual Errors]"
        )
        assert server._compute_reward(text) == 0.0


# ---------------------------------------------------------------------------
# Tests: verify (end-to-end with mocked judge)
# ---------------------------------------------------------------------------

class TestVerify:
    def _run_verify(self, policy_response: str, judge_response_text: str) -> float:
        server = _make_server()
        judge_response = _make_nemo_response(judge_response_text)

        mock_http_response = AsyncMock()
        mock_http_response.json = AsyncMock(return_value=judge_response.model_dump())
        server.server_client.post = AsyncMock(return_value=mock_http_response)

        request = _make_verify_request(policy_response)
        result = asyncio.run(server.verify(request))
        return result.reward

    def test_correct_response_gets_reward_1(self):
        reward = self._run_verify(
            policy_response="The Eiffel Tower was completed in 1889.",
            judge_response_text="No factual errors found in the response.",
        )
        assert reward == 1.0

    def test_hallucinated_response_gets_reward_0(self):
        reward = self._run_verify(
            policy_response="The Eiffel Tower was completed in 1901.",
            judge_response_text=(
                "[Beginning of Factual Errors]\n"
                "The Eiffel Tower was completed in 1889, not 1901.\n"
                "[End of Factual Errors]"
            ),
        )
        assert reward == 0.0

    def test_judge_returns_empty_block_gets_reward_1(self):
        reward = self._run_verify(
            policy_response="Water boils at 100 degrees Celsius at sea level.",
            judge_response_text="[Beginning of Factual Errors]\n[End of Factual Errors]",
        )
        assert reward == 1.0

    def test_verify_response_includes_evidence(self):
        server = _make_server(_StubSearchBackend(evidence="Custom evidence text"))
        judge_response = _make_nemo_response("All good.")

        mock_http_response = AsyncMock()
        mock_http_response.json = AsyncMock(return_value=judge_response.model_dump())
        server.server_client.post = AsyncMock(return_value=mock_http_response)

        request = _make_verify_request("Some answer")
        result = asyncio.run(server.verify(request))
        assert result.evidence == "Custom evidence text"
        assert result.judge_output == "All good."


# ---------------------------------------------------------------------------
# Tests: SearchBackend abstraction
# ---------------------------------------------------------------------------

class TestSearchBackend:
    def test_stub_backend_returns_evidence(self):
        backend = _StubSearchBackend("test evidence")
        assert backend.search("anything") == "test evidence"

    def test_kiwix_backend_is_subclass(self):
        assert issubclass(KiwixSearchBackend, SearchBackend)

    def test_retrieve_evidence_uses_question_as_default_query(self):
        calls = []

        class _TrackingBackend(SearchBackend):
            def start(self): pass
            def stop(self): pass
            def search(self, query, top_k=3):
                calls.append(query)
                return "evidence"

        server = _make_server(_TrackingBackend())
        request = _make_verify_request("answer", question="Who invented electricity?")
        server._retrieve_evidence(request)
        assert calls == ["Who invented electricity?"]

    def test_retrieve_evidence_uses_search_queries_when_provided(self):
        calls = []

        class _TrackingBackend(SearchBackend):
            def start(self): pass
            def stop(self): pass
            def search(self, query, top_k=3):
                calls.append(query)
                return "evidence"

        server = _make_server(_TrackingBackend())
        request = _make_verify_request(
            "answer",
            question="Tell me about Einstein",
            search_queries=["Albert Einstein biography", "Einstein relativity"],
        )
        server._retrieve_evidence(request)
        assert calls == ["Albert Einstein biography", "Einstein relativity"]
