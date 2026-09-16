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
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.abstention_proto_v1.app import (
    AbstentionProtoV1Config,
    AbstentionProtoV1Server,
    AbstentionProtoV1VerifyRequest,
    extract_boxed_answer,
    extract_text_from_response,
    normalize_answer,
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
    answer: str = "1889",
) -> AbstentionProtoV1VerifyRequest:
    response = _make_nemo_response(response_text)
    return AbstentionProtoV1VerifyRequest(
        id=1,
        question=question,
        answer=answer,
        responses_create_params={
            "input": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": question},
            ]
        },
        response=response,
    )


def _make_server(abstention_reward: float = 0.5) -> AbstentionProtoV1Server:
    config = AbstentionProtoV1Config(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        abstention_reward=abstention_reward,
    )
    return AbstentionProtoV1Server(config=config, server_client=MagicMock(spec=ServerClient))


# ---------------------------------------------------------------------------
# Tests: extract_boxed_answer
# ---------------------------------------------------------------------------


class TestExtractBoxedAnswer:
    def test_simple(self):
        assert extract_boxed_answer("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed_answer("\\boxed{f(x) = {x + 1}}") == "f(x) = {x + 1}"

    def test_no_boxed(self):
        assert extract_boxed_answer("No boxed answer here") is None

    def test_last_boxed_wins(self):
        assert extract_boxed_answer("\\boxed{wrong} then \\boxed{right}") == "right"

    def test_idk_token(self):
        assert extract_boxed_answer("\\boxed{[IDK]}") == "[IDK]"

    def test_unclosed_brace(self):
        assert extract_boxed_answer("\\boxed{unclosed") is None

    def test_empty_boxed(self):
        assert extract_boxed_answer("\\boxed{}") == ""

    def test_boxed_with_thinking(self):
        text = "<think>Let me think...</think>\nThe answer is \\boxed{yes}"
        assert extract_boxed_answer(text) == "yes"


# ---------------------------------------------------------------------------
# Tests: normalize_answer
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"

    def test_strip_articles(self):
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"

    def test_strip_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("  foo   bar  ") == "foo bar"

    def test_combined(self):
        assert normalize_answer("  The Answer Is: YES!  ") == "answer is yes"

    def test_idk_normalization(self):
        assert normalize_answer("[IDK]") == "idk"
        assert normalize_answer("[idk]") == "idk"


# ---------------------------------------------------------------------------
# Tests: extract_text_from_response
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_basic(self):
        assert extract_text_from_response(_make_nemo_response("Hello world")) == "Hello world"

    def test_strips_think_tags(self):
        text = extract_text_from_response(_make_nemo_response("<think>reasoning</think>The answer is 42."))
        assert text == "The answer is 42."

    def test_strips_thinking_tags(self):
        text = extract_text_from_response(_make_nemo_response("<thinking>reasoning</thinking>The answer is 42."))
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
        assert extract_text_from_response(resp) == ""


# ---------------------------------------------------------------------------
# Tests: verify (end-to-end)
# ---------------------------------------------------------------------------


class TestVerify:
    def _run_verify(self, response_text: str, answer: str = "1889", abstention_reward: float = 0.5):
        server = _make_server(abstention_reward=abstention_reward)
        request = _make_verify_request(response_text, answer=answer)
        return asyncio.run(server.verify(request))

    def test_correct_boxed_answer(self):
        result = self._run_verify("\\boxed{1889}", answer="1889")
        assert result.reward == 1.0
        assert result.verdict == "correct"
        assert result.is_correct == 1.0
        assert result.is_abstain == 0.0
        assert result.is_incorrect == 0.0
        assert result.omniscience_index == 1.0

    def test_correct_answer_case_insensitive(self):
        result = self._run_verify("\\boxed{Yes}", answer="yes")
        assert result.reward == 1.0
        assert result.verdict == "correct"

    def test_incorrect_answer(self):
        result = self._run_verify("\\boxed{1901}", answer="1889")
        assert result.reward == 0.0
        assert result.verdict == "incorrect"
        assert result.is_correct == 0.0
        assert result.is_abstain == 0.0
        assert result.is_incorrect == 1.0
        assert result.omniscience_index == -1.0

    def test_abstention_idk(self):
        result = self._run_verify("\\boxed{[IDK]}", answer="1889")
        assert result.reward == 0.5
        assert result.verdict == "abstain"
        assert result.is_correct == 0.0
        assert result.is_abstain == 1.0
        assert result.is_incorrect == 0.0
        assert result.omniscience_index == 0.0

    def test_abstention_custom_lambda(self):
        result = self._run_verify("\\boxed{[IDK]}", answer="1889", abstention_reward=0.3)
        assert result.reward == 0.3
        assert result.verdict == "abstain"

    def test_no_boxed_falls_back_to_full_text(self):
        result = self._run_verify("1889", answer="1889")
        assert result.reward == 1.0
        assert result.verdict == "correct"

    def test_no_boxed_incorrect(self):
        result = self._run_verify("I think it was 1901", answer="1889")
        assert result.reward == 0.0
        assert result.verdict == "incorrect"

    def test_thinking_stripped_before_extraction(self):
        result = self._run_verify("<think>Let me think about this...</think>\\boxed{1889}", answer="1889")
        assert result.reward == 1.0
        assert result.verdict == "correct"

    def test_response_includes_metadata(self):
        result = self._run_verify("\\boxed{1889}", answer="1889")
        assert result.extracted_answer == "1889"
        assert result.ground_truth == "1889"

    def test_articles_stripped_in_comparison(self):
        result = self._run_verify("\\boxed{The Eiffel Tower}", answer="Eiffel Tower")
        assert result.reward == 1.0
        assert result.verdict == "correct"
