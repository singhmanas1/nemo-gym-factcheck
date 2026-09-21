# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("yappi", types.SimpleNamespace())

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.fact_checking_reward_model_dev.app import (
    FactCheckingRewardModelDevConfig,
    FactCheckingRewardModelDevResourcesServer,
    FactCheckingRewardModelVerifyRequest,
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


def _make_server() -> FactCheckingRewardModelDevResourcesServer:
    config = FactCheckingRewardModelDevConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="fact_checking_reward_model_dev",
        retrieval_backend="none",
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_responses_create_params={"input": [], "max_output_tokens": 512},
        factuality_weight=1.0,
        severity_weight=0.1,
    )
    return FactCheckingRewardModelDevResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_request(
    response_text: str,
    *,
    expected_errors: list[str],
    hallucination_severity: float = 1.0,
) -> FactCheckingRewardModelVerifyRequest:
    return FactCheckingRewardModelVerifyRequest(
        id=1,
        expected_errors=expected_errors,
        hallucination_severity=hallucination_severity,
        responses_create_params={"input": []},
        response=_make_response(response_text),
    )


class _DummyHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _judge_prompt(json_body) -> str:
    inp = getattr(json_body, "input", None)
    if inp is None and isinstance(json_body, dict):
        inp = json_body.get("input")
    if not inp:
        return ""
    first = inp[0]
    content = getattr(first, "content", None)
    if content is None and isinstance(first, dict):
        content = first.get("content")
    return content if isinstance(content, str) else ""


def _errors_output(errors_body: str, severity: str = "1") -> str:
    return (
        f"[Beginning of Factual Severity]\n{severity}\n[End of Factual Severity]\n"
        f"[Beginning of Factual Errors]\n{errors_body}\n[End of Factual Errors]"
    )


class TestEmptyGoldVerify:
    def test_blank_box_skips_judge_and_scores_f1_one(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock()

        result = asyncio.run(
            server.verify(
                _make_verify_request(_errors_output(""), expected_errors=[])
            )
        )

        assert result.factuality_f1_score == 1.0
        assert result.num_errors == 0
        assert result.severity_reward == 1.0
        assert result.reward == 1.1
        server.server_client.post.assert_not_called()

    def test_none_found_text_is_f1_one_without_count_judge(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock()

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("No factual inaccuracies found."),
                    expected_errors=[],
                )
            )
        )

        assert result.factuality_f1_score == 1.0
        assert result.num_errors == 0
        assert result.reward == 1.1
        assert result.count_judge_response == "line_count"
        server.server_client.post.assert_not_called()

    def test_invented_error_is_f1_zero_from_line_count(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock()

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Karate always results in a knock-out."),
                    expected_errors=[],
                )
            )
        )

        assert result.factuality_f1_score == 0.0
        assert result.num_errors == 1
        assert result.reward == 0.1
        server.server_client.post.assert_not_called()

    def test_missing_tags_are_format_miss_even_when_gold_is_empty(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock()

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    "I ran out of tokens before writing a verdict.",
                    expected_errors=[],
                )
            )
        )

        assert result.factuality_f1_score == 0.0
        assert result.num_errors == 0
        server.server_client.post.assert_not_called()

    def test_nonempty_gold_still_uses_yes_no_matcher(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response("[[YES]]", response_id="match").model_dump()
            )
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )

        assert result.factuality_f1_score == 1.0
        assert result.num_errors == 1
        assert server.server_client.post.call_count == 1
        assert result.judge_evaluations is not None
        assert result.judge_evaluations[0].verdict == "YES"
        assert result.judge_evaluations[0].judge_response == "[[YES]]"

    def test_empty_matcher_response_is_retried(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            side_effect=[
                _DummyHTTPResponse(_make_response("", response_id="empty").model_dump()),
                _DummyHTTPResponse(
                    _make_response("[[YES]]", response_id="match").model_dump()
                ),
            ]
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )

        assert result.factuality_f1_score == 1.0
        assert result.judge_evaluations[0].verdict == "YES"
        assert server.server_client.post.call_count == 2

    def test_matcher_reads_reasoning_when_message_empty(self) -> None:
        server = _make_server()
        server.config.judge_empty_retries = 0
        reasoning_payload = {
            "id": "reason",
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [
                {
                    "id": "reason_rs",
                    "summary": [
                        {"text": "Same fact.\n[[YES]]", "type": "summary_text"}
                    ],
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(reasoning_payload)
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )

        assert result.factuality_f1_score == 1.0
        assert result.judge_evaluations[0].verdict == "YES"
        assert "[[YES]]" in result.judge_evaluations[0].judge_response

    def test_num_errors_uses_predicted_line_count(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response("[[YES]]", response_id="match").model_dump()
            )
        )

        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )

        assert result.num_errors == 1
        assert result.factuality_f1_score == 1.0
        assert result.count_judge_response == "line_count"
        assert result.timings["judge_count_s"] == 0.0
        assert result.timings["judge_count_attempts"] == 0
        server.server_client.post.assert_called_once()

    def test_verify_records_judge_timings(self) -> None:
        server = _make_server()
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response("[[YES]]", response_id="match").model_dump()
            )
        )
        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )
        assert result.timings is not None
        assert result.timings["verify_s"] >= 0.0
        assert result.timings["judge_yes_no_s"] is not None
        assert result.timings["judge_count_s"] == 0.0
        assert result.timings["judge_count_attempts"] == 0
        assert result.judge_evaluations[0].judge_s is not None
        assert result.judge_evaluations[0].judge_attempts == 1
        assert result.timings["judges_wall_s"] is not None

    def test_closed_think_in_message_is_scored_on_first_attempt(self) -> None:
        server = _make_server()
        server.config.judge_empty_retries = 0
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(
                _make_response(
                    "<think>Same incorrect fact.\n[[YES]]</think>",
                    response_id="match",
                ).model_dump()
            )
        )
        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )
        assert result.factuality_f1_score == 1.0
        assert result.num_errors == 1
        assert result.judge_evaluations[0].verdict == "YES"
        assert result.judge_evaluations[0].judge_attempts == 1
        assert server.server_client.post.call_count == 1

    def test_reasoning_tags_win_over_unlabeled_message(self) -> None:
        server = _make_server()
        server.config.judge_empty_retries = 0
        payload = {
            "id": "mixed",
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": [
                {
                    "id": "rs",
                    "summary": [{"text": "Same fact.\n[[YES]]", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "msg",
                    "content": [
                        {"annotations": [], "text": "Let me think.", "type": "output_text"}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        server.server_client.post = AsyncMock(
            return_value=_DummyHTTPResponse(payload)
        )
        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )
        assert result.judge_evaluations[0].verdict == "YES"
        assert result.judge_evaluations[0].judge_attempts == 1
        assert result.factuality_f1_score == 1.0

    def test_count_llm_is_not_called(self) -> None:
        server = _make_server()
        prompts = []

        async def post(*_args, **kwargs):
            prompts.append(_judge_prompt(kwargs.get("json")))
            return _DummyHTTPResponse(
                _make_response("[[YES]]", response_id="match").model_dump()
            )

        server.server_client.post = post
        result = asyncio.run(
            server.verify(
                _make_verify_request(
                    _errors_output("Hamlet was written by Charles Dickens."),
                    expected_errors=["Hamlet was written by Charles Dickens."],
                    hallucination_severity=4.0,
                )
            )
        )
        assert result.num_errors == 1
        assert result.count_judge_response == "line_count"
        assert len(prompts) == 1
        assert "<num_errors>" not in prompts[0]
        assert result.factuality_f1_score == 1.0
