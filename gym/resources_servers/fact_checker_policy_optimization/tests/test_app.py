import asyncio
import sys
import types
from unittest.mock import MagicMock

sys.modules.setdefault("yappi", types.SimpleNamespace())

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.fact_checker_policy_optimization.app import (
    FactCheckerPolicyOptimizationConfig,
    FactCheckerPolicyOptimizationResourcesServer,
    FactCheckerPolicyOptimizationVerifyRequest,
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


def _make_request(policy_answer: str) -> FactCheckerPolicyOptimizationVerifyRequest:
    return FactCheckerPolicyOptimizationVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Who wrote Hamlet?"}],
            "tools": [],
            "parallel_tool_calls": False,
        },
        response=_make_response(policy_answer, response_id="policy_resp"),
    )


class _DummyHTTPResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload


class TestFactCheckerPolicyOptimizationApp:
    def _server(self, checker_text: str) -> FactCheckerPolicyOptimizationResourcesServer:
        config = FactCheckerPolicyOptimizationConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="fact_checker_policy_optimization",
            fact_checker_agent_server={
                "type": "responses_api_agents",
                "name": "fact_checker_policy_optimization_fact_checker",
            },
            fact_checker_responses_create_params={"input": []},
        )
        server_client = MagicMock(spec=ServerClient)

        async def _post(*args, **kwargs):
            return _DummyHTTPResponse(
                _make_response(checker_text, response_id="checker_resp").model_dump()
            )

        server_client.post.side_effect = _post
        return FactCheckerPolicyOptimizationResourcesServer(
            config=config,
            server_client=server_client,
        )

    def _server_with_payload(
        self, checker_payload
    ) -> FactCheckerPolicyOptimizationResourcesServer:
        config = FactCheckerPolicyOptimizationConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="fact_checker_policy_optimization",
            fact_checker_agent_server={
                "type": "responses_api_agents",
                "name": "fact_checker_policy_optimization_fact_checker",
            },
            fact_checker_responses_create_params={"input": []},
        )
        server_client = MagicMock(spec=ServerClient)

        async def _post(*args, **kwargs):
            return _DummyHTTPResponse(checker_payload)

        server_client.post.side_effect = _post
        return FactCheckerPolicyOptimizationResourcesServer(
            config=config,
            server_client=server_client,
        )

    def test_rewards_factual_answers(self) -> None:
        checker_text = """
[Beginning of Factuality Prediction]
YES
[End of Factuality Prediction]

[Beginning of Factual Errors]
[End of Factual Errors]
"""
        result = asyncio.run(
            self._server(checker_text).verify(
                _make_request("Hamlet was written by William Shakespeare.")
            )
        )
        assert result.reward == 1.0
        assert result.hallucinated is False
        assert result.checker_prediction == "YES"
        assert result.checker_num_errors == 0
        assert result.checker_response_debug["id"] == "checker_resp"
        assert "message" in result.checker_response_debug["output_types"]

    def test_rewards_hallucinated_answers(self) -> None:
        checker_text = """
[Beginning of Factuality Prediction]
NO
[End of Factuality Prediction]

[Beginning of Factual Errors]
Hamlet was written by Charles Dickens.
[End of Factual Errors]
"""
        result = asyncio.run(
            self._server(checker_text).verify(
                _make_request("Hamlet was written by Charles Dickens.")
            )
        )
        assert result.reward == 0.0
        assert result.hallucinated is True
        assert result.checker_prediction == "NO"
        assert result.checker_num_errors == 1
        assert result.checker_response_debug["id"] == "checker_resp"

    def test_parse_failures_fall_back_to_default_reward(self) -> None:
        result = asyncio.run(
            self._server("Unstructured checker output").verify(
                _make_request("Hamlet was written by William Shakespeare.")
            )
        )
        assert result.reward == 1.0
        assert result.hallucinated is False
        assert result.checker_prediction == ""
        assert result.checker_response_debug["id"] == "checker_resp"

    def test_string_error_payload_does_not_crash(self) -> None:
        result = asyncio.run(
            self._server_with_payload(
                "ClientResponseError(status=400, message='bad request')"
            ).verify(_make_request("Hamlet was written by William Shakespeare."))
        )
        assert result.reward == 1.0
        assert result.hallucinated is False
        assert result.checker_prediction == ""
        assert (
            result.checker_response_text
            == "ClientResponseError(status=400, message='bad request')"
        )
        assert result.checker_response_debug["error_type"] == "ValidationError"
